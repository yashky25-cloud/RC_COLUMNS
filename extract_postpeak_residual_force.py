"""
RC Column Database - Post-Peak Residual Force Extraction Workflow
==================================================================
ACI 369 RC Column Experimental Database: Extraction of Post-Peak Residual Force (kN)
at a Configurable Target Drift Threshold (TARGET_DRIFT_PCT = 2.0% by default).

This script builds directly upon the Step 1 & Step 2 setup:
  - Preserves 498 specimens in '1_Master_Dataset' (326 Rectangular, 172 Circular).
  - Uses history CSVs mapped via Source_Database and Data_File_Name.
  - Converts units: in. -> mm (x 25.4), kips -> kN (x 4.4482216).
  - Calculates drift: drift_pct = (abs(displacement_mm) / Column_Clear_Length_L_mm) * 100
  - Locates peak force index: peak_idx = argmax(abs(force_kN))
  - Focuses strictly on the post-peak portion: drift_pct[peak_idx:] and abs(force_kN[peak_idx:])
  - Extracts residual force via interpolation (np.interp) after ascending sort if max_post_peak >= TARGET_DRIFT_PCT
  - Assigns NaN for specimens not reaching target drift post-peak or lacking history files
  - Exports merged dataset to CSV and Excel for downstream ML model training.
"""

import os
import sys
import zipfile
import numpy as np
import pandas as pd

# ==============================================================================
# TOP-LEVEL CONFIGURATION
# ==============================================================================
EXCEL_PATH = "ACI369_RC_Column_Experimental_Database (1).xlsx"
SHEET_NAME = "1_Master_Dataset"
ZIP_RECTANGULAR = "data_rectangular.zip"
ZIP_CIRCULAR = "data_circular.zip"
HISTORY_DIR = "history_csvs"

# Configurable Target Drift % (Default: 2.0%, covering 456 columns)
TARGET_DRIFT_PCT = 2.0

# Export file paths
OUTPUT_CSV_PATH = "dataset_with_postpeak_force.csv"
OUTPUT_EXCEL_PATH = "dataset_with_postpeak_force.xlsx"

# Physical unit conversion constants
INCH_TO_MM = 25.4
KIPS_TO_KN = 4.4482216


# ==============================================================================
# HELPER FUNCTIONS: ARCHIVE EXTRACTION & CSV PARSING
# ==============================================================================
def ensure_history_csvs_unpacked(dest_dir: str = HISTORY_DIR):
    """
    Ensures that history CSVs are extracted into separated folders
    'history_csvs/Rectangular' and 'history_csvs/Circular' to prevent filename collision.
    """
    rect_dir = os.path.join(dest_dir, "Rectangular")
    circ_dir = os.path.join(dest_dir, "Circular")
    os.makedirs(rect_dir, exist_ok=True)
    os.makedirs(circ_dir, exist_ok=True)

    # Check if rectangular already populated
    if not os.path.exists(rect_dir) or len(os.listdir(rect_dir)) < 100:
        if os.path.exists(ZIP_RECTANGULAR):
            print(f"Unpacking {ZIP_RECTANGULAR} -> {rect_dir}...")
            with zipfile.ZipFile(ZIP_RECTANGULAR, "r") as z:
                for member in z.infolist():
                    if not member.is_dir():
                        fname = os.path.basename(member.filename)
                        if fname.lower().endswith(".csv"):
                            with open(os.path.join(rect_dir, fname), "wb") as f_out:
                                f_out.write(z.read(member))

    # Check if circular already populated
    if not os.path.exists(circ_dir) or len(os.listdir(circ_dir)) < 100:
        if os.path.exists(ZIP_CIRCULAR):
            print(f"Unpacking {ZIP_CIRCULAR} -> {circ_dir}...")
            with zipfile.ZipFile(ZIP_CIRCULAR, "r") as z:
                for member in z.infolist():
                    if not member.is_dir():
                        fname = os.path.basename(member.filename)
                        if fname.lower().endswith(".csv"):
                            with open(os.path.join(circ_dir, fname), "wb") as f_out:
                                f_out.write(z.read(member))


def extract_residual_force_for_specimen(
    csv_path: str,
    column_clear_length_mm: float,
    target_drift_pct: float
) -> dict:
    """
    Parses a single column CSV history, converts units, computes drift percentage,
    locates peak lateral force, and interpolates post-peak residual force at target_drift_pct.

    Returns:
      dict with keys:
        - residual_force_kN: float or np.nan
        - peak_force_kN: float or np.nan
        - peak_drift_pct: float or np.nan
        - max_post_peak_drift_pct: float or np.nan
        - status: str
    """
    if not os.path.exists(csv_path) or pd.isna(column_clear_length_mm) or column_clear_length_mm <= 0:
        return {
            "residual_force_kN": np.nan,
            "peak_force_kN": np.nan,
            "peak_drift_pct": np.nan,
            "max_post_peak_drift_pct": np.nan,
            "status": "Missing CSV or Invalid Length",
        }

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        return {
            "residual_force_kN": np.nan,
            "peak_force_kN": np.nan,
            "peak_drift_pct": np.nan,
            "max_post_peak_drift_pct": np.nan,
            "status": f"Read Error: {e}",
        }

    # Identify displacement and force columns
    disp_cols = [c for c in df.columns if "disp" in c.lower()]
    force_cols = [c for c in df.columns if "force" in c.lower() or "load" in c.lower()]
    c_disp = disp_cols[0] if disp_cols else df.columns[0]
    c_force = force_cols[0] if force_cols else df.columns[1]

    # Clean rows with missing/corrupted values
    df_clean = df[[c_disp, c_force]].dropna()
    if len(df_clean) < 2:
        return {
            "residual_force_kN": np.nan,
            "peak_force_kN": np.nan,
            "peak_drift_pct": np.nan,
            "max_post_peak_drift_pct": np.nan,
            "status": "Insufficient / Empty Force Data",
        }

    disp_in = pd.to_numeric(df_clean[c_disp], errors="coerce").to_numpy(dtype=float)
    force_kips = pd.to_numeric(df_clean[c_force], errors="coerce").to_numpy(dtype=float)

    valid_mask = ~np.isnan(disp_in) & ~np.isnan(force_kips)
    if not np.any(valid_mask):
        return {
            "residual_force_kN": np.nan,
            "peak_force_kN": np.nan,
            "peak_drift_pct": np.nan,
            "max_post_peak_drift_pct": np.nan,
            "status": "No Valid Numeric Rows",
        }

    # Convert units: in. -> mm, kips -> kN
    disp_mm = disp_in[valid_mask] * INCH_TO_MM
    force_kN = force_kips[valid_mask] * KIPS_TO_KN
    abs_force_kN = np.abs(force_kN)

    # Compute drift percentage
    drift_pct = (np.abs(disp_mm) / float(column_clear_length_mm)) * 100.0

    # Locate peak lateral force index: peak_idx = argmax(abs(force_kN))
    peak_idx = int(np.argmax(abs_force_kN))
    peak_force_val = float(abs_force_kN[peak_idx])
    peak_drift_val = float(drift_pct[peak_idx])

    # Extract post-peak curve: from peak_idx onwards
    post_drift = drift_pct[peak_idx:]
    post_force = abs_force_kN[peak_idx:]

    max_post_peak = float(np.max(post_drift)) if len(post_drift) > 0 else 0.0

    # Check if the specimen qualifies by reaching target drift post-peak
    if max_post_peak >= target_drift_pct:
        # Sort post-peak drift in ascending order before np.interp
        sort_order = np.argsort(post_drift)
        sorted_drift = post_drift[sort_order]
        sorted_force = post_force[sort_order]

        # Linear interpolation of absolute lateral force at target_drift_pct
        residual_force_val = float(np.interp(target_drift_pct, sorted_drift, sorted_force))

        return {
            "residual_force_kN": residual_force_val,
            "peak_force_kN": peak_force_val,
            "peak_drift_pct": peak_drift_val,
            "max_post_peak_drift_pct": max_post_peak,
            "status": "Successfully Extracted",
        }
    else:
        # Does NOT reach TARGET_DRIFT_PCT post-peak -> set to NaN (no extrapolation)
        return {
            "residual_force_kN": np.nan,
            "peak_force_kN": peak_force_val,
            "peak_drift_pct": peak_drift_val,
            "max_post_peak_drift_pct": max_post_peak,
            "status": f"Max Post-Peak Drift ({max_post_peak:.2f}%) < Target ({target_drift_pct:.2f}%)",
        }


# ==============================================================================
# MAIN EXECUTION WORKFLOW
# ==============================================================================
def run_postpeak_extraction(
    excel_path: str = EXCEL_PATH,
    sheet_name: str = SHEET_NAME,
    target_drift_pct: float = TARGET_DRIFT_PCT,
    output_csv: str = OUTPUT_CSV_PATH,
    output_excel: str = OUTPUT_EXCEL_PATH,
):
    print("=" * 88)
    print("ACI 369 RC COLUMN DATABASE: POST-PEAK RESIDUAL FORCE EXTRACTION")
    print(f"Target Drift Threshold: TARGET_DRIFT_PCT = {target_drift_pct:.2f}%")
    print("=" * 88)

    # 1. Ensure archives are unpacked into separated directories
    ensure_history_csvs_unpacked(HISTORY_DIR)

    # 2. Load Master Dataset
    print(f"\nLoading '{sheet_name}' from: {excel_path}...")
    if not os.path.exists(excel_path):
        raise FileNotFoundError(f"Master workbook not found: {excel_path}")

    df_master = pd.read_excel(excel_path, sheet_name=sheet_name)
    total_specimens = len(df_master)
    print(f"Loaded {total_specimens} specimens ({len(df_master.columns)} columns).")

    # 3. Process every specimen and extract post-peak residual force
    print(f"\nExtracting post-peak residual force at {target_drift_pct:.2f}% drift...")
    extracted_forces = []
    drift_levels = []
    max_post_drifts = []
    peak_forces_hist = []
    statuses = []

    for idx, row in df_master.iterrows():
        db_source = str(row.get("Source_Database", "")).strip()
        data_file = str(row.get("Data_File_Name", "")).strip()
        clear_length = row.get("Column_Clear_Length_L_mm", np.nan)

        csv_path = os.path.join(HISTORY_DIR, db_source, data_file)
        res = extract_residual_force_for_specimen(csv_path, clear_length, target_drift_pct)

        extracted_forces.append(res["residual_force_kN"])
        drift_levels.append(target_drift_pct if not np.isnan(res["residual_force_kN"]) else np.nan)
        max_post_drifts.append(res["max_post_peak_drift_pct"])
        peak_forces_hist.append(res["peak_force_kN"])
        statuses.append(res["status"])

    # 4. Attach extracted columns to Master Dataset
    target_col_name = "Extracted_Residual_Force_at_TargetDrift_kN"
    df_master[target_col_name] = extracted_forces
    df_master["Target_Drift_Level_pct"] = drift_levels
    df_master["Max_Post_Peak_Drift_pct"] = max_post_drifts
    df_master["Peak_Force_from_History_kN"] = peak_forces_hist
    df_master["Extraction_Status"] = statuses

    # 5. Summary Statistics Verification
    s_target = df_master[target_col_name]
    valid_count = int(s_target.notna().sum())
    nan_count = int(s_target.isna().sum())

    print("\n" + "=" * 88)
    print("EXTRACTION SUMMARY & VERIFICATION")
    print("=" * 88)
    print(f"Total Database Specimens:       {total_specimens}")
    print(f"Successfully Extracted (>= {target_drift_pct:.1f}%): {valid_count} specimens ({valid_count/total_specimens*100:.1f}%)")
    print(f"Unassigned / NaN Rows:          {nan_count} specimens ({nan_count/total_specimens*100:.1f}%)")

    # Status breakdown
    print("\nExtraction Status Breakdown:")
    status_series = df_master["Extraction_Status"].value_counts()
    for st, count in status_series.items():
        print(f"  * {st:<45} : {count:>4} columns")

    # Descriptive statistics
    print("\nSummary Statistics of Extracted Residual Force (kN):")
    print("-" * 55)
    stats_df = pd.DataFrame({
        "Metric": ["Count", "Mean", "Std Dev", "Min", "25% (Q1)", "Median (50%)", "75% (Q3)", "Max"],
        "Value (kN)": [
            f"{s_target.count()}",
            f"{s_target.mean():.3f} kN",
            f"{s_target.std():.3f} kN",
            f"{s_target.min():.3f} kN",
            f"{s_target.quantile(0.25):.3f} kN",
            f"{s_target.median():.3f} kN",
            f"{s_target.quantile(0.75):.3f} kN",
            f"{s_target.max():.3f} kN",
        ]
    })
    print(stats_df.to_string(index=False))
    print("-" * 55)

    # Sample rows display
    display_cols = [
        "Unique_Specimen_ID",
        "Source_Database",
        "Column_Clear_Length_L_mm",
        "Peak_Strength_Vmax1_kN",
        target_col_name,
        "Max_Post_Peak_Drift_pct",
    ]
    print("\nSample Extracted Rows Preview (First 10 Valid Specimens):")
    preview_df = df_master[s_target.notna()][display_cols].head(10)
    print(preview_df.to_string(index=False))

    # 6. Export Resulting Dataset
    print("\n" + "=" * 88)
    print("EXPORTING MERGED DATASET")
    print("=" * 88)

    # Save to CSV
    df_master.to_csv(output_csv, index=False)
    print(f"  [+] Saved CSV:   {os.path.abspath(output_csv)}")

    # Save to Excel
    try:
        df_master.to_excel(output_excel, sheet_name=sheet_name, index=False)
        print(f"  [+] Saved Excel: {os.path.abspath(output_excel)}")
    except Exception as e:
        print(f"  [!] Note: Could not save Excel file ({e}). CSV is fully ready.")

    print("\nWorkflow completed successfully. Ready for ML model training.")
    print("=" * 88)

    return df_master


if __name__ == "__main__":
    run_postpeak_extraction()
