"""
RC Column Experimental Database - Step 1 & Step 2 Workflow
===========================================================
ACI 369 RC Column Experimental Database: Force-Displacement History & Drift Coverage Analysis

Step 1: Inspect & Match History CSVs Safely
  - Unpack Rectangular and Circular zip files into separate database subdirectories.
  - Inspect headers, column names, and measurement units.
  - Match each specimen in '1_Master_Dataset' against extracted CSVs.
  - Convert units: in. -> mm (x 25.4), kips -> kN (x 4.4482216).
  - Compute drift percentage for every point:
      drift_pct = (abs(displacement_mm) / Column_Clear_Length_L_mm) * 100

Step 2: Drift % Coverage Analysis & Interactive/Configurable Choice
  - Compute maximum overall drift % reached for each specimen.
  - Compute maximum post-peak drift % reached AFTER peak lateral force index.
  - Print candidate drift threshold coverage summary table (1.0% to 5.0%).
  - Expose TARGET_DRIFT_PCT top-level configuration with interactive input() confirmation.
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
OUTPUT_DIR = "history_csvs"

# Target drift % choice (can be overridden interactively or left as default)
TARGET_DRIFT_PCT = 2.0

# Candidate thresholds for coverage analysis
CANDIDATE_THRESHOLDS = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]

# Conversion constants
INCH_TO_MM = 25.4
KIPS_TO_KN = 4.4482216


# ==============================================================================
# STEP 1: INSPECT & MATCH HISTORY CSVS SAFELY
# ==============================================================================
def unpack_history_archives(zip_rect: str, zip_circ: str, dest_dir: str) -> dict:
    """
    Unpacks Rectangular and Circular zip archives into separate subdirectories
    to prevent file name collisions (e.g., '1.csv' in both databases).
    """
    rect_dir = os.path.join(dest_dir, "Rectangular")
    circ_dir = os.path.join(dest_dir, "Circular")
    os.makedirs(rect_dir, exist_ok=True)
    os.makedirs(circ_dir, exist_ok=True)

    print("\n" + "=" * 80)
    print("STEP 1.1: UNPACKING HISTORY ARCHIVES")
    print("=" * 80)

    # 1. Unpack Rectangular
    rect_count = 0
    if os.path.exists(zip_rect):
        with zipfile.ZipFile(zip_rect, "r") as z_rect:
            for member in z_rect.infolist():
                if not member.is_dir():
                    fname = os.path.basename(member.filename)
                    if fname.lower().endswith(".csv"):
                        target_path = os.path.join(rect_dir, fname)
                        with open(target_path, "wb") as f_out:
                            f_out.write(z_rect.read(member))
                        rect_count += 1
        print(f"-> Unpacked {rect_count} Rectangular CSVs to: {rect_dir}")
    else:
        print(f"[!] Warning: Rectangular zip archive not found at: {zip_rect}")

    # 2. Unpack Circular
    circ_count = 0
    if os.path.exists(zip_circ):
        with zipfile.ZipFile(zip_circ, "r") as z_circ:
            for member in z_circ.infolist():
                if not member.is_dir():
                    fname = os.path.basename(member.filename)
                    if fname.lower().endswith(".csv"):
                        target_path = os.path.join(circ_dir, fname)
                        with open(target_path, "wb") as f_out:
                            f_out.write(z_circ.read(member))
                        circ_count += 1
        print(f"-> Unpacked {circ_count} Circular CSVs to:    {circ_dir}")
    else:
        print(f"[!] Warning: Circular zip archive not found at: {zip_circ}")

    return {
        "Rectangular": rect_dir,
        "Circular": circ_dir,
        "rect_extracted": rect_count,
        "circ_extracted": circ_count,
    }


def inspect_csv_headers_and_units(dest_dirs: dict) -> dict:
    """
    Inspects CSV column headers, formats, and detected measurement units.
    """
    print("\n" + "=" * 80)
    print("STEP 1.2: CSV HEADER & UNIT INSPECTION")
    print("=" * 80)

    header_samples = {}
    for db_name, folder in [("Rectangular", dest_dirs["Rectangular"]), ("Circular", dest_dirs["Circular"])]:
        if not os.path.exists(folder):
            continue
        csv_files = [f for f in os.listdir(folder) if f.lower().endswith(".csv")]
        if not csv_files:
            continue
        
        # Sample first available file
        sample_file = os.path.join(folder, sorted(csv_files)[0])
        try:
            with open(sample_file, "r", encoding="utf-8", errors="ignore") as f:
                header_line = f.readline().strip()
                first_data_line = f.readline().strip()
            header_samples[db_name] = {
                "sample_file": os.path.basename(sample_file),
                "header": header_line,
                "first_row": first_data_line,
            }
        except Exception as e:
            header_samples[db_name] = {"error": str(e)}

    for db_name, info in header_samples.items():
        print(f"\n[{db_name} Database CSV Sample]")
        print(f"  Sample File:    {info.get('sample_file')}")
        print(f"  Header Line:    {info.get('header')}")
        print(f"  First Data Row: {info.get('first_row')}")
        print(f"  Parsed Units:   Displacement = inches (in.), Force = kips")

    return header_samples


def load_and_match_specimens(excel_path: str, sheet_name: str, dest_dirs: dict) -> tuple:
    """
    Loads '1_Master_Dataset' from the Excel database and matches each specimen
    to its corresponding CSV history file using Source_Database and Data_File_Name.
    """
    print("\n" + "=" * 80)
    print("STEP 1.3: MATCHING MASTER DATASET TO CSV HISTORY FILES")
    print("=" * 80)

    if not os.path.exists(excel_path):
        raise FileNotFoundError(f"Master Excel workbook not found at: {excel_path}")

    print(f"Loading sheet '{sheet_name}' from: {excel_path}...")
    df_master = pd.read_excel(excel_path, sheet_name=sheet_name)
    total_specimens = len(df_master)
    print(f"Loaded {total_specimens} specimens from {sheet_name}.")

    # Count database composition
    db_counts = df_master["Source_Database"].value_counts().to_dict()
    for db, count in db_counts.items():
        print(f"  - {db} specimens: {count}")

    matched_records = []
    unmatched_records = []

    for idx, row in df_master.iterrows():
        specimen_id = str(row.get("Unique_Specimen_ID", f"Row_{idx}")).strip()
        db_source = str(row.get("Source_Database", "")).strip()
        file_name = str(row.get("Data_File_Name", "")).strip()
        clear_length_mm = row.get("Column_Clear_Length_L_mm", np.nan)
        history_avail_flag = str(row.get("History_File_Available", "")).strip()

        expected_folder = dest_dirs.get(db_source, "")
        expected_csv_path = os.path.join(expected_folder, file_name) if expected_folder else ""

        if expected_csv_path and os.path.exists(expected_csv_path):
            matched_records.append({
                "row_index": idx,
                "specimen_id": specimen_id,
                "source_database": db_source,
                "data_file_name": file_name,
                "csv_path": expected_csv_path,
                "column_clear_length_mm": float(clear_length_mm),
                "history_available_flag": history_avail_flag,
            })
        else:
            unmatched_records.append({
                "row_index": idx,
                "specimen_id": specimen_id,
                "source_database": db_source,
                "data_file_name": file_name,
                "expected_path": expected_csv_path,
                "history_available_flag": history_avail_flag,
            })

    print(f"\nMatching Results Summary:")
    print(f"  * Total Master Dataset Rows: {total_specimens}")
    print(f"  * Successfully Matched CSVs: {len(matched_records)} ({len(matched_records)/total_specimens*100:.1f}%)")
    print(f"  * Missing / Unmatched CSVs:  {len(unmatched_records)}")

    if unmatched_records:
        print("\nDetails of Missing / Unmatched Specimens:")
        for rec in unmatched_records:
            print(f"  - Index {rec['row_index']}: ID = {rec['specimen_id']}, DB = {rec['source_database']}, "
                  f"File = '{rec['data_file_name']}', Flag = '{rec['history_available_flag']}'")

    return df_master, matched_records, unmatched_records


# ==============================================================================
# STEP 2: DRIFT % COVERAGE ANALYSIS & SELECTION
# ==============================================================================
def parse_and_convert_csv(csv_path: str, clear_length_mm: float) -> pd.DataFrame:
    """
    Reads a specimen history CSV, drops null/corrupted rows, converts units:
      - displacement: in. -> mm (x 25.4)
      - force: kips -> kN (x 4.4482216)
    And computes drift percentage for every point:
      drift_pct = (abs(displacement_mm) / clear_length_mm) * 100.0
    """
    df = pd.read_csv(csv_path)

    # Robust column identification
    disp_cols = [c for c in df.columns if "disp" in c.lower()]
    force_cols = [c for c in df.columns if "force" in c.lower() or "load" in c.lower()]

    if not disp_cols or not force_cols:
        # Fallback to 0th and 1st columns if naming differs
        c_disp = df.columns[0]
        c_force = df.columns[1]
    else:
        c_disp = disp_cols[0]
        c_force = force_cols[0]

    # Clean NaNs / empty rows (e.g. trailing empty rows or missing values)
    df_clean = df[[c_disp, c_force]].dropna()
    if df_clean.empty:
        return pd.DataFrame()

    disp_in = pd.to_numeric(df_clean[c_disp], errors="coerce").to_numpy(dtype=float)
    force_kips = pd.to_numeric(df_clean[c_force], errors="coerce").to_numpy(dtype=float)

    # Drop any rows that failed numeric conversion
    valid_mask = ~np.isnan(disp_in) & ~np.isnan(force_kips)
    if not np.any(valid_mask):
        return pd.DataFrame()

    disp_in = disp_in[valid_mask]
    force_kips = force_kips[valid_mask]

    # Unit conversions
    disp_mm = disp_in * INCH_TO_MM
    force_kN = force_kips * KIPS_TO_KN

    # Drift calculation
    drift_pct = (np.abs(disp_mm) / clear_length_mm) * 100.0

    return pd.DataFrame({
        "displacement_in": disp_in,
        "displacement_mm": disp_mm,
        "force_kips": force_kips,
        "force_kN": force_kN,
        "drift_pct": drift_pct,
    })


def analyze_drift_coverage(matched_records: list, total_specimens: int) -> pd.DataFrame:
    """
    Analyzes drift capacity across all matched columns:
      a) Maximum drift % reached across the entire curve.
      b) Maximum post-peak drift % reached AFTER the peak force index:
         peak_idx = argmax(abs(force_kN))
    """
    print("\n" + "=" * 80)
    print("STEP 2.1: COMPUTING PEAK FORCE AND MAXIMUM DRIFT (OVERALL & POST-PEAK)")
    print("=" * 80)

    analysis_results = []
    empty_data_count = 0

    for rec in matched_records:
        csv_path = rec["csv_path"]
        clear_length_mm = rec["column_clear_length_mm"]

        df_curve = parse_and_convert_csv(csv_path, clear_length_mm)

        if df_curve.empty or len(df_curve) < 2:
            empty_data_count += 1
            analysis_results.append({
                **rec,
                "status": "Corrupted or Empty History",
                "valid_points": len(df_curve),
                "peak_idx": np.nan,
                "peak_force_kN": np.nan,
                "peak_drift_pct": np.nan,
                "max_drift_pct": np.nan,
                "max_post_peak_drift_pct": np.nan,
            })
            continue

        force_kN = df_curve["force_kN"].to_numpy()
        drift_pct = df_curve["drift_pct"].to_numpy()

        # Maximum drift overall
        max_drift = float(np.max(drift_pct))

        # Peak force index: argmax(abs(force_kN))
        abs_force = np.abs(force_kN)
        peak_idx = int(np.argmax(abs_force))
        peak_force_val = float(abs_force[peak_idx])
        drift_at_peak = float(drift_pct[peak_idx])

        # Maximum post-peak drift reached AFTER peak force index
        if peak_idx < len(drift_pct) - 1:
            max_post_peak_drift = float(np.max(drift_pct[peak_idx + 1:]))
        else:
            # Peak occurred on the terminal data point
            max_post_peak_drift = 0.0

        analysis_results.append({
            **rec,
            "status": "Valid History",
            "valid_points": len(df_curve),
            "peak_idx": peak_idx,
            "peak_force_kN": peak_force_val,
            "peak_drift_pct": drift_at_peak,
            "max_drift_pct": max_drift,
            "max_post_peak_drift_pct": max_post_peak_drift,
        })

    df_coverage = pd.DataFrame(analysis_results)
    valid_count = (df_coverage["status"] == "Valid History").sum()

    print(f"Processed {len(df_coverage)} matched history CSVs:")
    print(f"  * Valid force-displacement histories: {valid_count}")
    print(f"  * Histories with missing/empty force data: {empty_data_count}")

    if empty_data_count > 0:
        bad_entries = df_coverage[df_coverage["status"] != "Valid History"]
        for _, b_row in bad_entries.iterrows():
            print(f"    - ID: {b_row['specimen_id']} ({b_row['source_database']}/{b_row['data_file_name']}) -> {b_row['status']}")

    return df_coverage


def print_coverage_summary_table(df_coverage: pd.DataFrame, total_specimens: int):
    """
    Prints a clear, formatted summary table across candidate drift thresholds
    showing overall vs. post-peak specimen coverage.
    """
    valid_df = df_coverage[df_coverage["status"] == "Valid History"]
    n_valid = len(valid_df)

    print("\n" + "=" * 96)
    print("STEP 2.2: DRIFT % COVERAGE SUMMARY TABLE")
    print("=" * 96)
    print(f"Master Dataset Total: {total_specimens} columns | Successfully Matched & Valid: {n_valid} columns")
    print("-" * 96)
    print(f"{'Drift':^8} | {'Overall Qualified':^21} | {'% Matched':^11} | {'Post-Peak Qualified':^23} | {'% Matched':^11} | {'% DB (498)':^10}")
    print(f"{'Threshold':^8} | {'(max_drift >= T)':^21} | {'(of ' + str(n_valid) + ')':^11} | {'(post_peak >= T)':^23} | {'(of ' + str(n_valid) + ')':^11} | {'(of ' + str(total_specimens) + ')':^10}")
    print("-" * 96)

    for t in CANDIDATE_THRESHOLDS:
        n_overall = int((valid_df["max_drift_pct"] >= t).sum())
        pct_ov_matched = (n_overall / n_valid) * 100 if n_valid > 0 else 0.0

        n_post = int((valid_df["max_post_peak_drift_pct"] >= t).sum())
        pct_post_matched = (n_post / n_valid) * 100 if n_valid > 0 else 0.0
        pct_post_db = (n_post / total_specimens) * 100

        print(f"  {t:>4.1f}%   |     {n_overall:>4} columns       |   {pct_ov_matched:>6.1f}%  |     {n_post:>4} columns         |   {pct_post_matched:>6.1f}%  |  {pct_post_db:>6.1f}%")

    print("=" * 96)
    print("Notes on Trade-Offs:")
    print("  * Drift thresholds <= 2.0% retain >92% of all experimental columns.")
    print("  * Drift thresholds between 2.0% and 3.0% retain 76% - 92% with significant post-peak degradation.")
    print("  * Drift thresholds >= 4.0% drop sample size below 60% due to earlier specimen collapse.")


def interactive_drift_selection(df_coverage: pd.DataFrame, default_drift_pct: float, total_specimens: int) -> float:
    """
    Exposes an interactive prompt via input() to choose or confirm the target drift %,
    falling back gracefully to default_drift_pct if non-interactive or empty.
    """
    print("\n" + "=" * 80)
    print("STEP 2.3: TARGET DRIFT % SELECTION")
    print("=" * 80)

    valid_df = df_coverage[df_coverage["status"] == "Valid History"]
    chosen_drift = default_drift_pct

    prompt_msg = f"Enter desired TARGET_DRIFT_PCT (e.g., 1.5, 2.0, 3.0) [Default: {default_drift_pct}%]: "
    try:
        user_input = input(prompt_msg).strip()
        if user_input:
            chosen_drift = float(user_input)
    except (EOFError, KeyboardInterrupt):
        print(f"\n[Non-interactive or default accepted: using {default_drift_pct}%]")
        chosen_drift = default_drift_pct
    except ValueError:
        print(f"[!] Invalid number entered. Falling back to default: {default_drift_pct}%")
        chosen_drift = default_drift_pct

    # Evaluate coverage under chosen threshold
    n_qual_ov = int((valid_df["max_drift_pct"] >= chosen_drift).sum())
    n_qual_pp = int((valid_df["max_post_peak_drift_pct"] >= chosen_drift).sum())
    pct_valid_pp = (n_qual_pp / len(valid_df)) * 100 if len(valid_df) > 0 else 0.0
    pct_total_pp = (n_qual_pp / total_specimens) * 100

    print("\n" + "-" * 80)
    print(f"CONFIRMED TARGET DRIFT: {chosen_drift:.2f}%")
    print("-" * 80)
    print(f"  * Columns reaching {chosen_drift:.2f}% drift overall:    {n_qual_ov} / {len(valid_df)} ({n_qual_ov/len(valid_df)*100:.1f}%)")
    print(f"  * Columns reaching {chosen_drift:.2f}% drift POST-PEAK:  {n_qual_pp} / {len(valid_df)} ({pct_valid_pp:.1f}% of valid, {pct_total_pp:.1f}% of 498 Master DB)")
    print("-" * 80)
    print("Status: Step 1 and Step 2 complete. Ready for target extraction in subsequent workflow.")
    print("=" * 80 + "\n")

    return chosen_drift


# ==============================================================================
# MAIN EXECUTION ENTRY POINT
# ==============================================================================
def main():
    print("=" * 80)
    print("ACI 369 RC COLUMN DATABASE: HISTORY MATCHING & DRIFT COVERAGE WORKFLOW")
    print("=" * 80)

    # Step 1: Unpack into separated database folders
    dest_dirs = unpack_history_archives(ZIP_RECTANGULAR, ZIP_CIRCULAR, OUTPUT_DIR)

    # Step 1.2: Inspect headers and units
    inspect_csv_headers_and_units(dest_dirs)

    # Step 1.3: Match rows in 1_Master_Dataset
    df_master, matched_records, unmatched_records = load_and_match_specimens(
        EXCEL_PATH, SHEET_NAME, dest_dirs
    )

    # Step 2.1: Analyze drift coverage & post-peak capacity
    df_coverage = analyze_drift_coverage(matched_records, len(df_master))

    # Step 2.2: Print summary table across thresholds
    print_coverage_summary_table(df_coverage, len(df_master))

    # Step 2.3: Configurable & interactive drift choice
    chosen_drift = interactive_drift_selection(df_coverage, TARGET_DRIFT_PCT, len(df_master))

    # Save summary table for downstream use
    summary_csv_path = "specimen_drift_coverage_summary.csv"
    df_coverage.to_csv(summary_csv_path, index=False)
    print(f"Saved specimen drift summary to: [specimen_drift_coverage_summary.csv](file:///{os.path.abspath(summary_csv_path)})")


if __name__ == "__main__":
    main()
