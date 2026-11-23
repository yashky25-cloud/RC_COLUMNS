"""
================================================================================
RC Column Post-Peak Lateral Resistance: Live Team Presentation & Demo Script
================================================================================
Project: Data-Driven Estimation of Post-Peak Lateral Resistance of RC Columns
         Considering Failure Modes.

This presentation script demonstrates live X -> Y predictions for post-peak
residual lateral force (kN at 2.0% lateral drift) across four ML models:
  1. Artificial Neural Network (ANN / MLPRegressor)
  2. K-Nearest Neighbors (KNN / KNeighborsRegressor)
  3. Decision Tree (DT / DecisionTreeRegressor)
  4. LightGBM (LGBMRegressor)
  plus Ensemble Mean.

Sections:
  - Part 1: Real experimental database specimens (Rectangular & Circular)
            covering Flexure, Flexure-shear, and Shear failure modes, with
            side-by-side comparison to actual experimental residual force.
  - Part 2: Custom "What-If" parametric demonstration showing how models
            respond to modifications in key design parameters (e.g., Axial Load,
            Stirrup Spacing, Concrete Strength).
  - Part 3: Formatted console output and export to 'outputs_ml/team_demo_predictions.csv'.
================================================================================
"""

import os
import sys
from typing import Dict, List, Any

import numpy as np
import pandas as pd

from predict_residual_force import (
    predict_postpeak_force,
    RECTANGULAR_FEATURES,
    CIRCULAR_FEATURES,
    DISPLAY_MODEL_NAMES
)

OUTPUT_CSV = os.path.join("outputs_ml", "team_demo_predictions.csv")
DATASET_PATH = "dataset_with_postpeak_force.csv"
TARGET_COL = "Extracted_Residual_Force_at_TargetDrift_kN"


def print_banner(title: str, char: str = "=", width: int = 88):
    print("\n" + char * width)
    print(f" {title} ".center(width, char))
    print(char * width)


def format_inputs_table(inputs: Dict[str, Any], shape: str) -> str:
    """Formats pre-test inputs into a compact 2-column key-value string."""
    lines = []
    items = list(inputs.items())
    half = (len(items) + 1) // 2
    for i in range(half):
        k1, v1 = items[i]
        col1 = f"  • {k1:<34}: {v1:>8.2f}" if isinstance(v1, (int, float)) else f"  • {k1:<34}: {str(v1):>8}"
        if i + half < len(items):
            k2, v2 = items[i + half]
            col2 = f"  • {k2:<34}: {v2:>8.2f}" if isinstance(v2, (int, float)) else f"  • {k2:<34}: {str(v2):>8}"
            lines.append(f"{col1}   |  {col2}")
        else:
            lines.append(col1)
    return "\n".join(lines)


def run_team_demo():
    print_banner("DATA-DRIVEN RC COLUMN POST-PEAK LATERAL RESISTANCE ESTIMATION", "=", 92)
    print(" Team Presentation & Live Inference Demonstration")
    print(" Target Metric: Post-Peak Lateral Resistance (kN) at 2.0% Lateral Drift")
    print(" Feature Space: Pre-test geometric dimensions, reinforcement, axial load & materials")
    print(" Models: ANN (MLP), KNN, Decision Tree, LightGBM (Fitted & Loaded via joblib)")
    print("=" * 92)

    # Verify dataset exists
    if not os.path.exists(DATASET_PATH):
        print(f"\n[ERROR] Dataset '{DATASET_PATH}' not found. Cannot run real specimens demo.")
        return

    df = pd.read_csv(DATASET_PATH)
    demo_records = []

    # ==========================================================================
    # PART 1: REAL EXPERIMENTAL SPECIMENS DEMO
    # ==========================================================================
    print_banner("PART 1: REAL SPECIMENS DEMO (RECTANGULAR & CIRCULAR ACROSS FAILURE MODES)", "-", 92)

    specimen_selections = [
        # Shape, Failure Mode, Target Specimen ID / Criteria
        ("Rectangular", "Flexure", "RECT_P037_No.1"),
        ("Rectangular", "Flexure-shear", "RECT_P001_HPRC19-32"),
        ("Rectangular", "Shear", "RECT_P001_HPRC10-63"),
        ("Circular", "Flexure", "CIRC_P083_Unit1"),
        ("Circular", "Flexure-shear", "CIRC_P038_Unit1"),
        ("Circular", "Shear", "CIRC_P068_UNIT4")
    ]

    for shape, mode, spec_id in specimen_selections:
        # Locate specimen
        match = df[(df["Source_Database"] == shape) & (df["Unique_Specimen_ID"] == spec_id)]
        if len(match) == 0:
            # Fallback by mode
            sub = df[(df["Source_Database"] == shape) & (df["Failure_Mode_Class_DB"] == mode) & (df[TARGET_COL].notna())]
            if len(sub) == 0:
                continue
            row = sub.iloc[0]
        else:
            row = match.iloc[0]

        features_list = RECTANGULAR_FEATURES if shape == "Rectangular" else CIRCULAR_FEATURES
        spec_name = row.get("Original_Specimen_Name", "Unknown")
        uid = row.get("Unique_Specimen_ID", spec_id)
        paper_id = row.get("Paper_ID", "N/A")
        actual_force = float(row[TARGET_COL]) if pd.notna(row.get(TARGET_COL)) else np.nan
        peak_force = float(row.get("Peak_Strength_Vmax1_kN", np.nan))

        # Extract X inputs
        input_dict = {feat: float(row[feat]) for feat in features_list}

        # Predict using saved pipelines
        preds = predict_postpeak_force(shape, input_dict)

        print(f"\n>>> Specimen: {spec_name} [{uid}] | Shape: {shape} | Failure Mode: {mode} (Paper: {paper_id})")
        print("    Pre-Test Physical Design Inputs (X):")
        print(format_inputs_table(input_dict, shape))
        print("    " + "-" * 84)
        print(f"    {'Model':<22} | {'Predicted (kN)':<16} | {'Actual (kN)':<14} | {'Abs Error (kN)':<14} | {'% Error':<10}")
        print("    " + "-" * 84)

        record_entry = {
            "Demo_Type": "Real Specimen",
            "Shape": shape,
            "Specimen_ID": uid,
            "Original_Name": spec_name,
            "Paper_ID": paper_id,
            "Failure_Mode": mode,
            "Actual_Force_kN": round(actual_force, 2) if pd.notna(actual_force) else np.nan,
            "Peak_Force_kN": round(peak_force, 2) if pd.notna(peak_force) else np.nan,
            **input_dict
        }

        for m_name, pred_val in preds.items():
            clean_m = m_name.replace(" (kN)", "")
            err = abs(pred_val - actual_force) if pd.notna(actual_force) else np.nan
            pct_err = (err / actual_force) * 100.0 if (pd.notna(actual_force) and actual_force > 0) else np.nan

            record_entry[f"Pred_{clean_m}_kN"] = pred_val
            record_entry[f"AbsErr_{clean_m}_kN"] = round(err, 2) if pd.notna(err) else np.nan

            err_str = f"{err:>10.2f} kN" if pd.notna(err) else "       N/A"
            pct_str = f"{pct_err:>7.1f}%" if pd.notna(pct_err) else "    N/A"
            print(f"    {m_name:<22} | {pred_val:>12.2f} kN  | {actual_force:>10.2f} kN  | {err_str}   | {pct_str}")

        print("    " + "-" * 84)
        demo_records.append(record_entry)

    # ==========================================================================
    # PART 2: CUSTOM "WHAT-IF" SENSITIVITY DEMONSTRATION
    # ==========================================================================
    print_banner("PART 2: LIVE CUSTOM 'WHAT-IF' PARAMETRIC DEMO", "-", 92)
    print(" Modify key physical parameters in real-time to observe model response.")

    # 1. Custom Rectangular Column Baseline
    custom_rect_base = {
        "Column_Clear_Length_L_mm": 1200.0,
        "Section_Width_b_mm": 300.0,
        "Section_Depth_h_mm": 300.0,
        "rho_L_Longitudinal_Ratio": 0.022,
        "Axial_Load_P_kN": 450.0,
        "rho_t_Transverse_Volumetric_Ratio": 0.008,
        "Stirrup_Spacing_s_mm": 75.0,
        "fc_prime_MPa": 32.5,
        "fy_Longitudinal_MPa": 420.0,
        "fyt_Transverse_MPa": 360.0
    }

    # "What-If" Variation: Effect of Transverse Reinforcement Spacing (s = 75mm vs s = 150mm vs s = 40mm)
    print("\n[A] Rectangular Column What-If Scenario: Effect of Stirrup Spacing (s)")
    spacing_scenarios = [
        ("Well-Confined (s = 40 mm, rho_t = 1.5%)", 40.0, 0.015),
        ("Standard Design (s = 75 mm, rho_t = 0.8%)", 75.0, 0.008),
        ("Wide Spacing (s = 150 mm, rho_t = 0.4%)", 150.0, 0.004)
    ]

    print(f"    {'Scenario':<42} | {'ANN (kN)':<10} | {'KNN (kN)':<10} | {'DT (kN)':<10} | {'LightGBM':<10} | {'Ensemble':<10}")
    print("    " + "-" * 98)
    for desc, s_val, rhot_val in spacing_scenarios:
        case = custom_rect_base.copy()
        case["Stirrup_Spacing_s_mm"] = s_val
        case["rho_t_Transverse_Volumetric_Ratio"] = rhot_val
        res = predict_postpeak_force("Rectangular", case)
        print(f"    {desc:<42} | {res['ANN (kN)']:>8.2f}  | {res['KNN (kN)']:>8.2f}  | {res['Decision Tree (kN)']:>8.2f}  | {res['LightGBM (kN)']:>8.2f}  | {res['Ensemble Mean (kN)']:>8.2f}")

        demo_records.append({
            "Demo_Type": "What-If (Rectangular Confinement)",
            "Shape": "Rectangular",
            "Specimen_ID": desc,
            "Original_Name": "Custom Design",
            "Paper_ID": "Synthetic",
            "Failure_Mode": "User Scenario",
            "Actual_Force_kN": np.nan,
            "Peak_Force_kN": np.nan,
            **case,
            "Pred_ANN_kN": res['ANN (kN)'],
            "Pred_KNN_kN": res['KNN (kN)'],
            "Pred_Decision_Tree_kN": res['Decision Tree (kN)'],
            "Pred_LightGBM_kN": res['LightGBM (kN)'],
            "Pred_Ensemble_Mean_kN": res['Ensemble Mean (kN)']
        })
    print("    " + "-" * 98)

    # 2. Custom Circular Column Baseline
    custom_circ_base = {
        "Column_Clear_Length_L_mm": 1500.0,
        "Section_Diameter_D_mm": 400.0,
        "rho_L_Longitudinal_Ratio": 0.020,
        "Axial_Load_P_kN": 500.0,
        "rho_t_Transverse_Volumetric_Ratio": 0.010,
        "Stirrup_Spacing_s_mm": 60.0,
        "fc_prime_MPa": 35.0,
        "fy_Longitudinal_MPa": 450.0,
        "fyt_Transverse_MPa": 380.0
    }

    # "What-If" Variation: Effect of Axial Load Ratio (Low vs Medium vs High Axial Load)
    print("\n[B] Circular Column What-If Scenario: Effect of Axial Load (P)")
    axial_scenarios = [
        ("Low Axial Load (P = 200 kN)", 200.0),
        ("Moderate Axial Load (P = 600 kN)", 600.0),
        ("High Axial Load (P = 1200 kN)", 1200.0)
    ]

    print(f"    {'Scenario':<42} | {'ANN (kN)':<10} | {'KNN (kN)':<10} | {'DT (kN)':<10} | {'LightGBM':<10} | {'Ensemble':<10}")
    print("    " + "-" * 98)
    for desc, p_val in axial_scenarios:
        case = custom_circ_base.copy()
        case["Axial_Load_P_kN"] = p_val
        res = predict_postpeak_force("Circular", case)
        print(f"    {desc:<42} | {res['ANN (kN)']:>8.2f}  | {res['KNN (kN)']:>8.2f}  | {res['Decision Tree (kN)']:>8.2f}  | {res['LightGBM (kN)']:>8.2f}  | {res['Ensemble Mean (kN)']:>8.2f}")

        demo_records.append({
            "Demo_Type": "What-If (Circular Axial Load)",
            "Shape": "Circular",
            "Specimen_ID": desc,
            "Original_Name": "Custom Design",
            "Paper_ID": "Synthetic",
            "Failure_Mode": "User Scenario",
            "Actual_Force_kN": np.nan,
            "Peak_Force_kN": np.nan,
            **case,
            "Pred_ANN_kN": res['ANN (kN)'],
            "Pred_KNN_kN": res['KNN (kN)'],
            "Pred_Decision_Tree_kN": res['Decision Tree (kN)'],
            "Pred_LightGBM_kN": res['LightGBM (kN)'],
            "Pred_Ensemble_Mean_kN": res['Ensemble Mean (kN)']
        })
    print("    " + "-" * 98)

    # ==========================================================================
    # PART 3: SUMMARY & CSV EXPORT
    # ==========================================================================
    print_banner("PART 3: SUMMARY & EXPORT", "-", 92)
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    summary_df = pd.DataFrame(demo_records)
    summary_df.to_csv(OUTPUT_CSV, index=False)
    print(f" Saved full demo prediction records ({len(summary_df)} cases) -> {OUTPUT_CSV}")
    print(" Live demonstration script executed cleanly.")
    print("=" * 92 + "\n")


if __name__ == "__main__":
    run_team_demo()
