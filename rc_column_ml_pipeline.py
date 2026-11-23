"""
================================================================================
RC Column Post-Peak Lateral Resistance Estimation & Failure-Mode Analysis Pipeline
================================================================================
Project: Data-Driven Estimation of Post-Peak Lateral Resistance of RC Columns
         Considering Failure Modes.

This module implements a complete, clean, modular Python-based ML and failure-mode
analysis pipeline:
  1. Auto-detects the merged dataset file and target post-peak force column.
  2. Cleans "NR"/"NA" placeholders and converts numeric columns cleanly.
  3. Normalizes Failure_Mode_Class_DB (Flexure, Flexure-shear, Shear, and NaN).
  4. Performs strict no-leakage feature selection tailored for Rectangular and Circular
     shapes, formatted for direct MATLAB GUI integration.
  5. Conducts comprehensive post-peak force degradation and retention analysis across
     failure modes (force, retention ratio, and drop).
  6. Applies 80/20 train-test splitting via GroupShuffleSplit (grouped by Paper_ID)
     with transparent tracking of specimen counts per failure mode across splits.
  7. Preprocesses data via sklearn Pipeline with ColumnTransformer (median imputation,
     with scaling strictly applied to ANN/KNN and unscaled for DT/LightGBM).
  8. Trains and evaluates four regression models per shape:
     - Artificial Neural Network (MLPRegressor)
     - K-Nearest Neighbors (KNeighborsRegressor)
     - Decision Tree (DecisionTreeRegressor)
     - LightGBM (LGBMRegressor)
  9. Evaluates overall R2, RMSE, MAE and breaks down test performance by failure mode.
 10. Generates publication-ready visualizations and structured CSV summary tables
     saved in a dedicated output directory ('outputs_ml/').
================================================================================
"""

import os
import sys
import glob
import logging
from typing import Dict, List, Tuple, Optional, Any, Union

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless execution
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("RC_ML_Pipeline")

# ==============================================================================
# CONFIGURATION & CONSTANTS
# ==============================================================================
OUTPUT_DIR = "outputs_ml"
RANDOM_STATE = 42
TEST_SIZE = 0.20

# Shape-specific MATLAB GUI-friendly input feature lists
RECTANGULAR_FEATURES = [
    "Column_Clear_Length_L_mm",
    "Section_Width_b_mm",
    "Section_Depth_h_mm",
    "rho_L_Longitudinal_Ratio",
    "Axial_Load_P_kN",
    "rho_t_Transverse_Volumetric_Ratio",
    "Stirrup_Spacing_s_mm",
    "fc_prime_MPa",
    "fy_Longitudinal_MPa",
    "fyt_Transverse_MPa"
]

CIRCULAR_FEATURES = [
    "Column_Clear_Length_L_mm",
    "Section_Diameter_D_mm",
    "rho_L_Longitudinal_Ratio",
    "Axial_Load_P_kN",
    "rho_t_Transverse_Volumetric_Ratio",
    "Stirrup_Spacing_s_mm",
    "fc_prime_MPa",
    "fy_Longitudinal_MPa",
    "fyt_Transverse_MPa"
]

FAILURE_MODE_ORDER = ["Flexure", "Flexure-shear", "Shear"]
FAILURE_MODE_PALETTE = {
    "Flexure": "#2b5c8f",        # Steel Blue
    "Flexure-shear": "#d9822b",  # Amber / Orange
    "Shear": "#c0392b",          # Crimson Red
    "NR": "#7f8c8d",             # Grey
    "Unclassified": "#7f8c8d"
}


# ==============================================================================
# 1. DATASET AUTO-DETECTION & LOADING
# ==============================================================================
def auto_detect_dataset_and_target() -> Tuple[str, str, pd.DataFrame]:
    """
    Auto-detects the merged dataset CSV/Excel file and the extracted post-peak
    lateral force target column from previous extraction workflows.
    """
    candidate_files = [
        "dataset_with_postpeak_force.csv",
        "dataset_with_postpeak_force.xlsx",
        "dataset_with_extracted_force.csv",
        "specimen_drift_coverage_summary.csv"
    ]
    # Add any other CSV matching pattern
    other_csvs = glob.glob("*.csv") + glob.glob("*.xlsx")
    for f in other_csvs:
        if f not in candidate_files and ("postpeak" in f.lower() or "residual" in f.lower() or "force" in f.lower()):
            candidate_files.append(f)

    selected_file = None
    df = None
    for filepath in candidate_files:
        if os.path.exists(filepath):
            try:
                if filepath.endswith(".csv"):
                    temp_df = pd.read_csv(filepath)
                else:
                    temp_df = pd.read_excel(filepath)
                
                # Check for target column candidates
                target_candidates = [
                    "Extracted_Residual_Force_at_TargetDrift_kN",
                    "Extracted_Residual_Force_kN",
                    "Residual_Force_at_TargetDrift_kN",
                    "PostPeak_Residual_Force_kN",
                    "Residual_Strength_kN"
                ]
                found_target = None
                for col in target_candidates:
                    if col in temp_df.columns:
                        found_target = col
                        break
                
                if found_target is None:
                    # Fuzzy search
                    for col in temp_df.columns:
                        c_lower = col.lower()
                        if ("residual" in c_lower or "post" in c_lower) and "force" in c_lower:
                            found_target = col
                            break

                if found_target is not None:
                    selected_file = filepath
                    df = temp_df
                    target_column = found_target
                    break
            except Exception as e:
                logger.warning("Error reading %s: %s", filepath, e)
                continue

    if selected_file is None or df is None:
        raise FileNotFoundError("Could not auto-detect a dataset containing the post-peak force target column.")

    logger.info("=" * 70)
    logger.info("DATASET AUTO-DETECTION SUCCESSFUL")
    logger.info("  -> Detected File: %s", selected_file)
    logger.info("  -> Detected Target Column: %s", target_column)
    logger.info("  -> Total Rows in File: %d, Columns: %d", len(df), len(df.columns))
    logger.info("=" * 70)

    return selected_file, target_column, df


# ==============================================================================
# 2. DATA CLEANING & FAILURE MODE NORMALIZATION
# ==============================================================================
def normalize_failure_mode(val: Any) -> Optional[str]:
    """
    Normalizes raw failure mode strings into canonical categories:
    'Flexure', 'Flexure-shear', 'Shear', or np.nan for missing/NR.
    """
    if pd.isna(val):
        return np.nan
    s = str(val).strip().lower()
    if s in ["nr", "na", "n/a", "none", "", "nan", "null"]:
        return np.nan
    if "flexure-shear" in s or "flexure shear" in s or "flexural-shear" in s or "f-s" in s or "fs" in s:
        return "Flexure-shear"
    if "flexure" in s or "flexural" in s:
        return "Flexure"
    if "shear" in s:
        return "Shear"
    return "Flexure"  # default fallback if unknown


def clean_dataset(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """
    Cleans "NR", "NA", whitespace placeholders across dataset, converts numeric
    columns, normalizes Failure_Mode_Class_DB, and ensures clean types.
    """
    df_clean = df.copy()

    # Normalize Failure Mode
    if "Failure_Mode_Class_DB" in df_clean.columns:
        df_clean["Failure_Mode_Normalized"] = df_clean["Failure_Mode_Class_DB"].apply(normalize_failure_mode)
    else:
        df_clean["Failure_Mode_Normalized"] = np.nan

    # Identify numeric candidate columns
    numeric_cols = list(set(RECTANGULAR_FEATURES + CIRCULAR_FEATURES + [
        target_col,
        "Peak_Strength_Vmax1_kN",
        "Peak_Force_from_History_kN",
        "Residual_Strength_kN",
        "Drift_at_Vmax1_pct",
        "Target_Drift_Level_pct"
    ]))

    for col in numeric_cols:
        if col in df_clean.columns:
            # Replace placeholder strings
            df_clean[col] = df_clean[col].replace(["NR", "NA", "N/A", "None", "nan", " "], np.nan)
            df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")

    return df_clean


def summarize_target(df_shape: pd.DataFrame, shape_name: str, target_col: str) -> pd.DataFrame:
    """
    Computes and prints target summary statistics (count, min, 25%, median, mean, 75%, max, std).
    """
    target_series = df_shape[target_col].dropna()
    stats = {
        "Shape": shape_name,
        "Count": len(target_series),
        "Min (kN)": target_series.min(),
        "25% (kN)": target_series.quantile(0.25),
        "Median (kN)": target_series.median(),
        "Mean (kN)": target_series.mean(),
        "75% (kN)": target_series.quantile(0.75),
        "Max (kN)": target_series.max(),
        "Std (kN)": target_series.std()
    }
    stats_df = pd.DataFrame([stats])
    logger.info("Target Variable (%s) Summary for %s Columns:", target_col, shape_name)
    for k, v in stats.items():
        if isinstance(v, float):
            logger.info("  - %-15s: %10.3f", k, v)
        else:
            logger.info("  - %-15s: %10s", k, v)
    return stats_df


# ==============================================================================
# 3. POST-PEAK LATERAL FORCE DEGRADATION STUDY
# ==============================================================================
def perform_degradation_study(df_shape: pd.DataFrame, shape_name: str, target_col: str, output_dir: str) -> pd.DataFrame:
    """
    Studies post-peak lateral force degradation across failure modes:
      - Post-Peak Force at target drift (kN)
      - Force Retention Ratio = target_force_kN / Peak_Strength_Vmax1_kN
      - Force Drop = Peak_Strength_Vmax1_kN - target_force_kN
    Calculates count, median, mean, min, max, std, and IQR per failure mode.
    """
    study_df = df_shape.copy()
    
    # Peak lateral strength reference
    peak_col = "Peak_Strength_Vmax1_kN" if "Peak_Strength_Vmax1_kN" in study_df.columns else "Peak_Force_from_History_kN"
    study_df["Peak_Force_kN"] = pd.to_numeric(study_df[peak_col], errors="coerce")
    
    # Analytical degradation metrics
    study_df["Retention_Ratio"] = study_df[target_col] / study_df["Peak_Force_kN"]
    study_df["Force_Drop_kN"] = study_df["Peak_Force_kN"] - study_df[target_col]
    study_df["Relative_Drop_pct"] = (study_df["Force_Drop_kN"] / study_df["Peak_Force_kN"]) * 100.0

    # Fill display label for failure mode
    study_df["Display_Mode"] = study_df["Failure_Mode_Normalized"].fillna("NR / Unspecified")

    results = []
    modes_to_evaluate = ["Flexure", "Flexure-shear", "Shear", "NR / Unspecified"]
    for mode in modes_to_evaluate:
        sub = study_df[study_df["Display_Mode"] == mode]
        if len(sub) == 0:
            continue
        
        force_sub = sub[target_col].dropna()
        ret_sub = sub["Retention_Ratio"].dropna()
        drop_sub = sub["Force_Drop_kN"].dropna()
        
        def calc_stats(s: pd.Series):
            if len(s) == 0:
                return (0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)
            iqr = s.quantile(0.75) - s.quantile(0.25)
            return (len(s), s.mean(), s.median(), s.std(), s.min(), s.max(), iqr)

        f_cnt, f_mean, f_med, f_std, f_min, f_max, f_iqr = calc_stats(force_sub)
        r_cnt, r_mean, r_med, r_std, r_min, r_max, r_iqr = calc_stats(ret_sub)
        d_cnt, d_mean, d_med, d_std, d_min, d_max, d_iqr = calc_stats(drop_sub)

        results.append({
            "Shape": shape_name,
            "Failure_Mode": mode,
            "Count": f_cnt,
            "Force_Mean_kN": f_mean,
            "Force_Median_kN": f_med,
            "Force_Std_kN": f_std,
            "Force_Min_kN": f_min,
            "Force_Max_kN": f_max,
            "Force_IQR_kN": f_iqr,
            "Retention_Ratio_Mean": r_mean,
            "Retention_Ratio_Median": r_med,
            "Retention_Ratio_Std": r_std,
            "Retention_Ratio_Min": r_min,
            "Retention_Ratio_Max": r_max,
            "Retention_Ratio_IQR": r_iqr,
            "Force_Drop_Mean_kN": d_mean,
            "Force_Drop_Median_kN": d_med,
            "Force_Drop_Std_kN": d_std,
            "Force_Drop_Min_kN": d_min,
            "Force_Drop_Max_kN": d_max,
            "Force_Drop_IQR_kN": d_iqr
        })

    summary_df = pd.DataFrame(results)
    out_csv = os.path.join(output_dir, f"degradation_summary_{shape_name.lower()}.csv")
    summary_df.to_csv(out_csv, index=False)
    logger.info("Saved failure-mode degradation summary -> %s", out_csv)

    return summary_df


# ==============================================================================
# 4. TRAIN-TEST SPLITTING (GROUP SHUFFLE SPLIT BY PAPER_ID)
# ==============================================================================
def split_dataset_by_paper(
    df_shape: pd.DataFrame,
    shape_name: str,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits the shape-specific dataset into 80% train and 20% test using GroupShuffleSplit
    grouped by Paper_ID to prevent laboratory/paper data leakage.
    Provides transparent counts across failure modes.
    """
    has_paper_id = ("Paper_ID" in df_shape.columns) and (df_shape["Paper_ID"].notna().sum() > 0)
    
    if has_paper_id:
        logger.info("Splitting %s columns via GroupShuffleSplit on 'Paper_ID' (test_size=%.2f, random_state=%d)...",
                    shape_name, test_size, random_state)
        # Fill any missing Paper_ID with individual IDs so they don't break grouping
        papers = df_shape["Paper_ID"].fillna(df_shape.index.to_series().apply(lambda x: f"PAPER_MISSING_{x}"))
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        try:
            train_idx, test_idx = next(gss.split(df_shape, groups=papers))
            train_df = df_shape.iloc[train_idx].copy()
            test_df = df_shape.iloc[test_idx].copy()
        except Exception as e:
            logger.warning("GroupShuffleSplit failed (%s). Falling back to train_test_split.", e)
            train_df, test_df = train_test_split(df_shape, test_size=test_size, random_state=random_state)
    else:
        logger.warning("'Paper_ID' missing or unavailable. Falling back to train_test_split.")
        train_df, test_df = train_test_split(df_shape, test_size=test_size, random_state=random_state)

    # Specimen count verification table
    def count_modes(df_subset: pd.DataFrame, subset_name: str) -> Dict[str, Any]:
        modes = df_subset["Failure_Mode_Normalized"].fillna("NR")
        counts = modes.value_counts().to_dict()
        return {
            "Subset": subset_name,
            "Total": len(df_subset),
            "Flexure": counts.get("Flexure", 0),
            "Flexure-shear": counts.get("Flexure-shear", 0),
            "Shear": counts.get("Shear", 0),
            "NR / Missing": counts.get("NR", 0)
        }

    count_rows = [
        count_modes(df_shape, f"Overall ({shape_name})"),
        count_modes(train_df, f"Train (80%)"),
        count_modes(test_df, f"Test (20%)")
    ]
    split_summary_df = pd.DataFrame(count_rows)
    logger.info("Split Specimen Counts for %s:\n%s", shape_name, split_summary_df.to_string(index=False))

    return train_df, test_df, split_summary_df


# ==============================================================================
# 5. PREPROCESSING PIPELINES & MODEL DEFINITIONS
# ==============================================================================
def create_model_pipelines(features: List[str]) -> Dict[str, Pipeline]:
    """
    Builds scikit-learn Pipelines with strict no-leakage preprocessing:
      - ANN (MLPRegressor): Median Imputation + StandardScaler
      - KNN (KNeighborsRegressor): Median Imputation + StandardScaler
      - Decision Tree (DecisionTreeRegressor): Median Imputation without scaling
      - LightGBM (LGBMRegressor): Median Imputation without scaling
    """
    # 1. Artificial Neural Network (MLP)
    pipe_ann = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("regressor", MLPRegressor(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            alpha=0.01,
            learning_rate_init=0.01,
            max_iter=1500,
            early_stopping=True,
            n_iter_no_change=20,
            random_state=RANDOM_STATE
        ))
    ])

    # 2. K-Nearest Neighbors
    pipe_knn = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("regressor", KNeighborsRegressor(
            n_neighbors=5,
            weights="distance",
            metric="minkowski",
            p=2
        ))
    ])

    # 3. Decision Tree Regressor
    pipe_dt = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("regressor", DecisionTreeRegressor(
            max_depth=5,
            min_samples_split=6,
            min_samples_leaf=4,
            random_state=RANDOM_STATE
        ))
    ])

    # 4. LightGBM Regressor
    pipe_lgbm = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("regressor", LGBMRegressor(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.03,
            min_child_samples=5,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=RANDOM_STATE,
            verbose=-1
        ))
    ])

    return {
        "ANN": pipe_ann,
        "KNN": pipe_knn,
        "Decision Tree": pipe_dt,
        "LightGBM": pipe_lgbm
    }


# ==============================================================================
# 6. MODEL TRAINING, EVALUATION & FAILURE-MODE BREAKDOWN
# ==============================================================================
def train_and_evaluate_shape(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    features: List[str],
    target_col: str,
    shape_name: str,
    output_dir: str
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Trains all four regression models on the training set, predicts on test set,
    evaluates R2, RMSE, MAE (both overall and broken down by failure mode),
    and saves metric tables and full test predictions.
    """
    X_train = train_df[features]
    y_train = train_df[target_col]
    X_test = test_df[features]
    y_test = test_df[target_col]

    models = create_model_pipelines(features)
    
    saved_models_dir = os.path.join(output_dir, "saved_models")
    os.makedirs(saved_models_dir, exist_ok=True)

    metrics_list = []
    test_preds_df = test_df[[col for col in ["Unique_Specimen_ID", "Original_Specimen_Name", "Paper_ID", "Failure_Mode_Normalized", target_col] if col in test_df.columns]].copy()
    test_preds_df.rename(columns={target_col: "Actual_Residual_Force_kN"}, inplace=True)

    trained_models = {}

    for name, model in models.items():
        # Fit model on training split only
        model.fit(X_train, y_train)
        trained_models[name] = model

        # Save fitted pipeline to disk (joblib)
        file_safe_name = name.replace(" ", "_")
        model_filename = f"{shape_name.lower()}_{file_safe_name}.joblib"
        model_save_path = os.path.join(saved_models_dir, model_filename)
        joblib.dump(model, model_save_path)
        logger.info("Saved fitted pipeline -> %s", model_save_path)

        # Predict on train & test
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)

        # Store test predictions
        test_preds_df[f"Pred_{name}_kN"] = test_pred

        # Overall Metrics
        test_r2 = r2_score(y_test, test_pred)
        test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
        test_mae = mean_absolute_error(y_test, test_pred)

        train_r2 = r2_score(y_train, train_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        train_mae = mean_absolute_error(y_train, train_pred)

        metrics_list.append({
            "Shape": shape_name,
            "Model": name,
            "Train_R2": train_r2,
            "Train_RMSE_kN": train_rmse,
            "Train_MAE_kN": train_mae,
            "Test_R2": test_r2,
            "Test_RMSE_kN": test_rmse,
            "Test_MAE_kN": test_mae
        })

    metrics_df = pd.DataFrame(metrics_list)
    out_metrics_csv = os.path.join(output_dir, f"metrics_{shape_name.lower()}_80_20.csv")
    metrics_df.to_csv(out_metrics_csv, index=False)
    logger.info("Saved %s performance metrics -> %s", shape_name, out_metrics_csv)

    # Save test predictions table
    out_preds_csv = os.path.join(output_dir, f"predictions_{shape_name.lower()}_test.csv")
    test_preds_df.to_csv(out_preds_csv, index=False)
    logger.info("Saved test predictions table -> %s", out_preds_csv)

    # Failure-mode specific test breakdown
    failure_breakdown_rows = []
    modes = ["Flexure", "Flexure-shear", "Shear"]
    for mode in modes:
        mode_mask = (test_preds_df["Failure_Mode_Normalized"] == mode)
        mode_count = mode_mask.sum()
        
        for name in models.keys():
            if mode_count >= 3:
                y_true_m = test_preds_df.loc[mode_mask, "Actual_Residual_Force_kN"]
                y_pred_m = test_preds_df.loc[mode_mask, f"Pred_{name}_kN"]
                m_r2 = r2_score(y_true_m, y_pred_m) if mode_count > 2 else np.nan
                m_rmse = np.sqrt(mean_squared_error(y_true_m, y_pred_m))
                m_mae = mean_absolute_error(y_true_m, y_pred_m)
                note = "Sufficient test samples"
            elif mode_count > 0:
                y_true_m = test_preds_df.loc[mode_mask, "Actual_Residual_Force_kN"]
                y_pred_m = test_preds_df.loc[mode_mask, f"Pred_{name}_kN"]
                m_r2 = np.nan
                m_rmse = np.sqrt(mean_squared_error(y_true_m, y_pred_m))
                m_mae = mean_absolute_error(y_true_m, y_pred_m)
                note = f"Limited test samples (N={mode_count})"
            else:
                m_r2 = np.nan
                m_rmse = np.nan
                m_mae = np.nan
                note = "Zero test samples (all in training split by Paper_ID)"

            failure_breakdown_rows.append({
                "Shape": shape_name,
                "Model": name,
                "Failure_Mode": mode,
                "Test_Specimen_Count": mode_count,
                "Test_R2": m_r2,
                "Test_RMSE_kN": m_rmse,
                "Test_MAE_kN": m_mae,
                "Sampling_Note": note
            })

    failure_breakdown_df = pd.DataFrame(failure_breakdown_rows)
    out_fail_csv = os.path.join(output_dir, f"failure_mode_test_error_{shape_name.lower()}.csv")
    failure_breakdown_df.to_csv(out_fail_csv, index=False)
    logger.info("Saved failure-mode test error breakdown -> %s", out_fail_csv)

    return metrics_df, test_preds_df, failure_breakdown_df


# ==============================================================================
# 7. PUBLICATION-QUALITY VISUALIZATIONS
# ==============================================================================
def plot_model_metrics(metrics_rect: pd.DataFrame, metrics_circ: pd.DataFrame, output_dir: str):
    """
    Generates side-by-side bar charts comparing MAE, RMSE, and R2 across ANN, KNN,
    Decision Tree, and LightGBM for Rectangular and Circular columns.
    """
    sns.set_theme(style="whitegrid", font="sans-serif")
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), dpi=300)

    model_colors = ["#2b5c8f", "#27ae60", "#e67e22", "#8e44ad"]

    shapes_data = [
        ("Rectangular Columns", metrics_rect, axes[0]),
        ("Circular Columns", metrics_circ, axes[1])
    ]

    for title_prefix, df_met, row_axes in shapes_data:
        models = df_met["Model"].tolist()
        x = np.arange(len(models))

        # 1. R2 Score
        ax_r2 = row_axes[0]
        bars_r2 = ax_r2.bar(x, df_met["Test_R2"], color=model_colors, width=0.55, edgecolor="black", alpha=0.85)
        ax_r2.axhline(0, color="gray", linestyle="--", linewidth=0.8)
        ax_r2.set_xticks(x)
        ax_r2.set_xticklabels(models, fontweight="bold")
        ax_r2.set_ylabel("Test $R^2$ Score", fontsize=11, fontweight="bold")
        ax_r2.set_title(f"{title_prefix}: Test $R^2$", fontsize=12, fontweight="bold", pad=8)
        for bar in bars_r2:
            h = bar.get_height()
            pos = h + 0.03 if h >= 0 else h - 0.08
            ax_r2.annotate(f"{h:.2f}",
                            xy=(bar.get_x() + bar.get_width() / 2, pos),
                            ha="center", va="bottom" if h < 0 else "bottom",
                            fontsize=9, fontweight="bold")

        # 2. RMSE
        ax_rmse = row_axes[1]
        bars_rmse = ax_rmse.bar(x, df_met["Test_RMSE_kN"], color=model_colors, width=0.55, edgecolor="black", alpha=0.85)
        ax_rmse.set_xticks(x)
        ax_rmse.set_xticklabels(models, fontweight="bold")
        ax_rmse.set_ylabel("Test RMSE (kN)", fontsize=11, fontweight="bold")
        ax_rmse.set_title(f"{title_prefix}: Test RMSE (kN)", fontsize=12, fontweight="bold", pad=8)
        for bar in bars_rmse:
            h = bar.get_height()
            ax_rmse.annotate(f"{h:.1f}",
                             xy=(bar.get_x() + bar.get_width() / 2, h + 1.5),
                             ha="center", va="bottom", fontsize=9, fontweight="bold")

        # 3. MAE
        ax_mae = row_axes[2]
        bars_mae = ax_mae.bar(x, df_met["Test_MAE_kN"], color=model_colors, width=0.55, edgecolor="black", alpha=0.85)
        ax_mae.set_xticks(x)
        ax_mae.set_xticklabels(models, fontweight="bold")
        ax_mae.set_ylabel("Test MAE (kN)", fontsize=11, fontweight="bold")
        ax_mae.set_title(f"{title_prefix}: Test MAE (kN)", fontsize=12, fontweight="bold", pad=8)
        for bar in bars_mae:
            h = bar.get_height()
            ax_mae.annotate(f"{h:.1f}",
                            xy=(bar.get_x() + bar.get_width() / 2, h + 1.5),
                            ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.suptitle("Comparative ML Model Performance for Post-Peak Lateral Resistance Estimation\n(80/20 Grouped Train-Test Split by Paper_ID)",
                 fontsize=14, fontweight="bold", y=0.99)
    plt.tight_layout()
    out_path = os.path.join(output_dir, "model_comparison_metrics_bar_charts.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Saved model comparison bar chart -> %s", out_path)


def plot_actual_vs_predicted(test_preds_df: pd.DataFrame, shape_name: str, metrics_df: pd.DataFrame, output_dir: str):
    """
    Creates a 2x2 grid of Actual vs Predicted post-peak lateral resistance scatter plots
    with an ideal 45-degree reference line (y = x) and failure-mode color coding.
    """
    sns.set_theme(style="whitegrid", font="sans-serif")
    fig, axes = plt.subplots(2, 2, figsize=(14, 12), dpi=300)
    models = ["ANN", "KNN", "Decision Tree", "LightGBM"]

    y_actual = test_preds_df["Actual_Residual_Force_kN"]
    max_val = max(y_actual.max(), max(test_preds_df[f"Pred_{m}_kN"].max() for m in models)) * 1.10
    min_val = 0

    mode_markers = {
        "Flexure": "o",
        "Flexure-shear": "s",
        "Shear": "^",
        "NR": "X"
    }

    for idx, (ax, model_name) in enumerate(zip(axes.flatten(), models)):
        pred_col = f"Pred_{model_name}_kN"
        y_pred = test_preds_df[pred_col]

        # Get metrics for annotation
        row = metrics_df[metrics_df["Model"] == model_name].iloc[0]
        r2_val = row["Test_R2"]
        rmse_val = row["Test_RMSE_kN"]
        mae_val = row["Test_MAE_kN"]

        # Plot points color-coded by failure mode
        for mode in ["Flexure", "Flexure-shear", "Shear", "NR"]:
            sub = test_preds_df[test_preds_df["Failure_Mode_Normalized"].fillna("NR") == mode]
            if len(sub) > 0:
                ax.scatter(
                    sub["Actual_Residual_Force_kN"],
                    sub[pred_col],
                    label=mode if idx == 0 else "",
                    color=FAILURE_MODE_PALETTE.get(mode, "#7f8c8d"),
                    marker=mode_markers.get(mode, "o"),
                    s=55,
                    alpha=0.85,
                    edgecolors="black",
                    linewidth=0.6
                )

        # 45-degree ideal line
        ax.plot([min_val, max_val], [min_val, max_val], "k--", linewidth=1.5, label="Ideal (y = x)" if idx == 0 else "")
        # +/- 20% error boundary bands
        ax.plot([min_val, max_val], [min_val * 1.2, max_val * 1.2], ":", color="gray", linewidth=1.0, label="±20% Error Band" if idx == 0 else "")
        ax.plot([min_val, max_val], [min_val * 0.8, max_val * 0.8], ":", color="gray", linewidth=1.0)

        ax.set_xlim(min_val, max_val)
        ax.set_ylim(min_val, max_val)
        ax.set_xlabel("Experimental Post-Peak Force (kN)", fontsize=11, fontweight="bold")
        ax.set_ylabel("Predicted Post-Peak Force (kN)", fontsize=11, fontweight="bold")
        ax.set_title(f"{model_name} Regressor", fontsize=12, fontweight="bold", pad=8)

        # Performance text box
        text_str = f"$R^2$: {r2_val:.3f}\nRMSE: {rmse_val:.1f} kN\nMAE: {mae_val:.1f} kN"
        ax.text(0.04, 0.95, text_str, transform=ax.transAxes, fontsize=10,
                verticalalignment="top", bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8f9fa", edgecolor="#ced4da", alpha=0.9))

    # Add single unified legend to figure
    lines_labels = [ax.get_legend_handles_labels() for ax in [axes[0, 0]]]
    handles, labels = lines_labels[0]
    fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.98, 0.98), fontsize=10, frameon=True)

    fig.suptitle(f"Experimental vs. Predicted Post-Peak Lateral Resistance: {shape_name} Columns\n(Held-Out 20% Test Set via Paper_ID Grouping)",
                 fontsize=14, fontweight="bold", y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = os.path.join(output_dir, f"actual_vs_predicted_{shape_name.lower()}.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Saved Actual vs Predicted plot for %s -> %s", shape_name, out_path)


def plot_degradation_by_failure_mode(df_clean: pd.DataFrame, target_col: str, output_dir: str):
    """
    Creates comparative boxplots and distribution plots of post-peak force degradation
    and force retention ratio across Flexure, Flexure-shear, and Shear modes.
    """
    sns.set_theme(style="whitegrid", font="sans-serif")
    fig, axes = plt.subplots(2, 2, figsize=(15, 11), dpi=300)

    # Filter to valid failure modes
    df_plot = df_clean[df_clean["Failure_Mode_Normalized"].isin(FAILURE_MODE_ORDER)].copy()
    peak_col = "Peak_Strength_Vmax1_kN" if "Peak_Strength_Vmax1_kN" in df_plot.columns else "Peak_Force_from_History_kN"
    df_plot["Retention_Ratio"] = df_plot[target_col] / df_plot[peak_col]

    shapes = [("Rectangular", axes[0]), ("Circular", axes[1])]

    for shape_name, row_axes in shapes:
        sub_shape = df_plot[df_plot["Source_Database"] == shape_name].dropna(subset=[target_col, "Retention_Ratio"])

        # 1. Post-Peak Absolute Force
        ax_force = row_axes[0]
        sns.boxplot(
            data=sub_shape,
            x="Failure_Mode_Normalized",
            y=target_col,
            order=FAILURE_MODE_ORDER,
            palette=FAILURE_MODE_PALETTE,
            ax=ax_force,
            width=0.45,
            fliersize=3,
            boxprops=dict(alpha=0.75, edgecolor="black")
        )
        sns.stripplot(
            data=sub_shape,
            x="Failure_Mode_Normalized",
            y=target_col,
            order=FAILURE_MODE_ORDER,
            color="black",
            alpha=0.4,
            jitter=0.2,
            size=4.5,
            ax=ax_force
        )
        ax_force.set_title(f"{shape_name}: Post-Peak Lateral Resistance at Target Drift (kN)", fontsize=11, fontweight="bold")
        ax_force.set_xlabel("Observed Failure Mode", fontsize=10, fontweight="bold")
        ax_force.set_ylabel("Post-Peak Resistance (kN)", fontsize=10, fontweight="bold")

        # 2. Force Retention Ratio
        ax_ret = row_axes[1]
        sns.boxplot(
            data=sub_shape,
            x="Failure_Mode_Normalized",
            y="Retention_Ratio",
            order=FAILURE_MODE_ORDER,
            palette=FAILURE_MODE_PALETTE,
            ax=ax_ret,
            width=0.45,
            fliersize=3,
            boxprops=dict(alpha=0.75, edgecolor="black")
        )
        sns.stripplot(
            data=sub_shape,
            x="Failure_Mode_Normalized",
            y="Retention_Ratio",
            order=FAILURE_MODE_ORDER,
            color="black",
            alpha=0.4,
            jitter=0.2,
            size=4.5,
            ax=ax_ret
        )
        ax_ret.set_title(f"{shape_name}: Force Retention Ratio ($V_{{post}} / V_{{peak}}$)", fontsize=11, fontweight="bold")
        ax_ret.set_xlabel("Observed Failure Mode", fontsize=10, fontweight="bold")
        ax_ret.set_ylabel("Retention Ratio ($V_{{post}} / V_{{peak}}$)", fontsize=10, fontweight="bold")
        ax_ret.axhline(0.80, color="gray", linestyle="--", linewidth=1.0, label="80% Residual Threshold")
        ax_ret.legend(loc="upper right", fontsize=9)

    plt.suptitle("Post-Peak Lateral Resistance and Strength Retention Across Column Failure Modes",
                 fontsize=14, fontweight="bold", y=0.99)
    plt.tight_layout()
    out_path = os.path.join(output_dir, "postpeak_degradation_by_failure_mode.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Saved post-peak degradation analysis chart -> %s", out_path)


# ==============================================================================
# 8. MAIN ORCHESTRATOR
# ==============================================================================
def run_pipeline():
    """
    Executes the complete machine learning and failure mode analysis pipeline.
    """
    logger.info("Starting RC Column Post-Peak ML & Failure-Mode Analysis Pipeline...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Dataset auto-detection & loading
    filename, target_col, raw_df = auto_detect_dataset_and_target()

    # 2. Data Cleaning & Normalization
    df_cleaned = clean_dataset(raw_df, target_col)

    # 3. Shape Splitting (Rectangular vs Circular)
    df_rect = df_cleaned[df_cleaned["Source_Database"] == "Rectangular"].copy()
    df_circ = df_cleaned[df_cleaned["Source_Database"] == "Circular"].copy()

    # Filter strictly to valid non-null target rows
    df_rect_valid = df_rect[df_rect[target_col].notna()].copy()
    df_circ_valid = df_circ[df_circ[target_col].notna()].copy()

    logger.info("=" * 70)
    logger.info("FILTERED VALID SPECIMEN COUNTS WITH POST-PEAK FORCE:")
    logger.info("  -> Rectangular Columns: %d valid rows (out of %d total)", len(df_rect_valid), len(df_rect))
    logger.info("  -> Circular Columns:    %d valid rows (out of %d total)", len(df_circ_valid), len(df_circ))
    logger.info("=" * 70)

    # 4. Target Summary Statistics
    stats_rect = summarize_target(df_rect_valid, "Rectangular", target_col)
    stats_circ = summarize_target(df_circ_valid, "Circular", target_col)
    all_stats_df = pd.concat([stats_rect, stats_circ], ignore_index=True)
    all_stats_df.to_csv(os.path.join(OUTPUT_DIR, "target_summary_statistics.csv"), index=False)

    # 5. MATLAB GUI-Friendly Feature Verification
    logger.info("=" * 70)
    logger.info("MATLAB GUI FEATURE SELECTION (STRICT ZERO-LEAKAGE):")
    logger.info("  -> X_rect (%d features): %s", len(RECTANGULAR_FEATURES), RECTANGULAR_FEATURES)
    logger.info("  -> X_circ (%d features): %s", len(CIRCULAR_FEATURES), CIRCULAR_FEATURES)
    logger.info("=" * 70)

    # 6. Post-Peak Force Degradation Study
    logger.info("Conducting Post-Peak Lateral Force Degradation Study...")
    deg_rect = perform_degradation_study(df_rect_valid, "Rectangular", target_col, OUTPUT_DIR)
    deg_circ = perform_degradation_study(df_circ_valid, "Circular", target_col, OUTPUT_DIR)

    # 7. 80/20 Train-Test Splits via GroupShuffleSplit (grouped by Paper_ID)
    train_rect, test_rect, split_counts_rect = split_dataset_by_paper(df_rect_valid, "Rectangular")
    train_circ, test_circ, split_counts_circ = split_dataset_by_paper(df_circ_valid, "Circular")
    all_splits_summary = pd.concat([split_counts_rect, split_counts_circ], ignore_index=True)
    all_splits_summary.to_csv(os.path.join(OUTPUT_DIR, "failure_mode_counts_split.csv"), index=False)

    # 8. Train and Evaluate Models for Rectangular Columns
    logger.info("\n" + "=" * 70)
    logger.info("TRAINING & EVALUATION: RECTANGULAR COLUMNS (80% Train, 20% Test)")
    logger.info("=" * 70)
    metrics_rect, preds_rect, failure_err_rect = train_and_evaluate_shape(
        train_rect, test_rect, RECTANGULAR_FEATURES, target_col, "Rectangular", OUTPUT_DIR
    )
    logger.info("Rectangular Evaluation Metrics:\n%s", metrics_rect.to_string(index=False))

    # 9. Train and Evaluate Models for Circular Columns
    logger.info("\n" + "=" * 70)
    logger.info("TRAINING & EVALUATION: CIRCULAR COLUMNS (80% Train, 20% Test)")
    logger.info("=" * 70)
    metrics_circ, preds_circ, failure_err_circ = train_and_evaluate_shape(
        train_circ, test_circ, CIRCULAR_FEATURES, target_col, "Circular", OUTPUT_DIR
    )
    logger.info("Circular Evaluation Metrics:\n%s", metrics_circ.to_string(index=False))

    # Combined failure mode error breakdown table
    combined_failure_err = pd.concat([failure_err_rect, failure_err_circ], ignore_index=True)
    combined_failure_err.to_csv(os.path.join(OUTPUT_DIR, "failure_mode_test_error_breakdown.csv"), index=False)

    # 10. Generate Visualizations
    logger.info("\n" + "=" * 70)
    logger.info("GENERATING PUBLICATION-QUALITY VISUALIZATIONS...")
    logger.info("=" * 70)
    plot_model_metrics(metrics_rect, metrics_circ, OUTPUT_DIR)
    plot_actual_vs_predicted(preds_rect, "Rectangular", metrics_rect, OUTPUT_DIR)
    plot_actual_vs_predicted(preds_circ, "Circular", metrics_circ, OUTPUT_DIR)
    plot_degradation_by_failure_mode(df_cleaned, target_col, OUTPUT_DIR)

    # 11. Run Verification Demo for Saved Pipelines
    logger.info("\n" + "=" * 70)
    logger.info("RUNNING VERIFICATION DEMO FOR SAVED JOBLIB PIPELINES...")
    logger.info("=" * 70)
    try:
        from predict_residual_force import run_prediction_demo
        run_prediction_demo()
    except Exception as e:
        logger.warning("Could not run external demo from predict_residual_force: %s", e)

    logger.info("=" * 70)
    logger.info("ALL PIPELINE STAGES COMPLETED SUCCESSFULLY!")
    logger.info("Outputs saved in directory: %s", os.path.abspath(OUTPUT_DIR))
    logger.info("Saved models located in: %s", os.path.abspath(os.path.join(OUTPUT_DIR, "saved_models")))
    logger.info("=" * 70)


if __name__ == "__main__":
    run_pipeline()
