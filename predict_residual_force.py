"""
================================================================================
RC Column Post-Peak Lateral Resistance Prediction Module
================================================================================
Project: Data-Driven Estimation of Post-Peak Lateral Resistance of RC Columns
         Considering Failure Modes.

This module provides a clean, reusable inference interface to load saved trained
scikit-learn / LightGBM pipelines (joblib) and predict post-peak residual lateral
force (kN at target drift) for new RC column designs (Rectangular or Circular).

Key Features:
  1. Strict input validation and alignment with MATLAB GUI-friendly input parameters
     (10 features for Rectangular, 9 features for Circular).
  2. Automatic missing feature handling via the pipeline's fitted SimpleImputer.
  3. Support for single specimen (dict) and batch specimens (pd.DataFrame).
  4. Multi-model inference (ANN, KNN, Decision Tree, LightGBM) or specific model selection.
================================================================================
"""

import os
import sys
from typing import Dict, List, Optional, Union, Any

import joblib
import numpy as np
import pandas as pd

# ==============================================================================
# FEATURE DEFINITIONS (ALIGNED WITH TRAINING PIPELINE & MATLAB GUI)
# ==============================================================================
RECTANGULAR_FEATURES: List[str] = [
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

CIRCULAR_FEATURES: List[str] = [
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

MODEL_NAME_MAP: Dict[str, str] = {
    "ann": "ANN",
    "mlp": "ANN",
    "mlpregressor": "ANN",
    "knn": "KNN",
    "kneighbors": "KNN",
    "kneighborsregressor": "KNN",
    "decision_tree": "Decision_Tree",
    "decision tree": "Decision_Tree",
    "decisiontree": "Decision_Tree",
    "dt": "Decision_Tree",
    "decisiontreeregressor": "Decision_Tree",
    "lightgbm": "LightGBM",
    "lgbm": "LightGBM",
    "lgbmregressor": "LightGBM"
}

CANONICAL_MODELS: List[str] = ["ANN", "KNN", "Decision_Tree", "LightGBM"]
DISPLAY_MODEL_NAMES: Dict[str, str] = {
    "ANN": "ANN",
    "KNN": "KNN",
    "Decision_Tree": "Decision Tree",
    "LightGBM": "LightGBM"
}


# ==============================================================================
# PREDICTION FUNCTION
# ==============================================================================
def predict_postpeak_force(
    shape: str,
    input_data: Union[Dict[str, Any], pd.DataFrame],
    model_name: Optional[str] = None,
    models_dir: str = "outputs_ml/saved_models"
) -> Union[Dict[str, float], float, pd.DataFrame]:
    """
    Predicts post-peak residual lateral force (kN) for new RC column input parameters.

    Parameters:
      shape: Section shape ("Rectangular" or "Circular", case-insensitive).
      input_data: Dictionary representing a single column specimen or a pandas DataFrame
                  representing one or more specimens.
      model_name: Optional model to use ("ANN", "KNN", "Decision Tree" / "Decision_Tree", "LightGBM").
                  If None, predicts with all four models and returns an ensemble summary.
      models_dir: Directory containing saved .joblib model files.

    Returns:
      - If input_data is a single dict and model_name is specified: float (predicted force in kN).
      - If input_data is a single dict and model_name is None: dict of predictions per model + ensemble mean.
      - If input_data is a pd.DataFrame: copy of DataFrame with new prediction columns.
    """
    # 1. Validate Shape
    shape_clean = shape.strip().capitalize()
    if shape_clean not in ["Rectangular", "Circular"]:
        raise ValueError(f"Invalid shape '{shape}'. Must be either 'Rectangular' or 'Circular'.")

    # 2. Select expected feature list
    expected_features = RECTANGULAR_FEATURES if shape_clean == "Rectangular" else CIRCULAR_FEATURES

    # 3. Format input into DataFrame
    is_single_dict = False
    if isinstance(input_data, dict):
        # Check if values are scalar or lists
        has_list_val = any(isinstance(v, (list, tuple, np.ndarray)) for v in input_data.values())
        if has_list_val:
            df_in = pd.DataFrame(input_data)
        else:
            is_single_dict = True
            df_in = pd.DataFrame([input_data])
    elif isinstance(input_data, pd.DataFrame):
        df_in = input_data.copy()
    else:
        raise TypeError(f"input_data must be a dict or pd.DataFrame, got {type(input_data)}.")

    # 4. Strictly align features (no leakage, fill missing with NaN for pipeline imputer)
    missing_features = []
    aligned_df = pd.DataFrame(index=df_in.index)
    for feat in expected_features:
        if feat in df_in.columns:
            aligned_df[feat] = pd.to_numeric(df_in[feat], errors="coerce")
        else:
            missing_features.append(feat)
            aligned_df[feat] = np.nan

    if missing_features:
        # Informational note: pipeline imputer will supply median defaults
        pass

    # 5. Check models directory
    if not os.path.exists(models_dir):
        raise FileNotFoundError(
            f"Saved models directory '{models_dir}' not found. "
            "Please run 'rc_column_ml_pipeline.py' to fit and save model pipelines first."
        )

    # 6. Determine which models to evaluate
    if model_name is not None:
        norm_key = model_name.strip().lower()
        if norm_key not in MODEL_NAME_MAP:
            valid_keys = list(MODEL_NAME_MAP.keys())
            raise ValueError(f"Unknown model_name '{model_name}'. Choose from: {valid_keys}")
        models_to_run = [MODEL_NAME_MAP[norm_key]]
    else:
        models_to_run = CANONICAL_MODELS

    # 7. Load pipelines and compute predictions
    predictions_dict = {}
    for m_key in models_to_run:
        filename = f"{shape_clean.lower()}_{m_key}.joblib"
        model_path = os.path.join(models_dir, filename)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file '{model_path}' not found. Ensure pipeline has been executed.")

        pipeline = joblib.load(model_path)
        preds = pipeline.predict(aligned_df)
        predictions_dict[m_key] = preds

    # 8. Return formatted output
    if is_single_dict:
        if model_name is not None:
            # Return single float
            return float(predictions_dict[models_to_run[0]][0])
        else:
            # Return dictionary with all model predictions plus ensemble mean
            out_dict = {}
            vals = []
            for m_key in CANONICAL_MODELS:
                disp_name = DISPLAY_MODEL_NAMES[m_key]
                val = float(predictions_dict[m_key][0])
                out_dict[f"{disp_name} (kN)"] = round(val, 2)
                vals.append(val)
            out_dict["Ensemble Mean (kN)"] = round(float(np.mean(vals)), 2)
            return out_dict
    else:
        # Return enriched DataFrame
        df_out = df_in.copy()
        for m_key, preds in predictions_dict.items():
            disp_name = DISPLAY_MODEL_NAMES[m_key]
            df_out[f"Pred_Force_{disp_name}_kN"] = np.round(preds, 2)

        if len(predictions_dict) > 1:
            all_preds_matrix = np.column_stack([predictions_dict[k] for k in models_to_run])
            df_out["Pred_Force_Ensemble_Mean_kN"] = np.round(np.mean(all_preds_matrix, axis=1), 2)

        return df_out


# ==============================================================================
# DEMO / VERIFICATION FUNCTION
# ==============================================================================
def run_prediction_demo():
    """
    Runs a standalone verification demo using sample Rectangular and Circular
    column inputs to demonstrate the inference pipeline.
    """
    print("=" * 75)
    print("DEMO: POST-PEAK RESIDUAL LATERAL RESISTANCE INFERENCE")
    print("=" * 75)

    # Sample 1: Rectangular Column
    sample_rectangular = {
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

    # Sample 2: Circular Column
    sample_circular = {
        "Column_Clear_Length_L_mm": 1500.0,
        "Section_Diameter_D_mm": 400.0,
        "rho_L_Longitudinal_Ratio": 0.020,
        "Axial_Load_P_kN": 600.0,
        "rho_t_Transverse_Volumetric_Ratio": 0.010,
        "Stirrup_Spacing_s_mm": 60.0,
        "fc_prime_MPa": 35.0,
        "fy_Longitudinal_MPa": 450.0,
        "fyt_Transverse_MPa": 380.0
    }

    print("\n1. Predicting for Sample Rectangular Column:")
    print("   Input Parameters:")
    for k, v in sample_rectangular.items():
        print(f"     - {k}: {v}")
    
    preds_rect = predict_postpeak_force("Rectangular", sample_rectangular)
    print("\n   Predicted Residual Lateral Force (kN at 2% drift):")
    for m, p in preds_rect.items():
        print(f"     * {m:<22}: {p:>8.2f} kN")

    print("\n2. Predicting for Sample Circular Column:")
    print("   Input Parameters:")
    for k, v in sample_circular.items():
        print(f"     - {k}: {v}")
    
    preds_circ = predict_postpeak_force("Circular", sample_circular)
    print("\n   Predicted Residual Lateral Force (kN at 2% drift):")
    for m, p in preds_circ.items():
        print(f"     * {m:<22}: {p:>8.2f} kN")

    # Single-model query demo
    single_model_pred = predict_postpeak_force("Rectangular", sample_rectangular, model_name="LightGBM")
    print(f"\n3. Single Model Query Demo (LightGBM only): {single_model_pred:.2f} kN")

    # Batch DataFrame Demo
    print("\n4. Batch Prediction Demo (2 Rectangular specimens in DataFrame):")
    batch_rect = pd.DataFrame([
        sample_rectangular,
        {**sample_rectangular, "Axial_Load_P_kN": 800.0, "rho_L_Longitudinal_Ratio": 0.030}
    ])
    batch_res = predict_postpeak_force("Rectangular", batch_rect)
    pred_cols = [c for c in batch_res.columns if "Pred" in c]
    print(batch_res[pred_cols].to_string())

    print("\n" + "=" * 75)
    print("VERIFICATION COMPLETED SUCCESSFULLY!")
    print("=" * 75)


if __name__ == "__main__":
    run_prediction_demo()
