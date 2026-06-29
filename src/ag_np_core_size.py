#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Publication-ready reproducible workflow for AgNP core-size regression.

This script is prepared as a source-code package for journal submission and public archiving.
It executes the complete computational workflow used in the manuscript: dataset loading,
feature selection, duplicate cleaning, train-holdout splitting, leakage-safe preprocessing,
baseline evaluation, single and combined preprocessing experiments, hyperparameter tuning,
final cross-validation, SHAP feature-importance interpretation, explicit holdout prediction,
and inverse-design candidate search.

Generated outputs follow a repository-friendly structure:

outputs/
├── tables/
├── figures/
├── json/
├── snippets/
├── logs/
├── data_snapshots/
├── audit/
├── preprocessed_matrices/
├── models/
└── predictions/

Default local usage:

Place the dataset inside the data/ folder and the source code inside the src/ folder:

AgNP_core_size_publication/
├── data/
│   └── 1-s2.0-S2001037024000333-mmc1.xlsx
└── src/
    └── ag_np_core_size.py

Then run from the repository root:

python src/ag_np_core_size.py --data data/1-s2.0-S2001037024000333-mmc1.xlsx --sheet core_size --output outputs

The script will automatically write outputs to:

AgNP_core_size_publication/outputs/

Notes:
- Default mode uses FAST_MODE=False and evaluates the full combined-preprocessing grid.
- The holdout test set remains separated from preprocessing and hyperparameter selection.
- CLI arguments only control file paths and runtime switches; they do not change the scientific workflow unless explicitly requested.
- Non-manuscript exploratory models and feature-subset retraining are intentionally excluded.
"""

from __future__ import annotations

import argparse as _argparse
from pathlib import Path as _RuntimePath

# ============================================================
# DEFAULT PATH FOR LOCAL PUBLICATION WORKFLOW
# ============================================================
# Place this script in the same folder as the dataset, for example:
# AgNP_core_size_publication/
# ├── data/
# │   └── 1-s2.0-S2001037024000333-mmc1.xlsx
# └── src/
#     └── ag_np_core_size.py
#
# When this script is executed, all outputs are written to:
# AgNP_core_size_publication/outputs/

_SCRIPT_DIR = _RuntimePath(__file__).resolve().parent if "__file__" in globals() else _RuntimePath.cwd()
_REPO_ROOT = _SCRIPT_DIR.parent if _SCRIPT_DIR.name == "src" else _SCRIPT_DIR
_DATASET_FILENAME_DEFAULT = "1-s2.0-S2001037024000333-mmc1.xlsx"
_DEFAULT_DATA_PATH = _REPO_ROOT / "data" / _DATASET_FILENAME_DEFAULT
_DEFAULT_OUTPUT_DIR = _REPO_ROOT / "outputs"


def _parse_runtime_args():
    parser = _argparse.ArgumentParser(
        description="Run reviewer-ready AgNP core-size regression workflow and export repository-style outputs."
    )
    parser.add_argument(
        "--data",
        default=str(_DEFAULT_DATA_PATH),
        help=(
            "Path to Furxhi Excel file. Default: dataset with the exact filename "
            "placed in the same folder as this .py file."
        ),
    )
    parser.add_argument(
        "--sheet",
        default="core_size",
        help="Excel sheet name. Default: core_size.",
    )
    parser.add_argument(
        "--output",
        default=str(_DEFAULT_OUTPUT_DIR),
        help=(
            "Output directory. Default: outputs folder in the same folder as this .py file, "
            "matching the reference ZIP structure."
        ),
    )
    parser.add_argument(
        "--fast-mode",
        action="store_true",
        help="Enable FAST_MODE for checking the script quickly. Default is disabled to preserve the full pipeline.",
    )
    parser.add_argument(
        "--cv-tuning",
        type=int,
        default=5,
        help="Number of CV folds for tuning. Default: 5.",
    )
    parser.add_argument(
        "--k-values-final",
        default="5,10,15",
        help="Comma-separated final K-Fold values. Default: 5,10,15.",
    )
    return parser.parse_args()


_RUNTIME_ARGS = _parse_runtime_args()
_RUNTIME_START_TIME = None

# %% [markdown]
# # Publication-ready reproducible workflow: AgNP core-size regression
#
# This script supports a journal manuscript on the effect of preprocessing and hyperparameter tuning on machine-learning regression for silver nanoparticle (AgNP) core-size prediction.
#
# The workflow covers:
# 1. Setup, global configuration, and output-folder preparation.
# 2. Loading the `core_size` sheet from the Furxhi et al. supplementary dataset.
# 3. Raw-data and modelling-dataset audits before model preprocessing.
# 4. Target cleaning, exact-duplicate removal, and initial missing-value/outlier summaries.
# 5. An 80:20 train-holdout split with log-target stratification bins.
# 6. Leakage-safe preprocessing functions, model definitions, holdout evaluation, and K-Fold CV.
# 7. Baseline snapshots before and after complete-case preprocessing.
# 8. Baseline evaluation and target-transformation checks.
# 9. Single-preprocessing experiments.
# 10. Combined-preprocessing grid search.
# 11. Hyperparameter tuning on the best preprocessing configuration per model.
# 12. Baseline/preprocessing/tuning comparison, final K-Fold validation, SHAP feature-importance interpretation, explicit holdout prediction, and inverse-design candidate search.
#
# Models evaluated:
# - Multiple Linear Regression (MLR)
# - Ridge Regression
# - Support Vector Regression (SVR)
# - XGBoost Regression
# - Extra Trees Regression
# - Random Forest Regression
#
# This source file is designed for publication and archival use. It contains no thesis-specific or coursework-specific sections. SHAP is used only for feature-importance interpretation, not for feature-subset retraining.

# %% [markdown]
# ## Regression task clarification
#
# The target variable is `core size (nm)`, a continuous numerical variable. Therefore, all evaluated models are regression models. Random Forest Regressor, for example, predicts the average numerical output across trees rather than a majority-vote class.

# %% [cell 3]
# ============================================================
# CELL 1 - SETUP, IMPORTS, AND GLOBAL CONFIGURATION
# ============================================================

# JUPYTER_MAGIC: !pip -q install xgboost openpyxl joblib
# JUPYTER_MAGIC: !pip -q install shap

import shap


print("SHAP version:", shap.__version__)

import os
import re
import json
import time
import math
import shutil
import warnings
from pathlib import Path
from copy import deepcopy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from IPython.display import display, Markdown
except Exception:
    def display(obj=None, *args, **kwargs):
        if obj is None:
            return None
        try:
            import pandas as _pd
            if isinstance(obj, (_pd.DataFrame, _pd.Series)):
                print(obj.to_string())
            else:
                print(obj)
        except Exception:
            print(obj)
    class Markdown(str):
        pass

from sklearn.model_selection import train_test_split, KFold, ParameterSampler
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler,
    QuantileTransformer, PowerTransformer, OneHotEncoder
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.base import clone
from sklearn.model_selection import ParameterGrid

from xgboost import XGBRegressor
import joblib

warnings.filterwarnings("ignore")

pd.set_option("display.max_columns", 250)
pd.set_option("display.max_rows", 250)
pd.set_option("display.width", 240)

RANDOM_STATE = 42

# ============================================================
# RUNTIME SETTINGS FOR THE core_size DATASET
# ============================================================
# The default baseline remains complete-case analysis/drop rows with missing values.
# Combined preprocessing uses the full grid defined in the preprocessing-experiment workflow.
# Hyperparameter tuning uses full-grid evaluation so all parameter candidates are tested.
# Note: runtime can be long, especially when Ridge, Extra Trees, and Random Forest are included.

FAST_MODE = bool(_RUNTIME_ARGS.fast_mode)

FULL_COMBINED_GRID = True

CV_TUNING = int(_RUNTIME_ARGS.cv_tuning)

# Final K-Fold settings for the final validation stage.
K_VALUES_FINAL = [int(x.strip()) for x in str(_RUNTIME_ARGS.k_values_final).split(",") if x.strip()]

# Hyperparameter tuning uses the full grid.
N_ITER_TUNING = None

# Because FULL_COMBINED_GRID=True, this value is ignored by Cell 11.
N_ITER_COMBINED_PREPROCESSING = None

INCLUDE_RANDOM_FOREST = True

# Primary metric for selecting preprocessing, tuning, and final models.
# The original scale is selected so the main target remains core size (nm), not only the log scale.
PRIMARY_SELECTION_METRIC = "CV_Mean_R2_Original"

# ============================================================
# OVERFITTING AUDIT SETTINGS
# ============================================================
# Train R2, train-validation gap, and train-holdout gap are used for overfitting audit, not as the main selection metrics.
# The main selection metric remains PRIMARY_SELECTION_METRIC to match the study objective: evaluating preprocessing and hyperparameter-tuning effects.

MAX_ALLOWED_TRAIN_VAL_GAP_R2 = 0.20
MAX_ALLOWED_TRAIN_HOLDOUT_GAP_R2 = 0.20

# The larger this value, the stronger the penalty for overfitted models.
OVERFIT_GAP_PENALTY_WEIGHT = 0.75

# Penalty for models whose fold-to-fold performance is unstable.
CV_STD_PENALTY_WEIGHT = 0.10

# The stability score is used only for overfitting audit, not as the main model-selection metric.
SELECTION_SCORE_METRIC = "CV_Stability_Score_Original"

print("MAX_ALLOWED_TRAIN_VAL_GAP_R2:", MAX_ALLOWED_TRAIN_VAL_GAP_R2)
print("MAX_ALLOWED_TRAIN_HOLDOUT_GAP_R2:", MAX_ALLOWED_TRAIN_HOLDOUT_GAP_R2)
print("OVERFIT_GAP_PENALTY_WEIGHT:", OVERFIT_GAP_PENALTY_WEIGHT)
print("CV_STD_PENALTY_WEIGHT:", CV_STD_PENALTY_WEIGHT)
print("SELECTION_SCORE_METRIC:", SELECTION_SCORE_METRIC)

SAVE_OUTPUTS = True

OUTPUT_DIR = _RuntimePath(_RUNTIME_ARGS.output).resolve()
TABLE_DIR = OUTPUT_DIR / "tables"
PLOT_DIR = OUTPUT_DIR / "figures"
MODEL_DIR = OUTPUT_DIR / "models"
PRED_DIR = OUTPUT_DIR / "predictions"

for d in [OUTPUT_DIR, TABLE_DIR, PLOT_DIR, MODEL_DIR, PRED_DIR]:
    d.mkdir(parents=True, exist_ok=True)

def save_table(df, name):
    if not SAVE_OUTPUTS:
        print("SAVE_OUTPUTS=False; table was not saved:", name)
        return None

    path = TABLE_DIR / name
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print("Saved:", path)
    return path

print("Setup completed.")
print("FAST_MODE:", FAST_MODE)
print("FULL_COMBINED_GRID:", FULL_COMBINED_GRID)
print("CV_TUNING:", CV_TUNING)
print("K_VALUES_FINAL:", K_VALUES_FINAL)
print("N_ITER_TUNING:", N_ITER_TUNING)
print("N_ITER_COMBINED_PREPROCESSING:", N_ITER_COMBINED_PREPROCESSING)
print("PRIMARY_SELECTION_METRIC:", PRIMARY_SELECTION_METRIC)
print("INCLUDE_RANDOM_FOREST:", INCLUDE_RANDOM_FOREST)
print("SCRIPT_DIR:", _SCRIPT_DIR)
print("DEFAULT_DATA_PATH:", _DEFAULT_DATA_PATH)
print("OUTPUT_DIR:", OUTPUT_DIR)

# %% [markdown]
# ## Additional output structure
#
# This version follows the reference ZIP pattern: results are not only displayed in notebook cells but are also saved into clear folders.
#
# The main output remains in `outputs_ag_np_core_size_article_furxhi_features/`, with additional subfolders:
#
# - `tables/` for all manuscript-ready CSV tables.
# - `plots/` and `figures/` for figures and plots.
# - `json/` for run metadata, package versions, and model configurations.
# - `snippets/` for manuscript-ready narrative summaries.
# - `logs/` for process logs.
# - `data_snapshots/` for data before and after preprocessing.
# - `audit/` for reviewer-audit tables.
# - `preprocessed_matrices/` for preprocessed/encoded matrices of key scenarios.


# %% [cell 5]
# ============================================================
# CELL 1B - REPRODUCIBLE OUTPUT STRUCTURE AND RUN METADATA
# ============================================================
# This cell does not change the modelling pipeline. It only organizes outputs to resemble a reproducible repository.

import sys
import platform
import importlib
from datetime import datetime

# The legacy folder alias is retained for compatibility with the original notebook.
FIGURE_DIR = PLOT_DIR
JSON_DIR = OUTPUT_DIR / "json"
SNIPPET_DIR = OUTPUT_DIR / "snippets"
LOG_DIR = OUTPUT_DIR / "logs"
DATA_SNAPSHOT_DIR = OUTPUT_DIR / "data_snapshots"
AUDIT_DIR = OUTPUT_DIR / "audit"
PREPROCESSED_MATRIX_DIR = OUTPUT_DIR / "preprocessed_matrices"

for d in [FIGURE_DIR, JSON_DIR, SNIPPET_DIR, LOG_DIR, DATA_SNAPSHOT_DIR, AUDIT_DIR, PREPROCESSED_MATRIX_DIR]:
    d.mkdir(parents=True, exist_ok=True)

def save_json(obj, name, folder=JSON_DIR):
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)
    print("Saved:", path)
    return path

def save_text(text, name, folder=SNIPPET_DIR):
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    with open(path, "w", encoding="utf-8") as f:
        f.write(str(text))
    print("Saved:", path)
    return path

def reviewer_safe_filename(text, max_len=160):
    text = str(text).strip().lower()
    text = text.replace("+", "plus")
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = text.strip("_")
    return text[:max_len] if len(text) > max_len else text

def get_package_versions(package_names):
    versions = {
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    for package_name in package_names:
        try:
            module = importlib.import_module(package_name)
            versions[package_name] = getattr(module, "__version__", "version_attribute_not_found")
        except Exception as exc:
            versions[package_name] = f"not_available: {exc}"
    return versions

package_versions = get_package_versions([
    "numpy", "pandas", "sklearn", "xgboost", "shap", "openpyxl", "joblib", "matplotlib"
])
save_json(package_versions, "package_versions.json")

run_metadata = {
    "project": "AgNP core size regression",
    "target": "core size (nm)",
    "random_state": RANDOM_STATE,
    "fast_mode": FAST_MODE,
    "full_combined_grid": FULL_COMBINED_GRID,
    "cv_tuning": CV_TUNING,
    "k_values_final": K_VALUES_FINAL,
    "primary_selection_metric": PRIMARY_SELECTION_METRIC,
    "output_dir": str(OUTPUT_DIR),
    "pipeline_note": "Metadata/export cells added for reviewer audit; modelling pipeline is unchanged."
}
save_json(run_metadata, "run_metadata.json")

protocol_summary_text = f"""
Reviewer-ready reproducibility protocol
======================================
Project: Regression model for AgNP core size prediction.
Target: {TARGET_DISPLAY_NAME if 'TARGET_DISPLAY_NAME' in globals() else 'core size'}.
Random state: {RANDOM_STATE}.
Train-test split: 80:20; holdout is not used to choose preprocessing/hyperparameters.
CV tuning folds: {CV_TUNING}.
Final K-Fold values: {K_VALUES_FINAL}.
Primary selection metric: {PRIMARY_SELECTION_METRIC}.
Baseline: complete-case/drop missing rows, no scaling, no target outlier handling, no feature outlier handling, no target transform.
Preprocessing evaluation: single preprocessing and combined preprocessing scenarios.
Tuning: grid-based tuning on the best preprocessing configuration per model.
Leakage control: preprocessing decisions for training/CV are fitted on training folds/data; holdout remains for final evaluation.
Additional reviewer audit: data snapshots before preprocessing, baseline after preprocessing, scenario-to-config map, and key preprocessed matrices.
""".strip()
save_text(protocol_summary_text, "protocol_summary.txt")

print("Reviewer-ready output folders prepared.")
print("DATA_SNAPSHOT_DIR:", DATA_SNAPSHOT_DIR)
print("AUDIT_DIR:", AUDIT_DIR)
print("PREPROCESSED_MATRIX_DIR:", PREPROCESSED_MATRIX_DIR)


# %% [cell 6]
# ============================================================
# CELL 2 - LOAD DATASET EXCEL SHEET core_size
# ============================================================

DATASET_FILENAME = "1-s2.0-S2001037024000333-mmc1.xlsx"
SHEET_NAME = str(_RUNTIME_ARGS.sheet)

DEFAULT_DRIVE_DIR = Path(
    "/content/drive/MyDrive/AgNP_Core_Size_Prediction/Dataset"
)

DATASET_CANDIDATES = []
if _RUNTIME_ARGS.data:
    DATASET_CANDIDATES.append(_RuntimePath(_RUNTIME_ARGS.data).expanduser())
DATASET_CANDIDATES.extend([
    # Primary local structure: the dataset is placed in the same folder as this .py file.
    _SCRIPT_DIR / DATASET_FILENAME,
    # Alternative path if the dataset is placed in a data/ subfolder.
    _SCRIPT_DIR / "data" / DATASET_FILENAME,
    # Legacy notebook/Colab path option.
    DEFAULT_DRIVE_DIR / DATASET_FILENAME,
    Path("data") / DATASET_FILENAME,
    Path(DATASET_FILENAME),
    Path("/mnt/data") / DATASET_FILENAME,
])

def maybe_mount_drive():
    try:
        from google.colab import drive
        drive.mount("/content/drive")
        return True
    except Exception as e:
        print("Not a Colab environment or Google Drive is not being used:", e)
        return False

maybe_mount_drive()

DATASET_PATH = None
for candidate in DATASET_CANDIDATES:
    if candidate.exists():
        DATASET_PATH = candidate
        break

if DATASET_PATH is None:
    print("Excel dataset was not found in the following paths:")
    for candidate in DATASET_CANDIDATES:
        print("-", candidate)

    print("\nPlease upload file:", DATASET_FILENAME)
    try:
        from google.colab import files
        uploaded = files.upload()
        if len(uploaded) == 0:
            raise FileNotFoundError("No file was uploaded.")
        DATASET_PATH = Path(list(uploaded.keys())[0])
    except Exception as e:
        raise FileNotFoundError(
            "Dataset was not found and the Colab upload fallback failed. "
            "Place the Excel file in the data/ folder, the working directory, /mnt/data, "
            "or run the script with --data path/to/file.xlsx."
        ) from e

xls = pd.ExcelFile(DATASET_PATH, engine="openpyxl")
print("Available sheets:", xls.sheet_names)

if SHEET_NAME not in xls.sheet_names:
    raise ValueError(f"Sheet '{SHEET_NAME}' was not found. Available sheets: {xls.sheet_names}")

df_raw = pd.read_excel(DATASET_PATH, sheet_name=SHEET_NAME, engine="openpyxl")
df_raw.columns = df_raw.columns.str.strip()

print("\nDATASET USED")
print("File:", DATASET_PATH)
print("Sheet:", SHEET_NAME)
print("Initial shape:", df_raw.shape)

display(df_raw.head())
display(df_raw.info())

# %% [markdown]
# ## Raw-data audit before preprocessing
#
# The following cell saves a copy of the raw data after reading the `core_size` sheet, but before feature selection, missing-token replacement, target cleaning, duplicate removal, and model preprocessing. This is useful for reviewer questions about the initial data state before baseline and preprocessing scenarios.


# %% [cell 8]
# ============================================================
# CELL 2B - RAW DATA SNAPSHOT BEFORE PREPROCESSING
# ============================================================
# This does not modify df_raw. It only saves the initial snapshot and profile.

raw_snapshot_path = DATA_SNAPSHOT_DIR / "00_raw_sheet_core_size_before_preprocessing.csv"
df_raw.to_csv(raw_snapshot_path, index=False)
print("Saved:", raw_snapshot_path)

def make_column_profile(df, dataset_name):
    rows = []
    for col in df.columns:
        s = df[col]
        rows.append({
            "Dataset": dataset_name,
            "Column": col,
            "Dtype": str(s.dtype),
            "Rows": int(len(s)),
            "Missing_Count": int(s.isna().sum()),
            "Missing_Percentage": float(s.isna().mean() * 100),
            "Unique_Count": int(s.nunique(dropna=True)),
            "Example_NonMissing_Value": None if s.dropna().empty else str(s.dropna().iloc[0])
        })
    return pd.DataFrame(rows)

raw_profile_df = make_column_profile(df_raw, "raw_sheet_core_size_before_preprocessing")
display(raw_profile_df)
save_table(raw_profile_df, "00a_raw_sheet_column_profile_before_preprocessing.csv")


# %% [cell 9]
# ============================================================
# CELL 3 - FEATURE AND TARGET CONFIGURATION FOR THE core_size SHEET
# ARTICLE-LIKE VERSION: THREE HIGH-MISSINGNESS FEATURES ARE EXCLUDED
# ============================================================

TARGET_COL = "core size (nm)"

TARGET_DISPLAY_NAME = "Core Size"
TARGET_UNIT = "nm"
PRED_ACTUAL_COL = "Actual_Core_Size_nm"
PRED_PRED_COL = "Predicted_Core_Size_nm"

# ============================================================
# FEATURES EXCLUDED ACCORDING TO THE ARTICLE-STYLE FEATURE SET
# ============================================================
# Furxhi et al. excluded features with large missing-value proportions:
# 1. Stirring speed
# 2. Capping agent concentration
# 3. Reducing agent concentration/quantity
#
# In the core_size sheet, the corresponding columns are:
# - Stiring_Speed_rpm_
# - Capping agent concentration (mg/mL)
# - Reducing agent quantity_Ml

EXCLUDED_HIGH_MISSING_FEATURES = [
    "Stiring_Speed_rpm_",
    "Capping agent concentration (mg/mL)",
    "Reducing agent quantity_Ml",
]

# ============================================================
# INPUT FEATURES AFTER EXCLUDING THREE HIGH-MISSINGNESS FEATURES
# ============================================================

FEATURE_COLS = [
    "Process",
    "Steps",
    "External_energy",
    "Temp_Celcius",
    "Stirring",
    "Synthesis_Duration_h",
    "Scale_synthesis_ml",
    "Precurcor_conc_mM",
    "NEW_Capping agent",
    "NEW_capping agent_Class",
    "NEW_reducing agent",
    "NEW_reducing agent_class",
    "Order of reagent_CODE",
    "Treatment",
]

NUMERIC_COLS = [
    "Temp_Celcius",
    "Synthesis_Duration_h",
    "Scale_synthesis_ml",
    "Precurcor_conc_mM",
]

CATEGORICAL_COLS = [c for c in FEATURE_COLS if c not in NUMERIC_COLS]

# The core_size data uses "?" as the missing-value token.
MISSING_TOKENS = ["?", "NA", "N/A", "na", "n/a", "nan", "None", "none", "-", ""]

df_raw = df_raw.copy()

for col in df_raw.columns:
    if df_raw[col].dtype == "object":
        df_raw[col] = (
            df_raw[col]
            .astype(str)
            .str.strip()
            .replace({token: np.nan for token in MISSING_TOKENS})
        )

# Convert numeric columns and the target variable.
for col in NUMERIC_COLS + [TARGET_COL]:
    if col in df_raw.columns:
        df_raw[col] = pd.to_numeric(df_raw[col], errors="coerce")

required_cols = [TARGET_COL] + FEATURE_COLS
missing_cols = [c for c in required_cols if c not in df_raw.columns]

if missing_cols:
    raise ValueError(f"The following columns were not found in the dataset: {missing_cols}")

df_data = df_raw[required_cols].copy()

# Baseline default remains:
# Missing values are handled by removing rows that contain missing values.
BASE_IMPUTER_MODE = "drop"

print("=" * 100)
print("VARIABLE CONFIGURATION FOR THE core_size DATASET - ARTICLE-STYLE FEATURES")
print("=" * 100)
print("Target:", TARGET_COL)
print("Number of features:", len(FEATURE_COLS))
print("Number of numeric features:", len(NUMERIC_COLS))
print("Number of categorical features:", len(CATEGORICAL_COLS))
print("Baseline imputer:", BASE_IMPUTER_MODE)

print("\nFeatures excluded due to high missingness:")
for c in EXCLUDED_HIGH_MISSING_FEATURES:
    if c in df_raw.columns:
        print("-", c, "| Missing:", df_raw[c].isna().sum(), f"({df_raw[c].isna().mean()*100:.2f}%)")
    else:
        print("-", c, "| Not found in the dataset")

print("\nNumeric features used:")
for c in NUMERIC_COLS:
    print("-", c)

print("\nCategorical features used:")
for c in CATEGORICAL_COLS:
    print("-", c)

print("\nPreview of final modelling data:")
display(df_data.head())

feature_summary = pd.DataFrame({
    "Feature": FEATURE_COLS,
    "Role": ["synthesis_parameter"] * len(FEATURE_COLS),
    "Type": ["numeric" if col in NUMERIC_COLS else "categorical" for col in FEATURE_COLS],
    "Missing_Count": [df_data[col].isna().sum() for col in FEATURE_COLS],
    "Missing_Percentage": [df_data[col].isna().mean() * 100 for col in FEATURE_COLS],
    "Unique_Count": [df_data[col].nunique(dropna=True) for col in FEATURE_COLS],
})

display(feature_summary)
save_table(feature_summary, "00_selected_features_core_size_article_like.csv")

# %% [markdown]
# ## Modelling dataset before baseline preprocessing
#
# At this stage, the dataset has been restricted to the target and 14 manuscript features, but baseline complete-case handling, imputation, scaling, outlier handling, target transformation, and model one-hot encoding have not yet been applied. This file serves as a direct comparison point for changes after baseline and preprocessing scenarios.


# %% [cell 11]
# ============================================================
# CELL 3B - MODELLING-DATASET SNAPSHOT BEFORE MODEL PREPROCESSING
# ============================================================
# df_data contains the target and selected features. Baseline row dropping and model-preprocessing scenarios have not yet been applied.

modeling_snapshot_path = DATA_SNAPSHOT_DIR / "01_modeling_dataset_selected_features_before_model_preprocessing.csv"
df_data.to_csv(modeling_snapshot_path, index=False)
print("Saved:", modeling_snapshot_path)

modeling_profile_df = make_column_profile(df_data, "selected_modeling_dataset_before_model_preprocessing")
feature_type_map = {c: "numeric" for c in NUMERIC_COLS}
feature_type_map.update({c: "categorical" for c in CATEGORICAL_COLS})
feature_type_map[TARGET_COL] = "target_numeric"
modeling_profile_df["Role"] = modeling_profile_df["Column"].map(feature_type_map).fillna("not_used")
display(modeling_profile_df)
save_table(modeling_profile_df, "00b_modeling_dataset_profile_before_model_preprocessing.csv")

high_missing_exclusion_audit = []
for col in EXCLUDED_HIGH_MISSING_FEATURES:
    high_missing_exclusion_audit.append({
        "Excluded_Feature": col,
        "Found_In_Raw_Data": bool(col in df_raw.columns),
        "Missing_Count": int(df_raw[col].isna().sum()) if col in df_raw.columns else np.nan,
        "Missing_Percentage": float(df_raw[col].isna().mean() * 100) if col in df_raw.columns else np.nan,
        "Reason": "Excluded because it has large missingness in the Furxhi-style article feature set."
    })
high_missing_exclusion_audit_df = pd.DataFrame(high_missing_exclusion_audit)
display(high_missing_exclusion_audit_df)
save_table(high_missing_exclusion_audit_df, "00c_excluded_high_missing_features_audit.csv")


# %% [cell 12]
# ============================================================
# CELL 4 - EDA, TARGET CLEANING, MISSING VALUES, AND INITIAL OUTLIERS
# ============================================================

def plot_hist_target(y, filename, title_suffix=""):
    plt.figure(figsize=(9, 4))
    plt.hist(y, bins=50)
    plt.title(f"Target Distribution {TARGET_DISPLAY_NAME} {title_suffix}")
    plt.xlabel(f"{TARGET_DISPLAY_NAME} ({TARGET_UNIT})")
    plt.ylabel("Frequency")
    plt.tight_layout()
    path = PLOT_DIR / filename
    plt.savefig(path, dpi=180)
    plt.show()
    return path

def iqr_bounds(series, multiplier=1.5):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - multiplier * iqr
    upper = q3 + multiplier * iqr
    return lower, upper, q1, q3, iqr

print("=" * 100)
print("INITIAL EDA FOR THE core_size DATASET")
print("=" * 100)
print("Shape before target cleaning:", df_data.shape)

missing_before = df_data.isna().sum().reset_index()
missing_before.columns = ["Column", "Missing_Count"]
missing_before["Missing_Percentage"] = missing_before["Missing_Count"] / len(df_data) * 100
missing_before = missing_before.sort_values("Missing_Count", ascending=False)

display(missing_before)
save_table(missing_before, "01_missing_value_before_cleaning.csv")

# Target cleaning: supervised regression requires a valid target.
df_valid_target = df_data.dropna(subset=[TARGET_COL]).copy()

# Exact duplicates should be removed so identical rows do not appear in both train and test sets.
DROP_EXACT_DUPLICATES = True

if DROP_EXACT_DUPLICATES:
    df_clean = df_valid_target.drop_duplicates().copy()
else:
    df_clean = df_valid_target.copy()

print("\nShape after dropping rows with missing targets:", df_valid_target.shape)
print("Rows removed due to missing target:", len(df_data) - len(df_valid_target))
print("Exact duplicates removed:", len(df_valid_target) - len(df_clean))
print("Final df_clean shape:", df_clean.shape)

missing_after = df_clean.isna().sum().reset_index()
missing_after.columns = ["Column", "Missing_Count"]
missing_after["Missing_Percentage"] = missing_after["Missing_Count"] / len(df_clean) * 100
missing_after = missing_after.sort_values("Missing_Count", ascending=False)

display(missing_after)
save_table(missing_after, "02_missing_value_after_cleaning.csv")

target_stats = (
    df_clean[TARGET_COL]
    .describe(percentiles=[0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99])
    .to_frame()
    .T
)
display(target_stats)
save_table(target_stats, "03_target_descriptive_statistics.csv")

lower, upper, q1, q3, iqr = iqr_bounds(df_clean[TARGET_COL])
target_outlier_mask = (df_clean[TARGET_COL] < lower) | (df_clean[TARGET_COL] > upper)

target_outlier_summary = pd.DataFrame([{
    "Target": TARGET_COL,
    "Q1": q1,
    "Q3": q3,
    "IQR": iqr,
    "Lower_Bound": lower,
    "Upper_Bound": upper,
    "Outlier_Count": int(target_outlier_mask.sum()),
    "Outlier_Percentage": float(target_outlier_mask.mean() * 100),
    "Target_Min": float(df_clean[TARGET_COL].min()),
    "Target_Max": float(df_clean[TARGET_COL].max())
}])

display(target_outlier_summary)
save_table(target_outlier_summary, "04_target_outlier_iqr_summary.csv")

numeric_missing_summary = df_clean[NUMERIC_COLS].isna().sum().reset_index()
numeric_missing_summary.columns = ["Numeric_Feature", "Missing_Count"]
numeric_missing_summary["Missing_Percentage"] = numeric_missing_summary["Missing_Count"] / len(df_clean) * 100
numeric_missing_summary = numeric_missing_summary.sort_values("Missing_Count", ascending=False)

display(numeric_missing_summary)
save_table(numeric_missing_summary, "05_numeric_missing_summary.csv")

plot_hist_target(df_clean[TARGET_COL], "target_distribution_original.png", "(Original)")
plot_hist_target(np.log1p(df_clean[TARGET_COL]), "target_distribution_log1p.png", "(log1p)")

print("\nImportant notes:")
print("- The core_size data uses '?' as a missing-value token and it has been converted to NaN.")
print("- Complete-case/drop is not recommended as the main baseline because missingness is substantial.")
print("- Core size is used as the target variable for silver nanoparticle core size in the Furxhi et al. dataset.")

# %% [markdown]
# ## Cleaning audit before the train-test split
#
# This cell saves the data after dropping missing targets and removing exact duplicates. This is still not model preprocessing; it is the initial clean dataset used for train-test splitting and all experimental scenarios.


# %% [cell 14]
# ============================================================
# CELL 4B - CLEAN DATA SNAPSHOT AFTER MISSING TARGETS AND DUPLICATES ARE REMOVED
# ============================================================

clean_snapshot_path = DATA_SNAPSHOT_DIR / "02_clean_dataset_after_target_drop_and_duplicate_removal_before_split.csv"
df_clean.to_csv(clean_snapshot_path, index=False)
print("Saved:", clean_snapshot_path)

cleaning_flow_df = pd.DataFrame([
    {
        "Step": "Selected target + selected features",
        "Rows": int(len(df_data)),
        "Columns": int(df_data.shape[1]),
        "Rows_Removed_From_Previous": 0,
        "Description": "Dataset after selecting target and 14 article-like features."
    },
    {
        "Step": "Drop rows with missing target",
        "Rows": int(len(df_valid_target)),
        "Columns": int(df_valid_target.shape[1]),
        "Rows_Removed_From_Previous": int(len(df_data) - len(df_valid_target)),
        "Description": "Rows without core size target cannot be used for supervised regression."
    },
    {
        "Step": "Remove exact duplicate rows" if DROP_EXACT_DUPLICATES else "Duplicate removal disabled",
        "Rows": int(len(df_clean)),
        "Columns": int(df_clean.shape[1]),
        "Rows_Removed_From_Previous": int(len(df_valid_target) - len(df_clean)) if DROP_EXACT_DUPLICATES else 0,
        "Description": "Exact duplicates removed before train-test split." if DROP_EXACT_DUPLICATES else "DROP_EXACT_DUPLICATES=False."
    }
])
display(cleaning_flow_df)
save_table(cleaning_flow_df, "02b_data_cleaning_flow_before_split.csv")


# %% [cell 15]
# ============================================================
# CELL 5 - 80:20 TRAIN-TEST SPLIT WITH LOG1P TARGET STRATIFICATION
# ============================================================
# This train-test split serves as a direct comparison against K-Fold Cross Validation.

def make_regression_stratify_bins(y, n_bins=10):
    y_log = np.log1p(np.asarray(y, dtype=float))
    try:
        bins = pd.qcut(y_log, q=n_bins, duplicates="drop")
        counts = pd.Series(bins).value_counts()
        if len(counts) < 2 or counts.min() < 2:
            return None
        return bins
    except Exception:
        return None

X_all = df_clean[FEATURE_COLS].copy()
y_all = df_clean[TARGET_COL].copy()

stratify_bins = make_regression_stratify_bins(y_all, n_bins=10)

X_train, X_test, y_train, y_test = train_test_split(
    X_all,
    y_all,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=stratify_bins
)

split_summary = pd.DataFrame([
    {"Subset": "Train", "Rows": len(X_train), "Percentage": len(X_train) / len(X_all) * 100},
    {"Subset": "Test", "Rows": len(X_test), "Percentage": len(X_test) / len(X_all) * 100},
    {"Subset": "Total", "Rows": len(X_all), "Percentage": 100.0},
])
display(split_summary)
save_table(split_summary, "06_train_test_split_summary.csv")

print("The train-test split is used as the standard holdout evaluation.")
print("K-Fold in the final cells is used as additional generalization validation.")

# %% [markdown]
# ## Train-test snapshot before model preprocessing
#
# The train-test split is performed before baseline, preprocessing, and tuning. This cell saves train and test data before model preprocessing so reviewers can compare initial train/test subsets with results after baseline and preprocessing scenarios.


# %% [cell 17]
# ============================================================
# CELL 5B - TRAIN-TEST SNAPSHOT BEFORE MODEL PREPROCESSING
# ============================================================

train_before_preprocessing_df = X_train.copy()
train_before_preprocessing_df[TARGET_COL] = y_train.values
train_before_preprocessing_df["Original_Index"] = X_train.index

test_before_preprocessing_df = X_test.copy()
test_before_preprocessing_df[TARGET_COL] = y_test.values
test_before_preprocessing_df["Original_Index"] = X_test.index

train_before_path = DATA_SNAPSHOT_DIR / "03_train_before_any_model_preprocessing.csv"
test_before_path = DATA_SNAPSHOT_DIR / "04_test_before_any_model_preprocessing.csv"
train_before_preprocessing_df.to_csv(train_before_path, index=False)
test_before_preprocessing_df.to_csv(test_before_path, index=False)
print("Saved:", train_before_path)
print("Saved:", test_before_path)

split_missing_by_subset_df = pd.concat([
    make_column_profile(train_before_preprocessing_df.drop(columns=["Original_Index"]), "train_before_model_preprocessing"),
    make_column_profile(test_before_preprocessing_df.drop(columns=["Original_Index"]), "test_before_model_preprocessing"),
], ignore_index=True)
display(split_missing_by_subset_df)
save_table(split_missing_by_subset_df, "06b_train_test_column_profile_before_model_preprocessing.csv")


# %% [cell 18]
# ============================================================
# CELL 6 - PREPROCESSING, TARGET TRANSFORMATION, OUTLIER HANDLING, AND MODEL FUNCTIONS
# ============================================================

# Physical prediction bounds for core size.
# This prevents overflow during inverse log1p transformation in unstable models.
try:
    _target_reference = pd.to_numeric(y_train, errors="coerce")
except Exception:
    _target_reference = pd.to_numeric(df_clean[TARGET_COL], errors="coerce")

_target_reference = pd.Series(_target_reference).dropna()

TARGET_ORIGINAL_LOWER_LIMIT = 0.0
TARGET_ORIGINAL_UPPER_LIMIT = float(
    max(
        _target_reference.max() * 3,
        _target_reference.quantile(0.99) * 5,
        500.0
    )
)

TARGET_LOG1P_LOWER_CLIP = np.log1p(TARGET_ORIGINAL_LOWER_LIMIT)
TARGET_LOG1P_UPPER_CLIP = np.log1p(TARGET_ORIGINAL_UPPER_LIMIT)

print("Original-scale prediction bounds for target inversion:")
print("Lower:", TARGET_ORIGINAL_LOWER_LIMIT)
print("Upper:", TARGET_ORIGINAL_UPPER_LIMIT)
print("Log1p lower clip:", TARGET_LOG1P_LOWER_CLIP)
print("Log1p upper clip:", TARGET_LOG1P_UPPER_CLIP)


def transform_target(y, mode="none"):
    y = np.asarray(y, dtype=float)

    if mode == "none":
        return y

    if mode == "log1p":
        if np.any(y < -1):
            raise ValueError("log1p is not valid for target values < -1.")
        return np.log1p(y)

    raise ValueError(f"Unknown target transformation: {mode}")


def inverse_transform_target(y_transformed, mode="none"):
    y_transformed = np.asarray(y_transformed, dtype=float)

    if mode == "none":
        # Do not alter normal predictions; only guard against NaN/inf values.
        return np.nan_to_num(
            y_transformed,
            nan=np.nanmedian(_target_reference),
            posinf=TARGET_ORIGINAL_UPPER_LIMIT,
            neginf=TARGET_ORIGINAL_LOWER_LIMIT
        )

    if mode == "log1p":
        # Predictions from log1p models must not produce physically negative sizes or values large enough to cause expm1 overflow.
        y_safe = np.nan_to_num(
            y_transformed,
            nan=TARGET_LOG1P_LOWER_CLIP,
            posinf=TARGET_LOG1P_UPPER_CLIP,
            neginf=TARGET_LOG1P_LOWER_CLIP
        )

        y_safe = np.clip(
            y_safe,
            TARGET_LOG1P_LOWER_CLIP,
            TARGET_LOG1P_UPPER_CLIP
        )

        y_original = np.expm1(y_safe)

        return np.clip(
            y_original,
            TARGET_ORIGINAL_LOWER_LIMIT,
            TARGET_ORIGINAL_UPPER_LIMIT
        )

    raise ValueError(f"Unknown target transformation: {mode}")


def get_onehot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_numeric_imputer(imputer_mode):
    if imputer_mode == "drop":
        return SimpleImputer(strategy="median")

    if imputer_mode == "mean":
        return SimpleImputer(strategy="mean")

    if imputer_mode == "median":
        return SimpleImputer(strategy="median")

    if imputer_mode == "most_frequent":
        return SimpleImputer(strategy="most_frequent")

    if imputer_mode == "constant_zero":
        return SimpleImputer(strategy="constant", fill_value=0)


    if imputer_mode == "iterative":
        return IterativeImputer(
            random_state=RANDOM_STATE,
            max_iter=20,
            sample_posterior=False
        )

    raise ValueError(f"Unknown imputer: {imputer_mode}")


def build_scaler(scaler_type, n_rows_for_quantile=None):
    if scaler_type == "none":
        return None

    if scaler_type == "standard":
        return StandardScaler()

    if scaler_type == "minmax":
        return MinMaxScaler()

    if scaler_type == "robust":
        return RobustScaler()

    if scaler_type == "maxabs":
        return MaxAbsScaler()

    if scaler_type == "quantile_normal":
        n_q = 100 if n_rows_for_quantile is None else min(100, max(10, n_rows_for_quantile))
        return QuantileTransformer(
            n_quantiles=n_q,
            output_distribution="normal",
            random_state=RANDOM_STATE
        )

    if scaler_type == "power_yeo":
        return PowerTransformer(method="yeo-johnson", standardize=True)

    raise ValueError(f"Unknown scaler: {scaler_type}")


def build_preprocessor(imputer_mode="median", scaler_type="none", n_rows_for_quantile=None, selected_features=None):
    selected_features = FEATURE_COLS if selected_features is None else list(selected_features)

    numeric_cols = [c for c in selected_features if c in NUMERIC_COLS]
    categorical_cols = [c for c in selected_features if c in CATEGORICAL_COLS]

    numeric_steps = [("imputer", build_numeric_imputer(imputer_mode))]

    scaler = build_scaler(scaler_type, n_rows_for_quantile=n_rows_for_quantile)
    if scaler is not None:
        numeric_steps.append(("scaler", scaler))

    categorical_steps = [
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", get_onehot_encoder())
    ]

    transformers = []

    if numeric_cols:
        transformers.append(("num", Pipeline(numeric_steps), numeric_cols))

    if categorical_cols:
        transformers.append(("cat", Pipeline(categorical_steps), categorical_cols))

    return ColumnTransformer(transformers=transformers, remainder="drop")


def get_base_model(model_name):
    if model_name == "MLR":
        return LinearRegression()

    if model_name == "Ridge Regression":
        # Linear model with L2 regularization.
        # Used as an MLR comparator to evaluate the effect of regularization on tabular silver nanoparticle synthesis data.
        return Ridge(random_state=RANDOM_STATE)

    if model_name == "SVR":
        return SVR()

    if model_name == "XGBoost":
        return XGBRegressor(
            objective="reg:squarederror",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=0
        )

    if model_name == "Extra Trees Regression":
        # Similar to Random Forest, but with stronger split randomization.
        return ExtraTreesRegressor(
            random_state=RANDOM_STATE,
            n_jobs=-1
        )

    if model_name == "Random Forest Regression":
        return RandomForestRegressor(
            random_state=RANDOM_STATE,
            n_jobs=-1
        )

    raise ValueError(f"Unknown model: {model_name}")


def get_model_names():
    names = [
        "MLR",
        "Ridge Regression",
        "SVR",
        "XGBoost",
        "Extra Trees Regression"
    ]

    if INCLUDE_RANDOM_FOREST:
        names.append("Random Forest Regression")

    return names

# ============================================================
# DEFAULT BASELINE PARAMETERS FOR REPORTING
# ============================================================

DEFAULT_PARAM_KEYS_FOR_REPORT = {
    "MLR": [
        "fit_intercept",
        "positive"
    ],

    "Ridge Regression": [
        "alpha",
        "fit_intercept",
        "positive",
        "solver"
    ],

    "SVR": [
        "kernel",
        "C",
        "epsilon",
        "gamma",
        "degree"
    ],

    "XGBoost": [
        "n_estimators",
        "max_depth",
        "learning_rate",
        "subsample",
        "colsample_bytree",
        "min_child_weight",
        "gamma",
        "reg_alpha",
        "reg_lambda"
    ],

    "Extra Trees Regression": [
        "n_estimators",
        "max_depth",
        "min_samples_split",
        "min_samples_leaf",
        "max_features",
        "bootstrap"
    ],

    "Random Forest Regression": [
        "n_estimators",
        "max_depth",
        "min_samples_split",
        "min_samples_leaf",
        "max_features",
        "bootstrap"
    ]
}


def get_report_default_params(model_name):
    """
    Retrieve important default baseline parameters for manuscript reporting.
    This function does not modify the model; it only reads parameters from get_base_model().
    """
    model = get_base_model(model_name)
    params = model.get_params()

    selected_keys = DEFAULT_PARAM_KEYS_FOR_REPORT.get(model_name, [])
    selected_params = {}

    for key in selected_keys:
        selected_params[key] = params.get(key, None)

    return selected_params


def build_full_pipeline(model_name, imputer_mode, scaler_type, n_rows_for_quantile=None, selected_features=None):
    preprocessor = build_preprocessor(
        imputer_mode=imputer_mode,
        scaler_type=scaler_type,
        n_rows_for_quantile=n_rows_for_quantile,
        selected_features=selected_features
    )

    model = get_base_model(model_name)

    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])


def apply_missing_drop_to_training(X_train_df, y_train_array, imputer_mode):
    X_train_df = X_train_df.copy()
    y_train_array = np.asarray(y_train_array, dtype=float)

    if imputer_mode != "drop":
        return X_train_df, y_train_array, {"Missing_Train_Dropped_Rows": 0}

    row_mask = ~X_train_df.isna().any(axis=1)
    dropped = int((~row_mask).sum())

    return (
        X_train_df.loc[row_mask].copy(),
        y_train_array[row_mask.values],
        {"Missing_Train_Dropped_Rows": dropped}
    )


def apply_target_outlier_to_training(y_train_array, mode="none"):
    y = np.asarray(y_train_array, dtype=float).copy()

    info = {
        "Target_Outlier_Train_Count": 0,
        "Target_Outlier_Lower": np.nan,
        "Target_Outlier_Upper": np.nan
    }

    if mode == "none":
        return np.ones(len(y), dtype=bool), y, info

    if mode in ["remove_iqr", "cap_iqr", "median_iqr"]:
        lower, upper, q1, q3, iqr = iqr_bounds(pd.Series(y))

    elif mode == "winsor_p01_p99":
        lower, upper = np.nanpercentile(y, [1, 99])

    else:
        raise ValueError(f"Unknown target-outlier mode: {mode}")

    outlier_mask = (y < lower) | (y > upper)

    info["Target_Outlier_Train_Count"] = int(outlier_mask.sum())
    info["Target_Outlier_Lower"] = float(lower)
    info["Target_Outlier_Upper"] = float(upper)

    if mode == "remove_iqr":
        keep_mask = ~outlier_mask
        return keep_mask, y[keep_mask], info

    if mode in ["cap_iqr", "winsor_p01_p99"]:
        y = np.clip(y, lower, upper)
        return np.ones(len(y), dtype=bool), y, info

    if mode == "median_iqr":
        inlier_median = np.median(y[~outlier_mask]) if (~outlier_mask).sum() > 0 else np.median(y)
        y[outlier_mask] = inlier_median
        return np.ones(len(y), dtype=bool), y, info


def apply_feature_outlier_fit_transform(X_train_df, X_eval_df, mode="none"):
    X_train_df = X_train_df.copy()
    X_eval_df = X_eval_df.copy()

    info = {
        "Feature_Outlier_Cell_Capped_Train": 0,
        "Feature_Outlier_Mode": mode,
        "Feature_Outlier_Params": {}
    }

    if mode == "none":
        return X_train_df, X_eval_df, info

    params = {}
    train_cap_count = 0

    for col in NUMERIC_COLS:
        if col not in X_train_df.columns:
            continue

        s = X_train_df[col]

        if s.dropna().empty:
            continue

        if mode == "feature_cap_iqr":
            lower, upper, *_ = iqr_bounds(s.dropna())

        elif mode == "feature_winsor_p01_p99":
            lower, upper = np.nanpercentile(s.dropna(), [1, 99])

        else:
            raise ValueError(f"Unknown feature-outlier mode: {mode}")

        original = X_train_df[col].copy()

        X_train_df[col] = X_train_df[col].clip(lower, upper)
        X_eval_df[col] = X_eval_df[col].clip(lower, upper)

        train_cap_count += int(((original < lower) | (original > upper)).sum())
        params[col] = {"lower": float(lower), "upper": float(upper)}

    info["Feature_Outlier_Cell_Capped_Train"] = int(train_cap_count)
    info["Feature_Outlier_Params"] = params

    return X_train_df, X_eval_df, info


def apply_feature_outlier_params(X_df, feature_outlier_params):
    X_df = X_df.copy()

    if not feature_outlier_params:
        return X_df

    for col, bounds in feature_outlier_params.items():
        if col in X_df.columns:
            lower = bounds.get("lower", None)
            upper = bounds.get("upper", None)

            if lower is not None and upper is not None:
                X_df[col] = X_df[col].clip(lower, upper)

    return X_df


def metric_dict(y_true_original, y_pred_original, prefix=""):
    y_true_original = np.asarray(y_true_original, dtype=float)
    y_pred_original = np.asarray(y_pred_original, dtype=float)

    # Guard against NaN/inf values before computing metrics.
    y_pred_original = np.nan_to_num(
        y_pred_original,
        nan=np.nanmedian(y_true_original),
        posinf=TARGET_ORIGINAL_UPPER_LIMIT,
        neginf=TARGET_ORIGINAL_LOWER_LIMIT
    )

    mse = mean_squared_error(y_true_original, y_pred_original)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true_original, y_pred_original)
    r2 = r2_score(y_true_original, y_pred_original)

    return {
        f"{prefix}R2_Original": r2,
        f"{prefix}MSE_Original": mse,
        f"{prefix}RMSE_Original": rmse,
        f"{prefix}MAE_Original": mae
    }


def safe_json(obj):
    try:
        return json.dumps(obj, ensure_ascii=False)
    except Exception:
        return str(obj)


# %% [cell 19]
# ============================================================
# CELL 7 - TRAIN-TEST SPLIT AND K-FOLD EVALUATION FUNCTIONS
# ============================================================

def fit_evaluate_train_test(
    model_name,
    scenario_name,
    X_train_df,
    y_train_series,
    X_test_df,
    y_test_series,
    imputer_mode="median",
    scaler_type="none",
    target_transform="none",
    target_outlier_mode="none",
    feature_outlier_mode="none",
    model_params=None,
    selected_features=None,
    tuned=False,
    stage="baseline"
):
    model_params = {} if model_params is None else dict(model_params)
    selected_features = FEATURE_COLS if selected_features is None else list(selected_features)

    Xtr = X_train_df[selected_features].copy()
    Xte = X_test_df[selected_features].copy()
    ytr_original = np.asarray(y_train_series, dtype=float)
    yte_original = np.asarray(y_test_series, dtype=float)

    # 1. Missing handling.
    # Baseline with imputer_mode='drop': train and test rows that still contain missing values are removed so the baseline is truly run without imputation. For preprocessing scenarios, missing values are handled by the imputer.
    if imputer_mode == "drop":
        train_mask = ~Xtr.isna().any(axis=1)
        test_mask = ~Xte.isna().any(axis=1)
        missing_info = {
            "Missing_Train_Dropped_Rows": int((~train_mask).sum()),
            "Missing_Test_Dropped_Rows": int((~test_mask).sum()),
            "Complete_Case_Baseline": True
        }
        Xtr = Xtr.loc[train_mask].copy()
        ytr_original = ytr_original[train_mask.values]
        Xte = Xte.loc[test_mask].copy()
        yte_original = yte_original[test_mask.values]
    else:
        # For non-drop scenarios, missing values are handled by the imputer.
        # within the preprocessing pipeline.
        Xtr, ytr_original, missing_info = apply_missing_drop_to_training(
            Xtr,
            ytr_original,
            imputer_mode
        )
        missing_info["Missing_Val_Dropped_Rows"] = 0
        missing_info["Missing_Test_Dropped_Rows"] = 0
        missing_info["Complete_Case_Baseline"] = False

    # 2. Target outlier handling is based only on training data.
    keep_mask, ytr_after_target_outlier, target_info = apply_target_outlier_to_training(ytr_original, target_outlier_mode)
    if len(keep_mask) == len(Xtr) and not np.all(keep_mask):
        Xtr = Xtr.loc[keep_mask].copy()
    ytr_original_for_metric = ytr_after_target_outlier.copy()

    # 3. Feature-outlier thresholds are computed on training data, then applied to training and test data.
    Xtr, Xte, feature_info = apply_feature_outlier_fit_transform(Xtr, Xte, feature_outlier_mode)

    # 4. Target transformation.
    ytr_transformed = transform_target(ytr_after_target_outlier, target_transform)
    yte_transformed = transform_target(yte_original, target_transform)

    # 5. Fit the model pipeline.
    pipe = build_full_pipeline(
        model_name=model_name,
        imputer_mode=imputer_mode,
        scaler_type=scaler_type,
        n_rows_for_quantile=len(Xtr),
        selected_features=selected_features
    )
    if model_params:
        pipe.set_params(**model_params)

    start = time.time()
    pipe.fit(Xtr, ytr_transformed)
    fit_seconds = time.time() - start

    pred_train_t = pipe.predict(Xtr)
    pred_test_t = pipe.predict(Xte)

    pred_train_original = inverse_transform_target(pred_train_t, target_transform)
    pred_test_original = inverse_transform_target(pred_test_t, target_transform)

    row = {
        "Stage": stage,
        "Scenario_Name": scenario_name,
        "Model": model_name,
        "Tuned": bool(tuned),
        "Imputer_Mode": imputer_mode,
        "Scaler_Type": scaler_type,
        "Target_Transform": target_transform,
        "Target_Outlier_Mode": target_outlier_mode,
        "Feature_Outlier_Mode": feature_outlier_mode,
        "Model_Params": safe_json(model_params),
        "Selected_Feature_Count": len(selected_features),
        "Selected_Features": safe_json(selected_features),
        "Train_Rows_Original": len(X_train_df),
        "Train_Rows_After_Missing": len(Xtr) + target_info.get("Target_Outlier_Train_Count", 0) if target_outlier_mode == "remove_iqr" else len(Xtr),
        "Train_Rows_Final": len(Xtr),
        "Test_Rows": len(Xte),
        "Fit_Seconds": fit_seconds,
        **missing_info,
        **{k: v for k, v in target_info.items() if k != "Feature_Outlier_Params"},
        **{k: v for k, v in feature_info.items() if k != "Feature_Outlier_Params"}
    }

    # Metrics transformed.
    row["R2_Train_Transformed"] = r2_score(ytr_transformed, pred_train_t)
    row["R2_Test_Transformed"] = r2_score(yte_transformed, pred_test_t)
    row["RMSE_Test_Transformed"] = np.sqrt(mean_squared_error(yte_transformed, pred_test_t))
    row["MAE_Test_Transformed"] = mean_absolute_error(yte_transformed, pred_test_t)

    # Metrics original scale.
    row.update(metric_dict(ytr_original_for_metric, pred_train_original, prefix="Train_"))
    row.update(metric_dict(yte_original, pred_test_original, prefix="Test_"))

    row["Train_Test_Gap_R2_Original"] = row["Train_R2_Original"] - row["Test_R2_Original"]
    row["Train_Test_Gap_R2_Transformed"] = row["R2_Train_Transformed"] - row["R2_Test_Transformed"]

    predictions_df = pd.DataFrame({
        PRED_ACTUAL_COL: yte_original,
        PRED_PRED_COL: pred_test_original,
        "Actual_Target_Transformed": yte_transformed,
        "Predicted_Target_Transformed": pred_test_t,
        "Residual_nm": yte_original - pred_test_original,
        "Absolute_Error_nm": np.abs(yte_original - pred_test_original)
    })

    return row, pipe, predictions_df

def kfold_cv_score_for_config(
    model_name,
    X_df,
    y_series,
    imputer_mode="median",
    scaler_type="none",
    target_transform="none",
    target_outlier_mode="none",
    feature_outlier_mode="none",
    model_params=None,
    selected_features=None,
    n_splits=5,
    random_state=RANDOM_STATE
):
    model_params = {} if model_params is None else dict(model_params)
    selected_features = FEATURE_COLS if selected_features is None else list(selected_features)

    X_df = X_df[selected_features].copy()
    y_array = np.asarray(y_series, dtype=float)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_rows = []

    for fold, (tr_idx, val_idx) in enumerate(kf.split(np.arange(len(X_df))), start=1):
        Xtr = X_df.iloc[tr_idx].copy()
        Xval = X_df.iloc[val_idx].copy()
        ytr_original = y_array[tr_idx]
        yval_original = y_array[val_idx]

        if imputer_mode == "drop":
            tr_mask = ~Xtr.isna().any(axis=1)
            val_mask = ~Xval.isna().any(axis=1)
            missing_info = {
                "Missing_Train_Dropped_Rows": int((~tr_mask).sum()),
                "Missing_Val_Dropped_Rows": int((~val_mask).sum()),
                "Complete_Case_Baseline": True
            }
            Xtr = Xtr.loc[tr_mask].copy()
            ytr_original = ytr_original[tr_mask.values]
            Xval = Xval.loc[val_mask].copy()
            yval_original = yval_original[val_mask.values]
        else:
            Xtr, ytr_original, missing_info = apply_missing_drop_to_training(Xtr, ytr_original, imputer_mode)
            missing_info["Missing_Val_Dropped_Rows"] = 0
            missing_info["Complete_Case_Baseline"] = False

        keep_mask, ytr_after_target_outlier, target_info = apply_target_outlier_to_training(ytr_original, target_outlier_mode)
        if len(keep_mask) == len(Xtr) and not np.all(keep_mask):
            Xtr = Xtr.loc[keep_mask].copy()

        Xtr, Xval, feature_info = apply_feature_outlier_fit_transform(Xtr, Xval, feature_outlier_mode)

        ytr_t = transform_target(ytr_after_target_outlier, target_transform)
        yval_t = transform_target(yval_original, target_transform)

        pipe = build_full_pipeline(
            model_name=model_name,
            imputer_mode=imputer_mode,
            scaler_type=scaler_type,
            n_rows_for_quantile=len(Xtr),
            selected_features=selected_features
        )
        if model_params:
            pipe.set_params(**model_params)

        pipe.fit(Xtr, ytr_t)

        # Predict on the training fold and validation fold.
        pred_train_t = pipe.predict(Xtr)
        pred_val_t = pipe.predict(Xval)

        pred_train_original = inverse_transform_target(pred_train_t, target_transform)
        pred_val_original = inverse_transform_target(pred_val_t, target_transform)

        fold_row = {
            "Model": model_name,
            "Fold": fold,
            "K": n_splits,

            # Train and validation R2 on the transformed scale.
            "R2_Train_Transformed": r2_score(ytr_t, pred_train_t),
            "R2_Val_Transformed": r2_score(yval_t, pred_val_t),

            # Legacy columns are retained to avoid breaking downstream cells.
            # R2_Transformed = validation R2 transformed.
            "R2_Transformed": r2_score(yval_t, pred_val_t),

            "RMSE_Transformed": np.sqrt(mean_squared_error(yval_t, pred_val_t)),
            "MAE_Transformed": mean_absolute_error(yval_t, pred_val_t),

            # Train and validation R2 on the original scale.
            "R2_Train_Original": r2_score(ytr_after_target_outlier, pred_train_original),
            "R2_Val_Original": r2_score(yval_original, pred_val_original),

            # Legacy columns are retained to avoid breaking downstream cells.
            # R2_Original = validation R2 original.
            "R2_Original": r2_score(yval_original, pred_val_original),

            "RMSE_Original": np.sqrt(mean_squared_error(yval_original, pred_val_original)),
            "MAE_Original": mean_absolute_error(yval_original, pred_val_original),

            "Train_Val_Gap_R2_Transformed": r2_score(ytr_t, pred_train_t) - r2_score(yval_t, pred_val_t),
            "Train_Val_Gap_R2_Original": r2_score(ytr_after_target_outlier, pred_train_original) - r2_score(yval_original, pred_val_original),

            "Train_Rows_Fold_Final": len(Xtr),
            **missing_info,
            **{k: v for k, v in target_info.items() if k != "Feature_Outlier_Params"},
            **{k: v for k, v in feature_info.items() if k != "Feature_Outlier_Params"}
        }

        fold_rows.append(fold_row)

    fold_df = pd.DataFrame(fold_rows)
    summary = {
        "CV_K": n_splits,

        # Validation/CV R2 transformed.
        "CV_Mean_R2_Transformed": fold_df["R2_Transformed"].mean(),
        "CV_Std_R2_Transformed": fold_df["R2_Transformed"].std(ddof=1),

        # Transformed-scale train R2 within CV folds.
        "CV_Mean_Train_R2_Transformed": fold_df["R2_Train_Transformed"].mean(),
        "CV_Std_Train_R2_Transformed": fold_df["R2_Train_Transformed"].std(ddof=1),

        # Gap train-validation transformed.
        "CV_Mean_Train_Val_Gap_R2_Transformed": fold_df["Train_Val_Gap_R2_Transformed"].mean(),

        # Validation/CV R2 original.
        "CV_Mean_R2_Original": fold_df["R2_Original"].mean(),
        "CV_Std_R2_Original": fold_df["R2_Original"].std(ddof=1),

        # Original-scale train R2 within CV folds.
        "CV_Mean_Train_R2_Original": fold_df["R2_Train_Original"].mean(),
        "CV_Std_Train_R2_Original": fold_df["R2_Train_Original"].std(ddof=1),

        # Gap train-validation original.
        "CV_Mean_Train_Val_Gap_R2_Original": fold_df["Train_Val_Gap_R2_Original"].mean(),

        "CV_Mean_RMSE_Original": fold_df["RMSE_Original"].mean(),
        "CV_Std_RMSE_Original": fold_df["RMSE_Original"].std(ddof=1),
        "CV_Mean_MAE_Original": fold_df["MAE_Original"].mean(),
        "CV_Std_MAE_Original": fold_df["MAE_Original"].std(ddof=1),
    }
    return summary, fold_df

def evaluate_scenario_with_cv(
    model_name,
    scenario_name,
    stage,
    imputer_mode,
    scaler_type,
    target_transform,
    target_outlier_mode,
    feature_outlier_mode,
    model_params=None,
    selected_features=None,
    tuned=False,
    cv_k=CV_TUNING
):
    cv_summary, fold_df = kfold_cv_score_for_config(
        model_name=model_name,
        X_df=X_train,
        y_series=y_train,
        imputer_mode=imputer_mode,
        scaler_type=scaler_type,
        target_transform=target_transform,
        target_outlier_mode=target_outlier_mode,
        feature_outlier_mode=feature_outlier_mode,
        model_params=model_params,
        selected_features=selected_features,
        n_splits=cv_k
    )

    tt_row, fitted_pipe, pred_df = fit_evaluate_train_test(
        model_name=model_name,
        scenario_name=scenario_name,
        X_train_df=X_train,
        y_train_series=y_train,
        X_test_df=X_test,
        y_test_series=y_test,
        imputer_mode=imputer_mode,
        scaler_type=scaler_type,
        target_transform=target_transform,
        target_outlier_mode=target_outlier_mode,
        feature_outlier_mode=feature_outlier_mode,
        model_params=model_params,
        selected_features=selected_features,
        tuned=tuned,
        stage=stage
    )

    tt_row.update(cv_summary)
    return tt_row, fitted_pipe, pred_df, fold_df


def add_overfitting_control_scores(df):
    """
    Add stability scores and overfitting flags to the evaluation-result DataFrame.
    """
    df = df.copy()

    if "CV_Mean_R2_Original" not in df.columns:
        df["CV_Stability_Score_Original"] = np.nan
        return df

    cv_r2 = pd.to_numeric(df["CV_Mean_R2_Original"], errors="coerce")

    if "CV_Mean_Train_Val_Gap_R2_Original" in df.columns:
        train_val_gap = pd.to_numeric(df["CV_Mean_Train_Val_Gap_R2_Original"], errors="coerce").abs()
    else:
        train_val_gap = pd.Series(0.0, index=df.index)

    if "CV_Std_R2_Original" in df.columns:
        cv_std = pd.to_numeric(df["CV_Std_R2_Original"], errors="coerce").abs()
    else:
        cv_std = pd.Series(0.0, index=df.index)

    excess_gap = (train_val_gap - MAX_ALLOWED_TRAIN_VAL_GAP_R2).clip(lower=0)

    df["CV_Stability_Score_Original"] = (
        cv_r2
        - OVERFIT_GAP_PENALTY_WEIGHT * excess_gap
        - CV_STD_PENALTY_WEIGHT * cv_std
    )

    df["CV_Overfit_Flag"] = train_val_gap > MAX_ALLOWED_TRAIN_VAL_GAP_R2

    if "Train_Test_Gap_R2_Original" in df.columns:
        holdout_gap = pd.to_numeric(df["Train_Test_Gap_R2_Original"], errors="coerce").abs()
        df["Holdout_Overfit_Flag"] = holdout_gap > MAX_ALLOWED_TRAIN_HOLDOUT_GAP_R2
    else:
        df["Holdout_Overfit_Flag"] = False

    return df

# %% [markdown]
# ## Preprocessing audit functions and baseline snapshot
#
# The following cell reconstructs the same preprocessing stages used by the model-evaluation function, but without changing the training process. The output is used to document:
#
# - which rows enter the baseline before preprocessing;
# - how many rows are removed by complete-case baseline processing;
# - the shape of baseline data after dropping missing values;
# - the matrix shape after imputation/encoding/scaling;
# - transformation summaries for other preprocessing scenarios.


# %% [cell 21]
# ============================================================
# CELL 7B - PREPROCESSING AUDIT FUNCTIONS AND BASELINE SNAPSHOT
# ============================================================
# This cell does not train models. It only uses the same preprocessing functions
# to create data snapshots/audits before and after preprocessing.

EXPORT_SELECTED_PREPROCESSED_MATRICES = True
EXPORT_ALL_UNIQUE_PREPROCESSING_MATRICES = False
# Change this to True only when exporting full matrices for ALL unique combinations.
# For FULL_COMBINED_GRID, the number of files can be very large.


def to_dense_array(X):
    if hasattr(X, "toarray"):
        return X.toarray()
    return np.asarray(X)


def get_preprocessor_feature_names(fitted_preprocessor, selected_features=None):
    selected_features = FEATURE_COLS if selected_features is None else list(selected_features)
    try:
        return list(fitted_preprocessor.get_feature_names_out())
    except Exception:
        names = []
        numeric_cols = [c for c in selected_features if c in NUMERIC_COLS]
        categorical_cols = [c for c in selected_features if c in CATEGORICAL_COLS]
        names.extend([f"num__{c}" for c in numeric_cols])
        try:
            cat_pipe = fitted_preprocessor.named_transformers_.get("cat", None)
            if cat_pipe is not None:
                onehot = cat_pipe.named_steps.get("onehot", None)
                if onehot is not None:
                    cat_names = list(onehot.get_feature_names_out(categorical_cols))
                    names.extend([f"cat__{c}" for c in cat_names])
        except Exception:
            names.extend([f"cat__{c}" for c in categorical_cols])
        return names


def make_xy_snapshot(X_df, y_original, y_transformed=None, label=""):
    out = X_df.copy()
    out[TARGET_COL] = np.asarray(y_original, dtype=float)
    if y_transformed is not None:
        out[f"{TARGET_COL}_Transformed"] = np.asarray(y_transformed, dtype=float)
    out["Original_Index"] = X_df.index
    if label:
        out["Snapshot_Label"] = label
    return out


def export_preprocessing_snapshot(
    config,
    stage="audit",
    scenario_name="scenario",
    model_name="not_model_specific",
    selected_features=None,
    save_prefix=None,
    export_full_matrices=False,
    export_preview=True,
    preview_rows=50,
):
    """Reconstruct preprocessing state using the same helper functions as model evaluation."""
    selected_features = FEATURE_COLS if selected_features is None else list(selected_features)

    imputer_mode = config.get("imputer_mode", config.get("Imputer_Mode", "median"))
    scaler_type = config.get("scaler_type", config.get("Scaler_Type", "none"))
    target_transform = config.get("target_transform", config.get("Target_Transform", "none"))
    target_outlier_mode = config.get("target_outlier_mode", config.get("Target_Outlier_Mode", "none"))
    feature_outlier_mode = config.get("feature_outlier_mode", config.get("Feature_Outlier_Mode", "none"))

    Xtr0 = X_train[selected_features].copy()
    Xte0 = X_test[selected_features].copy()
    ytr0 = np.asarray(y_train, dtype=float)
    yte0 = np.asarray(y_test, dtype=float)

    meta = {
        "Stage": stage,
        "Scenario_Name": scenario_name,
        "Model": model_name,
        "Imputer_Mode": imputer_mode,
        "Scaler_Type": scaler_type,
        "Target_Transform": target_transform,
        "Target_Outlier_Mode": target_outlier_mode,
        "Feature_Outlier_Mode": feature_outlier_mode,
        "Selected_Feature_Count": len(selected_features),
        "Train_Rows_Before_Preprocessing": int(len(Xtr0)),
        "Test_Rows_Before_Preprocessing": int(len(Xte0)),
        "Train_Missing_Cells_Before_Preprocessing": int(Xtr0.isna().sum().sum()),
        "Test_Missing_Cells_Before_Preprocessing": int(Xte0.isna().sum().sum()),
        "Train_Rows_With_Any_Missing_Before_Preprocessing": int(Xtr0.isna().any(axis=1).sum()),
        "Test_Rows_With_Any_Missing_Before_Preprocessing": int(Xte0.isna().any(axis=1).sum()),
    }

    # 1. Missing handling mengikuti fit_evaluate_train_test.
    if imputer_mode == "drop":
        train_mask = ~Xtr0.isna().any(axis=1)
        test_mask = ~Xte0.isna().any(axis=1)
        Xtr = Xtr0.loc[train_mask].copy()
        Xte = Xte0.loc[test_mask].copy()
        ytr = ytr0[train_mask.values]
        yte = yte0[test_mask.values]
        missing_info = {
            "Missing_Train_Dropped_Rows": int((~train_mask).sum()),
            "Missing_Test_Dropped_Rows": int((~test_mask).sum()),
            "Complete_Case_Baseline": True,
        }
    else:
        Xtr, ytr, missing_info = apply_missing_drop_to_training(Xtr0, ytr0, imputer_mode)
        Xte = Xte0.copy()
        yte = yte0.copy()
        missing_info.update({
            "Missing_Test_Dropped_Rows": 0,
            "Complete_Case_Baseline": False,
        })

    meta.update(missing_info)
    meta["Train_Rows_After_Missing_Handling"] = int(len(Xtr))
    meta["Test_Rows_After_Missing_Handling"] = int(len(Xte))

    # 2. Target-outlier handling is derived only from training data.
    keep_mask, ytr_after_target_outlier, target_info = apply_target_outlier_to_training(ytr, target_outlier_mode)
    if len(keep_mask) == len(Xtr) and not np.all(keep_mask):
        Xtr = Xtr.loc[keep_mask].copy()
    ytr = np.asarray(ytr_after_target_outlier, dtype=float)
    meta.update(target_info)
    meta["Train_Rows_After_Target_Outlier"] = int(len(Xtr))

    # 3. Feature-outlier handling is fitted on training data and applied to train-test data.
    Xtr_after_feature, Xte_after_feature, feature_info = apply_feature_outlier_fit_transform(
        Xtr, Xte, feature_outlier_mode
    )
    meta.update({k: v for k, v in feature_info.items() if k != "Feature_Outlier_Params"})
    meta["Feature_Outlier_Params_JSON"] = safe_json(feature_info.get("Feature_Outlier_Params", {}))

    # 4. Target transform.
    ytr_transformed = transform_target(ytr, target_transform)
    yte_transformed = transform_target(yte, target_transform)

    # 5. Fit the preprocessor only on training data, then transform train-test data.
    preprocessor = build_preprocessor(
        imputer_mode=imputer_mode,
        scaler_type=scaler_type,
        n_rows_for_quantile=len(Xtr_after_feature),
        selected_features=selected_features,
    )
    Xtr_encoded = to_dense_array(preprocessor.fit_transform(Xtr_after_feature))
    Xte_encoded = to_dense_array(preprocessor.transform(Xte_after_feature))
    encoded_feature_names = get_preprocessor_feature_names(preprocessor, selected_features=selected_features)

    meta["Train_Rows_After_Feature_Outlier"] = int(len(Xtr_after_feature))
    meta["Test_Rows_After_Feature_Outlier"] = int(len(Xte_after_feature))
    meta["Encoded_Feature_Count"] = int(Xtr_encoded.shape[1])
    meta["Train_Encoded_Shape"] = str(tuple(Xtr_encoded.shape))
    meta["Test_Encoded_Shape"] = str(tuple(Xte_encoded.shape))
    meta["Encoded_Missing_Cells_Train"] = int(np.isnan(Xtr_encoded).sum())
    meta["Encoded_Missing_Cells_Test"] = int(np.isnan(Xte_encoded).sum())
    meta["Target_Train_Min_After_Preprocessing"] = float(np.nanmin(ytr)) if len(ytr) else np.nan
    meta["Target_Train_Max_After_Preprocessing"] = float(np.nanmax(ytr)) if len(ytr) else np.nan
    meta["Target_Test_Min_After_Preprocessing"] = float(np.nanmin(yte)) if len(yte) else np.nan
    meta["Target_Test_Max_After_Preprocessing"] = float(np.nanmax(yte)) if len(yte) else np.nan

    objects = {
        "X_train_before": Xtr0,
        "X_test_before": Xte0,
        "X_train_after_unencoded": Xtr_after_feature,
        "X_test_after_unencoded": Xte_after_feature,
        "y_train_after": ytr,
        "y_test_after": yte,
        "y_train_transformed": ytr_transformed,
        "y_test_transformed": yte_transformed,
        "X_train_encoded": pd.DataFrame(Xtr_encoded, columns=encoded_feature_names, index=Xtr_after_feature.index),
        "X_test_encoded": pd.DataFrame(Xte_encoded, columns=encoded_feature_names, index=Xte_after_feature.index),
        "preprocessor": preprocessor,
    }

    if save_prefix is not None:
        prefix = reviewer_safe_filename(save_prefix)
        before_train = make_xy_snapshot(Xtr0, ytr0, label="train_before_preprocessing")
        before_test = make_xy_snapshot(Xte0, yte0, label="test_before_preprocessing")
        after_train = make_xy_snapshot(Xtr_after_feature, ytr, ytr_transformed, label="train_after_preprocessing_unencoded")
        after_test = make_xy_snapshot(Xte_after_feature, yte, yte_transformed, label="test_after_preprocessing_unencoded")

        before_train.to_csv(DATA_SNAPSHOT_DIR / f"{prefix}_train_before_preprocessing.csv", index=False)
        before_test.to_csv(DATA_SNAPSHOT_DIR / f"{prefix}_test_before_preprocessing.csv", index=False)
        after_train.to_csv(DATA_SNAPSHOT_DIR / f"{prefix}_train_after_preprocessing_unencoded.csv", index=False)
        after_test.to_csv(DATA_SNAPSHOT_DIR / f"{prefix}_test_after_preprocessing_unencoded.csv", index=False)

        feature_names_df = pd.DataFrame({
            "Encoded_Feature_Index": range(len(encoded_feature_names)),
            "Encoded_Feature_Name": encoded_feature_names,
        })
        feature_names_df.to_csv(PREPROCESSED_MATRIX_DIR / f"{prefix}_encoded_feature_names.csv", index=False)

        if export_full_matrices:
            train_encoded_out = objects["X_train_encoded"].copy()
            train_encoded_out[TARGET_COL] = ytr
            train_encoded_out[f"{TARGET_COL}_Transformed"] = ytr_transformed
            train_encoded_out["Original_Index"] = Xtr_after_feature.index

            test_encoded_out = objects["X_test_encoded"].copy()
            test_encoded_out[TARGET_COL] = yte
            test_encoded_out[f"{TARGET_COL}_Transformed"] = yte_transformed
            test_encoded_out["Original_Index"] = Xte_after_feature.index

            train_encoded_out.to_csv(PREPROCESSED_MATRIX_DIR / f"{prefix}_train_after_preprocessing_encoded.csv", index=False)
            test_encoded_out.to_csv(PREPROCESSED_MATRIX_DIR / f"{prefix}_test_after_preprocessing_encoded.csv", index=False)

        if export_preview:
            after_train.head(preview_rows).to_csv(DATA_SNAPSHOT_DIR / f"{prefix}_preview_train_after_preprocessing_unencoded.csv", index=False)
            objects["X_train_encoded"].head(preview_rows).to_csv(PREPROCESSED_MATRIX_DIR / f"{prefix}_preview_train_after_preprocessing_encoded.csv", index=False)

    return meta, objects


BASELINE_REVIEWER_CONFIG = {
    "imputer_mode": BASE_IMPUTER_MODE,
    "scaler_type": "none",
    "target_transform": "none",
    "target_outlier_mode": "none",
    "feature_outlier_mode": "none",
}

baseline_preprocessing_meta, baseline_preprocessing_objects = export_preprocessing_snapshot(
    BASELINE_REVIEWER_CONFIG,
    stage="01_baseline_default",
    scenario_name="baseline_drop_missing_no_scaling_no_outlier",
    model_name="not_model_specific",
    save_prefix="baseline_default_complete_case",
    export_full_matrices=True,
)

baseline_preprocessing_audit_df = pd.DataFrame([baseline_preprocessing_meta])
display(baseline_preprocessing_audit_df)
save_table(baseline_preprocessing_audit_df, "06c_baseline_preprocessing_data_flow_audit.csv")

print("Baseline snapshots saved:")
print("- baseline data before preprocessing: data_snapshots/baseline_default_complete_case_*_before_preprocessing.csv")
print("- unencoded baseline data after preprocessing: data_snapshots/baseline_default_complete_case_*_after_preprocessing_unencoded.csv")
print("- encoded baseline data after preprocessing: preprocessed_matrices/baseline_default_complete_case_*_after_preprocessing_encoded.csv")


# %% [cell 22]
# ============================================================
# CELL 8 - DEFAULT PARAMETERS AND HYPERPARAMETER-TUNING SPACE
# ============================================================
# The baseline uses default hyperparameters from get_base_model().
# Hyperparameter tuning is performed only on training data using K-Fold CV.
# The holdout test set is not used for parameter selection to avoid data leakage.

MODEL_NAMES = get_model_names()

# ============================================================
# BASELINE DEFAULT PARAMETER TABLE
# ============================================================
# A pure baseline means:
# - the model uses default parameters from get_base_model()
# - no hyperparameter tuning has been applied
# - baseline preprocessing follows the baseline configuration, namely dropping rows with missing values
#   and no scaling/outlier handling/target transformation.

baseline_default_param_rows = []

for model_name in MODEL_NAMES:
    baseline_default_param_rows.append({
        "Model": model_name,
        "Baseline_Default_Params": safe_json(get_report_default_params(model_name)),
        "Baseline_Tuned": False
    })

baseline_default_params_df = pd.DataFrame(baseline_default_param_rows)

print("BASELINE MODEL DEFAULT PARAMETERS")
display(baseline_default_params_df)

save_table(
    baseline_default_params_df,
    "07a_baseline_default_params_for_report.csv"
)

TUNING_PARAM_SPACES = {
    "MLR": {
        "model__fit_intercept": [True, False],
        "model__positive": [False, True]
    },

    "Ridge Regression": {
        "model__alpha": [0.01, 0.1, 1.0, 10.0, 100.0, 300.0, 500.0],
        "model__fit_intercept": [True, False]
    },

    "SVR": {
        "model__kernel": ["rbf", "linear", "poly"],
        "model__C": [0.1, 1, 3, 10, 30, 100],
        "model__epsilon": [0.01, 0.03, 0.05, 0.1, 0.2, 0.5],
        "model__gamma": ["scale", "auto"],
        "model__degree": [2, 3]
    },

    "XGBoost": {
        "model__n_estimators": [200, 400, 700],
        "model__max_depth": [2, 3, 4],
        "model__learning_rate": [0.01, 0.03, 0.05],
        "model__subsample": [0.7, 0.85],
        "model__colsample_bytree": [0.7, 0.85],
        "model__min_child_weight": [3, 5, 10],
        "model__gamma": [0.05, 0.1, 0.3],
        "model__reg_alpha": [0.01, 0.1, 1.0],
        "model__reg_lambda": [3.0, 5.0, 10.0]
    },

    "Extra Trees Regression": {
        "model__n_estimators": [300, 500, 800],
        "model__max_depth": [8, 12, 20],
        "model__min_samples_split": [5, 10, 20],
        "model__min_samples_leaf": [2, 4, 6, 8],
        "model__max_features": ["sqrt", "log2", 0.5, 0.7],
        "model__bootstrap": [True, False]
    },

    "Random Forest Regression": {
        "model__n_estimators": [300, 500, 800],
        "model__max_depth": [8, 12, 20],
        "model__min_samples_split": [5, 10, 20],
        "model__min_samples_leaf": [2, 4, 6, 8],
        "model__max_features": ["sqrt", "log2", 0.5, 0.7],
        "model__bootstrap": [True, False]
    }
}

# ============================================================
# SUMMARY OF DEFAULT PARAMETERS AND TUNING SPACE
# ============================================================

default_summary_rows = []

for m in MODEL_NAMES:
    base = get_base_model(m)
    space = TUNING_PARAM_SPACES.get(m, {})

    n_candidates = 1
    for values in space.values():
        n_candidates *= len(values)

    if len(space) == 0:
        n_candidates = 1

    if N_ITER_TUNING is not None and n_candidates > N_ITER_TUNING:
        n_candidates_used = N_ITER_TUNING
        tuning_mode = f"Random sampled grid ({N_ITER_TUNING} candidates)"
    else:
        n_candidates_used = n_candidates
        tuning_mode = "Full grid tuning"

    default_summary_rows.append({
        "Model": m,
        "Baseline_Default_Params_Report": safe_json(get_report_default_params(m)),
        "Baseline_Default_Params_Full": safe_json(base.get_params()),
        "Tuning_Param_Space": safe_json(space),
        "Total_Tuning_Candidates": n_candidates,
        "Tuning_Candidates_Used": n_candidates_used,
        "FiveFold_Fits_Used": n_candidates_used * CV_TUNING,
        "Tuning_Mode": tuning_mode,
        "Tuning_Status": "Not tuned" if len(space) == 0 else "Tuned"
    })

default_params_df = pd.DataFrame(default_summary_rows)

print("BASELINE DEFAULT PARAMETERS AND HYPERPARAMETER-TUNING SPACE")
display(default_params_df)

save_table(
    default_params_df,
    "07_model_default_and_tuning_space.csv"
)

# %% [markdown]
# ## Model-configuration and tuning-space metadata
#
# The following cell saves default parameters, tuning-candidate counts, and hyperparameter spaces in JSON format for audit outside the notebook.


# %% [cell 24]
# ============================================================
# CELL 8B - MODEL CONFIGURATION JSON FOR REVIEWER AUDIT
# ============================================================

model_configurations = []
for m in MODEL_NAMES:
    space = TUNING_PARAM_SPACES.get(m, {})
    n_candidates = 1
    for values in space.values():
        n_candidates *= len(values)
    if len(space) == 0:
        n_candidates = 1
    model_configurations.append({
        "Model": m,
        "Default_Params": get_report_default_params(m),
        "Tuning_Param_Space": space,
        "Total_Grid_Candidates": int(n_candidates),
        "Candidates_Used": int(min(n_candidates, N_ITER_TUNING)) if N_ITER_TUNING is not None else int(n_candidates),
        "Tuning_Mode": "sampled" if N_ITER_TUNING is not None and n_candidates > N_ITER_TUNING else "full_grid",
    })

save_json(model_configurations, "model_configurations.json")
print("Model configuration JSON saved.")


# %% [cell 25]
# ============================================================
# CELL 9 - DEFAULT BASELINE AND TARGET-TRANSFORMATION SCENARIOS
# ============================================================
# The default baseline is implemented as complete-case analysis:
# rows with missing feature values are removed.
#
# Therefore, the baseline does not use imputation.
# The log1p scenario below still uses the same missing-value handling, namely row dropping,
# but the target is transformed using log1p.

scenario_results = []
scenario_predictions = {}
scenario_fold_details = {}

BASELINE_CONFIG = {
    "imputer_mode": BASE_IMPUTER_MODE,   # default sekarang: "drop"
    "scaler_type": "none",
    "target_transform": "none",
    "target_outlier_mode": "none",
    "feature_outlier_mode": "none"
}

LOG1P_ONLY_CONFIG = {
    **BASELINE_CONFIG,
    "target_transform": "log1p"
}

for model_name in MODEL_NAMES:
    print("=" * 100)
    print("BASELINE AND TARGET TRANSFORMATION | Model:", model_name)
    print("=" * 100)

    row, pipe, pred, folds = evaluate_scenario_with_cv(
        model_name=model_name,
        scenario_name="baseline_drop_missing_no_scaling_no_outlier",
        stage="01_baseline_default",
        **BASELINE_CONFIG
    )
    scenario_results.append(row)
    scenario_predictions[(row["Stage"], model_name)] = pred
    scenario_fold_details[(row["Stage"], model_name)] = folds

    print(
        "Baseline drop missing row | Test R2 original:",
        row["Test_R2_Original"],
        "| Test RMSE:",
        row["Test_RMSE_Original"],
        "| Train rows final:",
        row.get("Train_Rows_Final"),
        "| Test rows final:",
        row.get("Test_Rows")
    )

    row, pipe, pred, folds = evaluate_scenario_with_cv(
        model_name=model_name,
        scenario_name="target_transform_log1p_with_drop_missing",
        stage="02_target_transform_log1p",
        **LOG1P_ONLY_CONFIG
    )
    scenario_results.append(row)
    scenario_predictions[(row["Stage"], model_name)] = pred
    scenario_fold_details[(row["Stage"], model_name)] = folds

    print(
        "Target transform log1p + drop missing row | Test R2 original:",
        row["Test_R2_Original"],
        "| Test RMSE:",
        row["Test_RMSE_Original"],
        "| Train rows final:",
        row.get("Train_Rows_Final"),
        "| Test rows final:",
        row.get("Test_Rows")
    )

baseline_df = pd.DataFrame(scenario_results)

# ============================================================
# ADD BASELINE DEFAULT PARAMETERS TO BASELINE RESULTS
# ============================================================

baseline_df["Baseline_Default_Params"] = baseline_df["Model"].apply(
    lambda m: safe_json(get_report_default_params(m))
)

baseline_df["Baseline_Tuned"] = False

baseline_display_df = baseline_df[[
    "Stage", "Model",
    "Baseline_Tuned",
    "Baseline_Default_Params",

    "CV_Mean_Train_R2_Original",
    "CV_Mean_R2_Original",
    "CV_Mean_Train_Val_Gap_R2_Original",

    "Train_R2_Original",
    "Test_R2_Original",
    "Train_Test_Gap_R2_Original",

    "Test_RMSE_Original",
    "Test_MAE_Original",

    "Imputer_Mode",
    "Scaler_Type",
    "Target_Transform",
    "Missing_Train_Dropped_Rows",
    "Missing_Test_Dropped_Rows",
    "Train_Rows_Original",
    "Train_Rows_Final",
    "Test_Rows"
]].copy()

baseline_display_df = baseline_display_df.rename(columns={
    "CV_Mean_Train_R2_Original": "CV_Train_R2_Original",
    "CV_Mean_R2_Original": "CV_Val_R2_Original",
    "CV_Mean_Train_Val_Gap_R2_Original": "CV_Train_Val_Gap_R2_Original",
    "Test_R2_Original": "Holdout_R2_Original",

    "CV_Mean_Train_R2_Transformed": "CV_Train_R2_Transformed",
    "CV_Mean_R2_Transformed": "CV_Val_R2_Transformed",
    "R2_Train_Transformed": "Train_R2_Transformed",
    "R2_Test_Transformed": "Holdout_R2_Transformed",

    "Test_RMSE_Original": "Holdout_RMSE_Original",
    "Test_MAE_Original": "Holdout_MAE_Original",
})

display(baseline_display_df.sort_values(
    ["Stage", "Holdout_R2_Original"],
    ascending=[True, False]
))

save_table(baseline_df, "08_baseline_default_and_target_transform_results.csv")
save_table(
    baseline_display_df,
    "08_baseline_results_with_default_params.csv"
)


# %% [cell 26]
# ============================================================
# CELL 10 - PURE SINGLE-PREPROCESSING SCENARIOS
# ============================================================
# Tujuan:
# - Measure the effect of each preprocessing component separately.
# - Each scenario changes only ONE component from the baseline.
# - log1p target transformation is used only in the target-transformation scenario.
# - CV_R2 and Holdout_R2 columns are made single/simplified:
#   * Jika target_transform = none  -> pakai skor original.
#   * Jika target_transform = log1p -> pakai skor transformed.
# - Thus, log1p is evaluated on its own transformed scale and is not mixed
#   with the original scale used in the main summary table.

separate_results = []

IMPUTER_SCENARIOS = [
    "drop",
    "mean",
    "median",
    "most_frequent",
    "constant_zero",
    "iterative"
]

SCALER_SCENARIOS = [
    "none",
    "standard",
    "minmax",
    "robust",
    "maxabs",
    "quantile_normal",
    "power_yeo"
]

TARGET_OUTLIER_SCENARIOS = [
    "none",
    "remove_iqr",
    "cap_iqr",
    "median_iqr",
    "winsor_p01_p99"
]

FEATURE_OUTLIER_SCENARIOS = [
    "none",
    "feature_cap_iqr",
    "feature_winsor_p01_p99"
]

TARGET_TRANSFORM_SCENARIOS = [
    "none",
    "log1p"
]


def add_single_score_columns(row):
    """
    Create a single score column so pure preprocessing tables are easier to read.

    For target_transform = log1p scenarios:
        CV_R2 and Holdout_R2 use transformed-scale scores.

    For other scenarios:
        CV_R2 and Holdout_R2 use original-scale scores.
        For scenarios without target transformation, transformed and original scores are identical.
    """

    target_transform = row.get("Target_Transform", "none")

    if target_transform == "log1p":
        row["CV_Train_R2"] = row["CV_Mean_Train_R2_Transformed"]
        row["CV_R2"] = row["CV_Mean_R2_Transformed"]
        row["Train_R2"] = row["R2_Train_Transformed"]
        row["Holdout_R2"] = row["R2_Test_Transformed"]
        row["Train_Holdout_Gap_R2"] = row["Train_Test_Gap_R2_Transformed"]
        row["Evaluation_Scale"] = "transformed_log1p"
    else:
        row["CV_Train_R2"] = row["CV_Mean_Train_R2_Original"]
        row["CV_R2"] = row["CV_Mean_R2_Original"]
        row["Train_R2"] = row["Train_R2_Original"]
        row["Holdout_R2"] = row["Test_R2_Original"]
        row["Train_Holdout_Gap_R2"] = row["Train_Test_Gap_R2_Original"]
        row["Evaluation_Scale"] = "original"

    return row


for model_name in MODEL_NAMES:
    print("\n" + "=" * 100)
    print("PURE SINGLE PREPROCESSING | Model:", model_name)
    print("=" * 100)

    # ========================================================
    # 1. Single missing-value handling
    # Changed component: imputer_mode
    # Other components remain at baseline settings:
    # - scaler none
    # - outlier target none
    # - feature outlier: none
    # - target transform none
    # ========================================================
    for imp in IMPUTER_SCENARIOS:
        row, pipe, pred, folds = evaluate_scenario_with_cv(
            model_name=model_name,
            scenario_name=f"single_missing_value_{imp}",
            stage="03_single_missing_value",
            imputer_mode=imp,
            scaler_type="none",
            target_transform="none",
            target_outlier_mode="none",
            feature_outlier_mode="none"
        )
        row = add_single_score_columns(row)
        separate_results.append(row)

    # ========================================================
    # 2. Single scaling scenario
    # Changed component: scaler_type
    # ========================================================
    for scaler in SCALER_SCENARIOS:
        row, pipe, pred, folds = evaluate_scenario_with_cv(
            model_name=model_name,
            scenario_name=f"single_scaler_{scaler}",
            stage="04_single_scaler",
            imputer_mode="drop",
            scaler_type=scaler,
            target_transform="none",
            target_outlier_mode="none",
            feature_outlier_mode="none"
        )
        row = add_single_score_columns(row)
        separate_results.append(row)

    # ========================================================
    # 3. Single target-outlier scenario
    # Changed component: target_outlier_mode
    # ========================================================
    for out_mode in TARGET_OUTLIER_SCENARIOS:
        row, pipe, pred, folds = evaluate_scenario_with_cv(
            model_name=model_name,
            scenario_name=f"single_target_outlier_{out_mode}",
            stage="05_single_target_outlier",
            imputer_mode="drop",
            scaler_type="none",
            target_transform="none",
            target_outlier_mode=out_mode,
            feature_outlier_mode="none"
        )
        row = add_single_score_columns(row)
        separate_results.append(row)

    # ========================================================
    # 4. Single feature-outlier handling
    # Changed component: feature_outlier_mode
    # ========================================================
    for f_out_mode in FEATURE_OUTLIER_SCENARIOS:
        row, pipe, pred, folds = evaluate_scenario_with_cv(
            model_name=model_name,
            scenario_name=f"single_feature_outlier_{f_out_mode}",
            stage="06_single_feature_outlier",
            imputer_mode="drop",
            scaler_type="none",
            target_transform="none",
            target_outlier_mode="none",
            feature_outlier_mode=f_out_mode
        )
        row = add_single_score_columns(row)
        separate_results.append(row)

    # ========================================================
    # 5. Single target-transformation scenario
    # Changed component: target_transform
    # ========================================================
    for t_mode in TARGET_TRANSFORM_SCENARIOS:
        row, pipe, pred, folds = evaluate_scenario_with_cv(
            model_name=model_name,
            scenario_name=f"single_target_transform_{t_mode}",
            stage="07_single_target_transform",
            imputer_mode="drop",
            scaler_type="none",
            target_transform=t_mode,
            target_outlier_mode="none",
            feature_outlier_mode="none"
        )
        row = add_single_score_columns(row)
        separate_results.append(row)


separate_preprocessing_df = pd.DataFrame(separate_results)

display_columns = [
    "Stage",
    "Scenario_Name",
    "Model",
    "CV_R2",
    "Holdout_R2",
    "Evaluation_Scale",
    "Test_RMSE_Original",
    "Test_MAE_Original",
    "Train_Rows_Final",
    "Test_Rows",
    "Imputer_Mode",
    "Scaler_Type",
    "Target_Transform",
    "Target_Outlier_Mode",
    "Feature_Outlier_Mode"
]

display(
    separate_preprocessing_df[display_columns]
    .sort_values(
        ["Stage", "Model", "CV_R2"],
        ascending=[True, True, False]
    )
)

save_table(
    separate_preprocessing_df,
    "09_pure_single_preprocessing_results.csv"
)


# ============================================================
# Best single-preprocessing method by component and model
# Ranking: CV_R2
# ============================================================

best_separate_by_stage_model = (
    separate_preprocessing_df
    .sort_values("CV_R2", ascending=False)
    .groupby(["Stage", "Model"], as_index=False)
    .head(1)
    .reset_index(drop=True)
)

display(
    best_separate_by_stage_model[display_columns]
    .sort_values(["Stage", "Model"])
)

save_table(
    best_separate_by_stage_model,
    "10_best_pure_single_preprocessing_by_stage_model.csv"
)


# ============================================================
# Summary table for the manuscript:
# one best row for each preprocessing component
# Ranking: CV_R2
# ============================================================

def get_component_name(stage):
    mapping = {
        "03_single_missing_value": "Missing value",
        "04_single_scaler": "Scaling",
        "05_single_target_outlier": "Outlier target",
        "06_single_feature_outlier": "Feature outlier",
        "07_single_target_transform": "Target transformation"
    }
    return mapping.get(stage, stage)


def get_selected_method(row):
    stage = row["Stage"]

    if stage == "03_single_missing_value":
        return row["Imputer_Mode"]

    if stage == "04_single_scaler":
        return row["Scaler_Type"]

    if stage == "05_single_target_outlier":
        return row["Target_Outlier_Mode"]

    if stage == "06_single_feature_outlier":
        return row["Feature_Outlier_Mode"]

    if stage == "07_single_target_transform":
        return row["Target_Transform"]

    return row["Scenario_Name"]


best_separate_by_component = (
    separate_preprocessing_df
    .sort_values("CV_R2", ascending=False)
    .groupby("Stage", as_index=False)
    .head(1)
    .reset_index(drop=True)
)

best_separate_by_component["Preprocessing_Component"] = (
    best_separate_by_component["Stage"].apply(get_component_name)
)

best_separate_by_component["Best_Method"] = (
    best_separate_by_component.apply(get_selected_method, axis=1)
)

component_order = {
    "Missing value": 1,
    "Scaling": 2,
    "Outlier target": 3,
    "Feature outlier": 4,
    "Target transformation": 5
}

best_separate_by_component["Urutan"] = (
    best_separate_by_component["Preprocessing_Component"].map(component_order)
)

paper_single_component_df = (
    best_separate_by_component
    .sort_values("Urutan")
    [[
        "Preprocessing_Component",
        "Best_Method",
        "Model",
        "CV_Train_R2",
        "CV_R2",
        "Train_R2",
        "Holdout_R2",
        "Train_Holdout_Gap_R2",
        "Evaluation_Scale",
        "Test_RMSE_Original",
        "Test_MAE_Original",
        "Train_Rows_Final",
        "Test_Rows",
        "Imputer_Mode",
        "Scaler_Type",
        "Target_Transform",
        "Target_Outlier_Mode",
        "Feature_Outlier_Mode"
    ]]
    .rename(columns={
        "CV_Mean_Train_R2_Original": "CV_Train_R2_Original",
        "CV_Mean_R2_Original": "CV_Val_R2_Original",
        "CV_Mean_Train_Val_Gap_R2_Original": "CV_Train_Val_Gap_R2_Original",
        "Test_R2_Original": "Holdout_R2_Original",

        "CV_Mean_Train_R2_Transformed": "CV_Train_R2_Transformed",
        "CV_Mean_R2_Transformed": "CV_Val_R2_Transformed",
        "R2_Train_Transformed": "Train_R2_Transformed",
        "R2_Test_Transformed": "Holdout_R2_Transformed",

        "Test_RMSE_Original": "RMSE_Original",
        "Test_MAE_Original": "MAE_Original"
    })
)

display(paper_single_component_df)

save_table(
    paper_single_component_df,
    "11_paper_table_pure_single_preprocessing_summary.csv"
)


# ============================================================
# Detail khusus transformasi target
# Keep the none vs log1p comparison transparent
# ============================================================

target_transform_detail_df = (
    separate_preprocessing_df[
        separate_preprocessing_df["Stage"] == "07_single_target_transform"
    ]
    .sort_values(
        ["Model", "CV_R2"],
        ascending=[True, False]
    )
    [[
        "Stage",
        "Scenario_Name",
        "Model",
        "Target_Transform",
        "CV_Train_R2",
        "CV_R2",
        "Train_R2",
        "Holdout_R2",
        "Train_Holdout_Gap_R2",
        "Evaluation_Scale",
        "Test_RMSE_Original",
        "Test_MAE_Original",
        "Train_Rows_Final",
        "Test_Rows"
    ]]
)

display(target_transform_detail_df)

save_table(
    target_transform_detail_df,
    "12_detail_pure_single_target_transform_none_vs_log1p.csv"
)


print("\nCatatan:")
print("- This table shows pure single-preprocessing results.")
print("- Each row changes only one component from the baseline.")
print("- log1p target transformation is used only in the target-transformation component.")
print("- CV_R2 and Holdout_R2 are the main scores according to the scenario evaluation scale.")
print("- If Evaluation_Scale = original, the score is computed on the original nanometer scale.")
print("- If Evaluation_Scale = transformed_log1p, the score is computed on the log1p target scale.")
print("- RMSE_Original and MAE_Original are still displayed as original-scale error information.")


# %% [cell 27]
# ============================================================
# CELL 11 - COMBINED PREPROCESSING SEARCH FOR THE core_size DATASET
# ============================================================
# Combined preprocessing dibuat FULL GRID.
# The grid keeps drop as the complete-case comparator,
# but it also tests multiple imputation methods for comparison
# with a data-preserving strategy.
#
# Best preprocessing is selected based on training CV.
# The holdout set remains only for evaluation, not configuration selection.

combined_param_grid = {
    "imputer_mode": [
        "drop",
        "mean",
        "median",
        "most_frequent",
        "constant_zero",
        "iterative"
    ],
    "scaler_type": [
        "none",
        "standard",
        "minmax",
        "robust",
        "maxabs",
        "quantile_normal",
        "power_yeo"
    ],
    "target_outlier_mode": [
        "none",
        "remove_iqr",
        "cap_iqr",
        "median_iqr",
        "winsor_p01_p99"
    ],
    "feature_outlier_mode": [
        "none",
        "feature_cap_iqr",
        "feature_winsor_p01_p99"
    ],
    "target_transform": [
        "none",
        "log1p"
    ]
}

from itertools import product

keys = list(combined_param_grid.keys())

if FULL_COMBINED_GRID:
    combined_candidates = [
        dict(zip(keys, vals))
        for vals in product(*[combined_param_grid[k] for k in keys])
    ]
    combined_mode = "FULL GRID"
else:
    combined_all_candidates = [
        dict(zip(keys, vals))
        for vals in product(*[combined_param_grid[k] for k in keys])
    ]

    if N_ITER_COMBINED_PREPROCESSING is None:
        combined_candidates = combined_all_candidates
        combined_mode = "FULL GRID because N_ITER_COMBINED_PREPROCESSING=None"
    else:
        combined_candidates = list(ParameterSampler(
            combined_param_grid,
            n_iter=N_ITER_COMBINED_PREPROCESSING,
            random_state=RANDOM_STATE
        ))
        combined_mode = f"SAMPLED GRID n={N_ITER_COMBINED_PREPROCESSING}"

print("Mode combined preprocessing:", combined_mode)
print("Number of combined-preprocessing combinations tested per model:", len(combined_candidates))
print("Total combined-preprocessing evaluations for all models:", len(combined_candidates) * len(MODEL_NAMES))
print("Preprocessing selection metric:", PRIMARY_SELECTION_METRIC)

display(pd.DataFrame(combined_candidates).head(10))

combined_results = []

for model_name in MODEL_NAMES:
    print("\n" + "=" * 100)
    print("COMBINED PREPROCESSING SEARCH | Model:", model_name)
    print("=" * 100)

    for i, cfg in enumerate(combined_candidates, start=1):
        scenario_name = (
            f"combined_{i:04d}_imp-{cfg['imputer_mode']}_scaler-{cfg['scaler_type']}"
            f"_tOut-{cfg['target_outlier_mode']}_fOut-{cfg['feature_outlier_mode']}_target-{cfg['target_transform']}"
        )

        try:
            row, pipe, pred, folds = evaluate_scenario_with_cv(
                model_name=model_name,
                scenario_name=scenario_name,
                stage="07_combined_preprocessing",
                imputer_mode=cfg["imputer_mode"],
                scaler_type=cfg["scaler_type"],
                target_transform=cfg["target_transform"],
                target_outlier_mode=cfg["target_outlier_mode"],
                feature_outlier_mode=cfg["feature_outlier_mode"]
            )
            row["Error"] = ""
        except Exception as e:
            row = {
                "Stage": "07_combined_preprocessing",
                "Scenario_Name": scenario_name,
                "Model": model_name,
                "Imputer_Mode": cfg["imputer_mode"],
                "Scaler_Type": cfg["scaler_type"],
                "Target_Outlier_Mode": cfg["target_outlier_mode"],
                "Feature_Outlier_Mode": cfg["feature_outlier_mode"],
                "Target_Transform": cfg["target_transform"],
                "CV_Mean_R2_Transformed": -np.inf,
                "CV_Mean_R2_Original": -np.inf,
                "R2_Test_Transformed": -np.inf,
                "Test_R2_Original": -np.inf,
                "Test_RMSE_Original": np.inf,
                "Test_MAE_Original": np.inf,
                "Error": str(e)
            }

        combined_results.append(row)

        if i % 25 == 0 or i == len(combined_candidates):
            print(f"  completed {i}/{len(combined_candidates)} combinations")

combined_preprocessing_df = pd.DataFrame(combined_results)

display(combined_preprocessing_df[[
    "Model", "Scenario_Name",

    # CV training and validation.
    "CV_Mean_Train_R2_Original",
    "CV_Mean_R2_Original",
    "CV_Mean_Train_Val_Gap_R2_Original",

    # Train-test split.
    "Train_R2_Original",
    "Test_R2_Original",
    "Train_Test_Gap_R2_Original",

    # Transformed score.
    "CV_Mean_Train_R2_Transformed",
    "CV_Mean_R2_Transformed",
    "R2_Train_Transformed",
    "R2_Test_Transformed",
    "Train_Test_Gap_R2_Transformed",

    "Test_RMSE_Original", "Test_MAE_Original",
    "Imputer_Mode", "Scaler_Type",
    "Target_Outlier_Mode", "Feature_Outlier_Mode",
    "Target_Transform", "Error"
]].sort_values(["Model", PRIMARY_SELECTION_METRIC], ascending=[True, False]).head(30))

save_table(combined_preprocessing_df, "11_combined_preprocessing_all_results.csv")

valid_combined_df = combined_preprocessing_df[
    np.isfinite(combined_preprocessing_df[PRIMARY_SELECTION_METRIC])
].copy()

# Add a stability score so the best preprocessing is not selected only
# based on the highest CV R2 while also considering the train-validation gap.
valid_combined_df = add_overfitting_control_scores(valid_combined_df)
combined_preprocessing_df = add_overfitting_control_scores(combined_preprocessing_df)

best_preprocessing_per_model = (
    valid_combined_df
    .sort_values(PRIMARY_SELECTION_METRIC, ascending=False)
    .groupby("Model", as_index=False)
    .head(1)
    .reset_index(drop=True)
)

print("\nBEST COMBINED PREPROCESSING PER MODEL - selected based on training CV")
print("Selection metric:", PRIMARY_SELECTION_METRIC)

display(best_preprocessing_per_model[[
    "Model",

    "CV_Stability_Score_Original",
    "CV_Overfit_Flag",
    "Holdout_Overfit_Flag",

    "CV_Mean_Train_R2_Original",
    "CV_Mean_R2_Original",
    "CV_Mean_Train_Val_Gap_R2_Original",

    "Train_R2_Original",
    "Test_R2_Original",
    "Train_Test_Gap_R2_Original",

    "CV_Mean_Train_R2_Transformed",
    "CV_Mean_R2_Transformed",
    "R2_Train_Transformed",
    "R2_Test_Transformed",
    "Train_Test_Gap_R2_Transformed",

    "Test_RMSE_Original", "Test_MAE_Original",
    "Imputer_Mode", "Scaler_Type",
    "Target_Outlier_Mode", "Feature_Outlier_Mode",
    "Target_Transform", "Scenario_Name"
]])

save_table(best_preprocessing_per_model, "12_best_combined_preprocessing_per_model.csv")


# %% [cell 28]
# ============================================================
# BUKTI DINAMIS:
# GLOBAL BEST SINGLE-PREPROCESSING METHOD COMBINATION
# NOT ALWAYS THE SAME AS THE BEST COMBINED PREPROCESSING
# ============================================================
# Maksud cell ini:
# 1. Retrieve the best method from single preprocessing for each component
#    GLOBALLY across all models.
#
#    Example:
#    - Best missing-value method from all missing-value scenarios and all models
#    - Best scaling method from all scaling scenarios and all models
#    - Best target-outlier method from all target-outlier scenarios and all models
#    - Best feature-outlier method from all feature-outlier scenarios and all models
#    - Best target-transformation method from all target-transformation scenarios and all models
#
# 2. Combine those global best methods into one preprocessing configuration.
#
# 3. Search/evaluate that combined configuration in combined_preprocessing_df
#    for each model.
#
# 4. Compare the result with Best Combined Search.
#
# Important notes:
# This differs from the previous version that selected the best method per component
# for each model. This version follows the preprocessing-summary table
# global single-preprocessing summary, such as:
# Missing value      -> constant_zero
# Scaling            -> none
# Outlier target     -> winsor_p01_p99
# Outlier features      -> feature_cap_iqr
# Target transformation -> none

required_vars = [
    "combined_preprocessing_df",
    "best_preprocessing_per_model",
    "separate_preprocessing_df"
]

missing_vars = [v for v in required_vars if v not in globals()]

if missing_vars:
    raise NameError(
        "The following variables are not yet available: "
        + ", ".join(missing_vars)
        + ". Run Cell 10 and Cell 11 first."
    )

SELECTION_METRIC = globals().get("PRIMARY_SELECTION_METRIC", "CV_Mean_R2_Original")

# ============================================================
# Mapping of single-preprocessing components
# ============================================================

stage_info = {
    "03_single_missing_value": {
        "Preprocessing_Component": "Missing value",
        "Parameter_Config": "Imputer_Mode"
    },
    "04_single_scaler": {
        "Preprocessing_Component": "Scaling",
        "Parameter_Config": "Scaler_Type"
    },
    "05_single_target_outlier": {
        "Preprocessing_Component": "Outlier target",
        "Parameter_Config": "Target_Outlier_Mode"
    },
    "06_single_feature_outlier": {
        "Preprocessing_Component": "Feature outlier",
        "Parameter_Config": "Feature_Outlier_Mode"
    },
    "07_single_target_transform": {
        "Preprocessing_Component": "Target transformation",
        "Parameter_Config": "Target_Transform"
    }
}

stage_order = [
    "03_single_missing_value",
    "04_single_scaler",
    "05_single_target_outlier",
    "06_single_feature_outlier",
    "07_single_target_transform"
]

# ============================================================
# 1. Retrieve the GLOBAL best single-preprocessing method
#    from all models for each component
# ============================================================

global_best_component_rows = []

for stage in stage_order:
    temp = separate_preprocessing_df[
        separate_preprocessing_df["Stage"] == stage
    ].copy()

    if temp.empty:
        raise ValueError(f"No results are available for stage: {stage}")

    # Single-preprocessing ranking uses CV_R2,
    # according to the pure single-preprocessing summary table.
    best_row = temp.sort_values("CV_R2", ascending=False).iloc[0].copy()

    param_col = stage_info[stage]["Parameter_Config"]
    component_name = stage_info[stage]["Preprocessing_Component"]
    selected_method = best_row[param_col]

    global_best_component_rows.append({
        "Stage": stage,
        "Preprocessing_Component": component_name,
        "Parameter_Config": param_col,
        "Best_Method": selected_method,
        "Best_Model_In_Single_Preprocessing": best_row["Model"],
        "CV_R2_Single": best_row["CV_R2"],
        "Holdout_R2_Single": best_row["Holdout_R2"],
        "Single_Evaluation_Scale": best_row["Evaluation_Scale"],
        "RMSE_Original_Single": best_row["Test_RMSE_Original"],
        "MAE_Original_Single": best_row["Test_MAE_Original"],
        "Imputer_Mode_Single": best_row["Imputer_Mode"],
        "Scaler_Type_Single": best_row["Scaler_Type"],
        "Target_Transform_Single": best_row["Target_Transform"],
        "Target_Outlier_Mode_Single": best_row["Target_Outlier_Mode"],
        "Feature_Outlier_Mode_Single": best_row["Feature_Outlier_Mode"]
    })

global_best_component_df = pd.DataFrame(global_best_component_rows)

print("=" * 120)
print("GLOBAL BEST METHODS FROM SINGLE PREPROCESSING")
print("=" * 120)
display(global_best_component_df[[
    "Preprocessing_Component",
    "Best_Method",
    "Best_Model_In_Single_Preprocessing",
    "CV_R2_Single",
    "Holdout_R2_Single",
    "Single_Evaluation_Scale"
]])

save_table(
    global_best_component_df,
    "13a_global_best_single_preprocessing_components.csv"
)

# ============================================================
# 2. Build one combined configuration from the global best methods
# ============================================================

global_single_best_config = {}

for _, row in global_best_component_df.iterrows():
    global_single_best_config[row["Parameter_Config"]] = row["Best_Method"]

required_config_keys = [
    "Imputer_Mode",
    "Scaler_Type",
    "Target_Outlier_Mode",
    "Feature_Outlier_Mode",
    "Target_Transform"
]

missing_config_keys = [
    k for k in required_config_keys
    if k not in global_single_best_config
]

if missing_config_keys:
    raise ValueError(
        "The combined configuration is incomplete. Missing keys: "
        + ", ".join(missing_config_keys)
    )

single_best_config_df = pd.DataFrame([{
    "Config_Type": "Global best single-preprocessing method combination",
    "Imputer_Mode": global_single_best_config["Imputer_Mode"],
    "Scaler_Type": global_single_best_config["Scaler_Type"],
    "Target_Outlier_Mode": global_single_best_config["Target_Outlier_Mode"],
    "Feature_Outlier_Mode": global_single_best_config["Feature_Outlier_Mode"],
    "Target_Transform": global_single_best_config["Target_Transform"]
}])

print("\n" + "=" * 120)
print("COMBINED CONFIGURATION FROM GLOBAL BEST SINGLE-PREPROCESSING METHODS")
print("=" * 120)
display(single_best_config_df)

save_table(
    single_best_config_df,
    "13b_global_single_best_combined_config.csv"
)

# For backward compatibility with older cells that may call this object.
global_single_best_config_df = single_best_config_df.copy()

# ============================================================
# 3. Search for the global combined configuration in combined_preprocessing_df
#    for each model
# ============================================================

manual_single_best_rows = []

for model_name in MODEL_NAMES:
    temp = combined_preprocessing_df[
        (combined_preprocessing_df["Model"] == model_name) &
        (combined_preprocessing_df["Imputer_Mode"] == global_single_best_config["Imputer_Mode"]) &
        (combined_preprocessing_df["Scaler_Type"] == global_single_best_config["Scaler_Type"]) &
        (combined_preprocessing_df["Target_Outlier_Mode"] == global_single_best_config["Target_Outlier_Mode"]) &
        (combined_preprocessing_df["Feature_Outlier_Mode"] == global_single_best_config["Feature_Outlier_Mode"]) &
        (combined_preprocessing_df["Target_Transform"] == global_single_best_config["Target_Transform"])
    ].copy()

    if temp.empty:
        manual_single_best_rows.append({
            "Comparison_Type": "Global best single-method combination",
            "Model": model_name,
            "CV_Mean_R2_Transformed": np.nan,
            "CV_Mean_R2_Original": np.nan,
            "R2_Test_Transformed": np.nan,
            "Test_R2_Original": np.nan,
            "Test_RMSE_Original": np.nan,
            "Test_MAE_Original": np.nan,
            "Imputer_Mode": global_single_best_config["Imputer_Mode"],
            "Scaler_Type": global_single_best_config["Scaler_Type"],
            "Target_Transform": global_single_best_config["Target_Transform"],
            "Target_Outlier_Mode": global_single_best_config["Target_Outlier_Mode"],
            "Feature_Outlier_Mode": global_single_best_config["Feature_Outlier_Mode"],
            "Scenario_Name": "Configuration not found in combined_preprocessing_df"
        })
    else:
        # There should be only one row because the full grid tests each combination once per model.
        row = temp.sort_values(SELECTION_METRIC, ascending=False).iloc[0].to_dict()
        row["Comparison_Type"] = "Global best single-method combination"
        manual_single_best_rows.append(row)

manual_single_best_combined_df = pd.DataFrame(manual_single_best_rows)

print("\n" + "=" * 120)
print("EVALUATION RESULTS FOR THE GLOBAL SINGLE-BEST METHOD COMBINATION")
print("=" * 120)
display(manual_single_best_combined_df[[
    "Comparison_Type",
    "Model",
    "CV_Mean_R2_Transformed",
    "CV_Mean_R2_Original",
    "R2_Test_Transformed",
    "Test_R2_Original",
    "Test_RMSE_Original",
    "Test_MAE_Original",
    "Imputer_Mode",
    "Scaler_Type",
    "Target_Transform",
    "Target_Outlier_Mode",
    "Feature_Outlier_Mode",
    "Scenario_Name"
]])

save_table(
    manual_single_best_combined_df,
    "13c_global_single_best_combined_results.csv"
)

# ============================================================
# 4. Compare against the best combined search for each model
# ============================================================

best_combined_compare_df = best_preprocessing_per_model.copy()
best_combined_compare_df["Comparison_Type"] = "Best combined search"

proof_compare_df = pd.concat(
    [manual_single_best_combined_df, best_combined_compare_df],
    ignore_index=True,
    sort=False
)

proof_cols = [
    "Comparison_Type",
    "Model",
    "CV_Mean_R2_Transformed",
    "CV_Mean_R2_Original",
    "R2_Test_Transformed",
    "Test_R2_Original",
    "Test_RMSE_Original",
    "Test_MAE_Original",
    "Imputer_Mode",
    "Scaler_Type",
    "Target_Transform",
    "Target_Outlier_Mode",
    "Feature_Outlier_Mode",
    "Scenario_Name"
]

proof_compare_df = proof_compare_df[proof_cols].sort_values(
    ["Model", "Comparison_Type"]
).reset_index(drop=True)

print("\n" + "=" * 120)
print("COMPARISON: GLOBAL SINGLE-BEST METHOD COMBINATION vs BEST COMBINED SEARCH")
print("=" * 120)
display(proof_compare_df)

save_table(
    proof_compare_df,
    "13d_proof_global_single_best_combined_vs_best_search.csv"
)

# Also save using the legacy filename to avoid confusion with previous outputs.
save_table(
    single_best_config_df,
    "13a_single_best_config_dynamic.csv"
)

save_table(
    proof_compare_df,
    "13b_proof_single_best_combined_vs_best_search.csv"
)

print("\nCatatan:")
print("- The global single-best method combination is derived from the best method of each single-preprocessing component.")
print("- Single-best method selection is performed globally across all models based on CV_R2 in Cell 10.")
print("- That global configuration is then compared with the best combined search.")
print("- If the global single-best result is lower, the best method for each individual component is not necessarily the best combined preprocessing strategy.")

# %% [markdown]
# ## Target-transform comparison in combined preprocessing
#
# The following cell compares the best preprocessing combination for `target_transform=none` and `target_transform=log1p` for each model. This comparison makes the effect of target transformation explicit on both original and transformed scales.


# %% [cell 30]
# ============================================================
# CELL 11B - PERBANDINGAN TARGET TRANSFORM NONE vs LOG1P
# ============================================================
# This table retrieves the best combination per model for each target_transform setting.
# Selection is performed based on PRIMARY_SELECTION_METRIC for consistency
# with the notebook main selection process.

best_combined_by_transform = (
    combined_preprocessing_df
    .sort_values(PRIMARY_SELECTION_METRIC, ascending=False)
    .groupby(["Model", "Target_Transform"], as_index=False)
    .head(1)
    .reset_index(drop=True)
)

transform_comparison_cols = [
    "Model", "Target_Transform", "CV_Mean_R2_Original", "Test_R2_Original",
    "CV_Mean_R2_Transformed", "R2_Test_Transformed",
    "Test_RMSE_Original", "Test_MAE_Original",
    "Imputer_Mode", "Scaler_Type", "Target_Outlier_Mode", "Feature_Outlier_Mode",
    "Scenario_Name"
]

print("BEST COMBINED PREPROCESSING PER MODEL AND TARGET TRANSFORMATION")
display(best_combined_by_transform[transform_comparison_cols].sort_values(["Model", "Target_Transform"]))
save_table(best_combined_by_transform[transform_comparison_cols], "12b_best_combined_by_target_transform.csv")

def plot_transform_comparison(df, metric_cv, metric_holdout, title, filename):
    plot_df = df.sort_values(["Model", "Target_Transform"]).reset_index(drop=True)
    labels = plot_df["Model"] + "\n" + plot_df["Target_Transform"]
    x = np.arange(len(plot_df))
    width = 0.38

    plt.figure(figsize=(max(10, len(plot_df) * 0.75), 5.5))
    plt.bar(x - width/2, plot_df[metric_cv], width, label="CV training")
    plt.bar(x + width/2, plot_df[metric_holdout], width, label="Holdout test")
    plt.xticks(x, labels, rotation=65, ha="right")
    plt.ylabel("R²")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    path = PLOT_DIR / filename
    plt.savefig(path, dpi=180)
    plt.show()
    return path

plot_transform_comparison(
    best_combined_by_transform,
    metric_cv="CV_Mean_R2_Original",
    metric_holdout="Test_R2_Original",
    title="Target-Transform Comparison in Combined Preprocessing - Original R²",
    filename="target_transform_comparison_r2_original.png"
)

plot_transform_comparison(
    best_combined_by_transform,
    metric_cv="CV_Mean_R2_Transformed",
    metric_holdout="R2_Test_Transformed",
    title="Target-Transform Comparison in Combined Preprocessing - Transformed R²",
    filename="target_transform_comparison_r2_transformed.png"
)


# %% [cell 31]
# ============================================================
# CELL 12 - HYPERPARAMETER TUNING ON THE BEST PREPROCESSING CONFIGURATION
# ============================================================
# Tuning is performed only on training data using K-Fold CV.
# The best candidate is selected based on PRIMARY_SELECTION_METRIC.
# The holdout set is not used for parameter selection to avoid leakage.

def sample_param_candidates(model_name, n_iter=N_ITER_TUNING):
    space = TUNING_PARAM_SPACES.get(model_name, {})

    if not space:
        return [{}]

    candidates = list(ParameterGrid(space))
    print(f"Total tuning candidates {model_name} before sampling: {len(candidates)}")

    if n_iter is not None and len(candidates) > n_iter:
        rng = np.random.default_rng(RANDOM_STATE)
        selected_idx = rng.choice(len(candidates), size=n_iter, replace=False)
        candidates = [candidates[i] for i in selected_idx]
        print(f"Tuning candidates {model_name} after sampling: {len(candidates)}")
    else:
        print(f"Tuning candidates {model_name} all candidates used: {len(candidates)}")

    return candidates

tuning_results = []
tuning_fold_details = {}

for _, best_row in best_preprocessing_per_model.iterrows():
    model_name = best_row["Model"]

    print("\n" + "=" * 100)
    print("HYPERPARAMETER TUNING | Model:", model_name)
    print("=" * 100)
    print("Best preprocessing:")
    print("  Imputer:", best_row["Imputer_Mode"])
    print("  Scaler:", best_row["Scaler_Type"])
    print("  Target transform:", best_row["Target_Transform"])
    print("  Target outlier:", best_row["Target_Outlier_Mode"])
    print("  Feature outlier:", best_row["Feature_Outlier_Mode"])
    print("Tuning selection metric:", PRIMARY_SELECTION_METRIC)

    param_candidates = sample_param_candidates(model_name)

    best_candidate = None
    best_cv_score = -np.inf
    candidate_rows = []

    for j, params in enumerate(param_candidates, start=1):
        try:
            cv_summary, fold_df = kfold_cv_score_for_config(
                model_name=model_name,
                X_df=X_train,
                y_series=y_train,
                imputer_mode=best_row["Imputer_Mode"],
                scaler_type=best_row["Scaler_Type"],
                target_transform=best_row["Target_Transform"],
                target_outlier_mode=best_row["Target_Outlier_Mode"],
                feature_outlier_mode=best_row["Feature_Outlier_Mode"],
                model_params=params,
                n_splits=CV_TUNING
            )

            score = cv_summary[PRIMARY_SELECTION_METRIC]

            candidate_row = {
                "Model": model_name,
                "Candidate": j,
                "Params": safe_json(params),
                "Selection_Metric": PRIMARY_SELECTION_METRIC,
                "Selection_Score": score,
                **cv_summary,
                "Error": ""
            }

            if score > best_cv_score:
                best_cv_score = score
                best_candidate = params

            print(
                f"  candidate {j:03d}/{len(param_candidates)} | "
                f"{PRIMARY_SELECTION_METRIC}={score:.4f} | "
                f"CV R2 original={cv_summary['CV_Mean_R2_Original']:.4f} | "
                f"CV R2 transformed={cv_summary['CV_Mean_R2_Transformed']:.4f}"
            )

        except Exception as e:
            candidate_row = {
                "Model": model_name,
                "Candidate": j,
                "Params": safe_json(params),
                "Selection_Metric": PRIMARY_SELECTION_METRIC,
                "Selection_Score": -np.inf,
                "CV_Mean_R2_Transformed": -np.inf,
                "CV_Mean_R2_Original": -np.inf,
                "Error": str(e)
            }
            print(f"  candidate {j:03d}/{len(param_candidates)} failed:", e)

        candidate_rows.append(candidate_row)

    candidates_df = pd.DataFrame(candidate_rows)
    save_table(candidates_df, f"13_tuning_candidates_{re.sub('[^a-zA-Z0-9]+','_',model_name).strip('_')}.csv")

    if best_candidate is None:
        print("No valid tuning candidates for model:", model_name)
        continue

    tuned_row, tuned_pipe, tuned_pred, tuned_folds = evaluate_scenario_with_cv(
        model_name=model_name,
        scenario_name="best_preprocessing_plus_hyperparameter_tuning",
        stage="08_hyperparameter_tuning",
        imputer_mode=best_row["Imputer_Mode"],
        scaler_type=best_row["Scaler_Type"],
        target_transform=best_row["Target_Transform"],
        target_outlier_mode=best_row["Target_Outlier_Mode"],
        feature_outlier_mode=best_row["Feature_Outlier_Mode"],
        model_params=best_candidate,
        tuned=(len(best_candidate) > 0)
    )

    tuned_row["Best_Tuning_Params"] = safe_json(best_candidate)
    tuned_row["Selection_Metric"] = PRIMARY_SELECTION_METRIC
    tuned_row["Best_Tuning_CV_Score_Selected"] = best_cv_score
    tuned_row["Best_Tuning_CV_Score_Original"] = tuned_row["CV_Mean_R2_Original"]
    tuned_row["Best_Tuning_CV_Score_Transformed"] = tuned_row["CV_Mean_R2_Transformed"]

    tuning_results.append(tuned_row)
    tuning_fold_details[model_name] = tuned_folds
    scenario_predictions[(tuned_row["Stage"], model_name)] = tuned_pred

tuning_df = pd.DataFrame(tuning_results)

tuning_paper_df = (
    tuning_df[[
        "Model", "Tuned",
        "Selection_Metric",
        "Best_Tuning_CV_Score_Selected",

        # CV training and validation.
        "CV_Mean_Train_R2_Original",
        "CV_Mean_R2_Original",
        "CV_Mean_Train_Val_Gap_R2_Original",

        # Train-test split original.
        "Train_R2_Original",
        "Test_R2_Original",
        "Train_Test_Gap_R2_Original",

        # Transformed score.
        "CV_Mean_Train_R2_Transformed",
        "CV_Mean_R2_Transformed",
        "R2_Train_Transformed",
        "R2_Test_Transformed",
        "Train_Test_Gap_R2_Transformed",

        "Test_RMSE_Original",
        "Test_MAE_Original",
        "Imputer_Mode",
        "Scaler_Type",
        "Target_Transform",
        "Target_Outlier_Mode",
        "Feature_Outlier_Mode",
        "Best_Tuning_Params"
    ]]
    .sort_values(PRIMARY_SELECTION_METRIC, ascending=False)
    .rename(columns={
        "CV_Mean_Train_R2_Original": "CV_Train_R2_Original",
        "CV_Mean_R2_Original": "CV_Val_R2_Original",
        "CV_Mean_Train_Val_Gap_R2_Original": "CV_Train_Val_Gap_R2_Original",
        "Test_R2_Original": "Holdout_R2_Original",

        "CV_Mean_Train_R2_Transformed": "CV_Train_R2_Transformed",
        "CV_Mean_R2_Transformed": "CV_Val_R2_Transformed",
        "R2_Train_Transformed": "Train_R2_Transformed",
        "R2_Test_Transformed": "Holdout_R2_Transformed",

        "Test_RMSE_Original": "RMSE_Original",
        "Test_MAE_Original": "MAE_Original"
    })
)

# ============================================================
# COMPARISON TABLE OF BASELINE DEFAULT PARAMETERS VS BEST TUNING
# ============================================================

default_vs_tuning_rows = []

for model_name in MODEL_NAMES:
    default_params = get_report_default_params(model_name)

    tuned_match = tuning_df[tuning_df["Model"] == model_name].copy()

    if len(tuned_match) > 0:
        tuned_row = tuned_match.iloc[0]

        best_tuning_params = tuned_row.get("Best_Tuning_Params", "{}")

        default_vs_tuning_rows.append({
            "Model": model_name,

            "Baseline_Default_Params": safe_json(default_params),
            "Best_Tuning_Params": best_tuning_params,

            "Baseline_CV_R2_Original": baseline_df.loc[
                baseline_df["Model"] == model_name,
                "CV_Mean_R2_Original"
            ].iloc[0] if len(baseline_df[baseline_df["Model"] == model_name]) > 0 else np.nan,

            "Baseline_Holdout_R2_Original": baseline_df.loc[
                baseline_df["Model"] == model_name,
                "Test_R2_Original"
            ].iloc[0] if len(baseline_df[baseline_df["Model"] == model_name]) > 0 else np.nan,

            "Tuning_CV_R2_Original": tuned_row.get("CV_Mean_R2_Original", np.nan),
            "Tuning_Holdout_R2_Original": tuned_row.get("Test_R2_Original", np.nan),
            "Tuning_RMSE_Original": tuned_row.get("Test_RMSE_Original", np.nan),
            "Tuning_MAE_Original": tuned_row.get("Test_MAE_Original", np.nan),

            "Preprocessing_Used_For_Tuning": (
                f"imputer={tuned_row.get('Imputer_Mode', '')}; "
                f"scaler={tuned_row.get('Scaler_Type', '')}; "
                f"target_transform={tuned_row.get('Target_Transform', '')}; "
                f"target_outlier={tuned_row.get('Target_Outlier_Mode', '')}; "
                f"feature_outlier={tuned_row.get('Feature_Outlier_Mode', '')}"
            )
        })

    else:
        default_vs_tuning_rows.append({
            "Model": model_name,
            "Baseline_Default_Params": safe_json(default_params),
            "Best_Tuning_Params": "",
            "Baseline_CV_R2_Original": np.nan,
            "Baseline_Holdout_R2_Original": np.nan,
            "Tuning_CV_R2_Original": np.nan,
            "Tuning_Holdout_R2_Original": np.nan,
            "Tuning_RMSE_Original": np.nan,
            "Tuning_MAE_Original": np.nan,
            "Preprocessing_Used_For_Tuning": ""
        })

default_vs_tuning_df = pd.DataFrame(default_vs_tuning_rows)

pd.set_option("display.max_colwidth", None)

print("COMPARISON OF BASELINE DEFAULT PARAMETERS VS BEST HYPERPARAMETER TUNING")
display(default_vs_tuning_df)

save_table(
    default_vs_tuning_df,
    "15b_default_params_vs_best_tuning_params.csv"
)

display(tuning_paper_df)

save_table(tuning_df, "14_hyperparameter_tuning_results_full.csv")
save_table(tuning_paper_df, "14_hyperparameter_tuning_results_paper.csv")


# %% [cell 32]
# ============================================================
# CELL 13 - BASELINE vs PREPROCESSING vs TUNING COMPARISON TABLE
# ============================================================
# The final model is selected based on training CV, not holdout.
# The holdout set is used only as the final evaluation to avoid data leakage.

all_results_df = pd.concat([
    baseline_df,
    separate_preprocessing_df,
    combined_preprocessing_df,
    tuning_df
], ignore_index=True, sort=False)

def add_improvement_vs_baseline(df, baseline_stage="01_baseline_default"):
    df = df.copy()

    baseline_by_model = (
        df[df["Stage"] == baseline_stage]
        .sort_values(PRIMARY_SELECTION_METRIC, ascending=False)
        .groupby("Model", as_index=False)
        .head(1)
        .set_index("Model")
    )

    rows = []
    for _, row in df.iterrows():
        model = row["Model"]
        if model not in baseline_by_model.index:
            rows.append({})
            continue

        base = baseline_by_model.loc[model]

        delta_r2_test = row["Test_R2_Original"] - base["Test_R2_Original"]
        delta_r2_cv_original = row["CV_Mean_R2_Original"] - base["CV_Mean_R2_Original"]
        delta_r2_cv_transformed = row["CV_Mean_R2_Transformed"] - base["CV_Mean_R2_Transformed"]

        rmse_improvement_pct = (
            (base["Test_RMSE_Original"] - row["Test_RMSE_Original"]) / abs(base["Test_RMSE_Original"]) * 100
            if base["Test_RMSE_Original"] != 0 else np.nan
        )
        mae_improvement_pct = (
            (base["Test_MAE_Original"] - row["Test_MAE_Original"]) / abs(base["Test_MAE_Original"]) * 100
            if base["Test_MAE_Original"] != 0 else np.nan
        )

        if abs(base["Test_R2_Original"]) < 1e-6:
            r2_relative_pct = np.nan
        else:
            r2_relative_pct = delta_r2_test / abs(base["Test_R2_Original"]) * 100

        rows.append({
            "Baseline_Test_R2_Original": base["Test_R2_Original"],
            "Delta_Test_R2_Original": delta_r2_test,
            "Delta_Test_R2_Percentage_Points": delta_r2_test * 100,
            "Relative_Test_R2_Improvement_Percent": r2_relative_pct,
            "Baseline_CV_R2_Original": base["CV_Mean_R2_Original"],
            "Delta_CV_R2_Original": delta_r2_cv_original,
            "Delta_CV_R2_Original_Percentage_Points": delta_r2_cv_original * 100,
            "Baseline_CV_R2_Transformed": base["CV_Mean_R2_Transformed"],
            "Delta_CV_R2_Transformed": delta_r2_cv_transformed,
            "Delta_CV_R2_Transformed_Percentage_Points": delta_r2_cv_transformed * 100,
            "Baseline_Test_RMSE_Original": base["Test_RMSE_Original"],
            "Test_RMSE_Improvement_Percent": rmse_improvement_pct,
            "Baseline_Test_MAE_Original": base["Test_MAE_Original"],
            "Test_MAE_Improvement_Percent": mae_improvement_pct
        })

    imp_df = pd.DataFrame(rows)
    return pd.concat([df.reset_index(drop=True), imp_df], axis=1)

comparison_df = add_improvement_vs_baseline(all_results_df)

main_stages = [
    "01_baseline_default",
    "07_combined_preprocessing",
    "08_hyperparameter_tuning"
]

main_comparison_df = comparison_df[comparison_df["Stage"].isin(main_stages)].copy()

best_combined_keys = set(zip(best_preprocessing_per_model["Model"], best_preprocessing_per_model["Scenario_Name"]))

main_comparison_df = main_comparison_df[
    (main_comparison_df["Stage"] != "07_combined_preprocessing") |
    (main_comparison_df.apply(lambda r: (r["Model"], r["Scenario_Name"]) in best_combined_keys, axis=1))
].copy()

display(main_comparison_df[[
    "Stage", "Model", "Tuned",

    # CV training and validation.
    "CV_Mean_Train_R2_Original",
    "CV_Mean_R2_Original",
    "CV_Mean_Train_Val_Gap_R2_Original",

    # Train-test split.
    "Train_R2_Original",
    "Test_R2_Original",
    "Train_Test_Gap_R2_Original",

    "Delta_Test_R2_Percentage_Points",
    "Test_RMSE_Original", "Test_RMSE_Improvement_Percent",
    "Test_MAE_Original", "Test_MAE_Improvement_Percent",

    "Imputer_Mode", "Scaler_Type",
    "Target_Transform", "Target_Outlier_Mode", "Feature_Outlier_Mode"
]].sort_values(["Model", "Stage"]))

save_table(comparison_df, "15_all_results_with_baseline_improvement.csv")
save_table(main_comparison_df, "16_main_baseline_preprocessing_tuning_comparison.csv")

best_final_per_model = (
    pd.concat([best_preprocessing_per_model, tuning_df], ignore_index=True, sort=False)
    .sort_values(PRIMARY_SELECTION_METRIC, ascending=False)
    .groupby("Model", as_index=False)
    .head(1)
    .reset_index(drop=True)
)

best_overall = best_final_per_model.sort_values(PRIMARY_SELECTION_METRIC, ascending=False).iloc[0]

print("\nBEST FINAL PER MODEL")
print("Final selection metric:", PRIMARY_SELECTION_METRIC)

display(best_final_per_model[[
    "Model", "Stage", "Tuned",

    "CV_Mean_Train_R2_Original",
    "CV_Mean_R2_Original",
    "CV_Mean_Train_Val_Gap_R2_Original",

    "Train_R2_Original",
    "Test_R2_Original",
    "Train_Test_Gap_R2_Original",

    "Test_RMSE_Original", "Test_MAE_Original",
    "Imputer_Mode", "Scaler_Type",
    "Target_Transform", "Target_Outlier_Mode", "Feature_Outlier_Mode",
    "Model_Params"
]])

save_table(best_final_per_model, "17_best_final_per_model.csv")

print("\nBEST OVERALL BASED ON TRAINING CV:")
display(pd.DataFrame([best_overall]))

if "Train_Test_Gap_R2_Original" in best_final_per_model.columns:
    gap_warning_df = best_final_per_model[
        best_final_per_model["Train_Test_Gap_R2_Original"].abs() > 0.20
    ].copy()

    if len(gap_warning_df) > 0:
        print("\nOVERFITTING AUDIT WARNING:")
        print("At least one model has an original-scale Train-Test R2 gap > 0.20.")
        print("Such models are not automatically invalid, but they should be discussed as potential overfitting indications.")
        display(gap_warning_df[[
            "Model", "Stage", "CV_Mean_R2_Original", "Test_R2_Original",
            "Train_Test_Gap_R2_Original", "Imputer_Mode", "Scaler_Type",
            "Target_Transform", "Target_Outlier_Mode", "Feature_Outlier_Mode"
        ]])

# %% [markdown]
# ## Complete preprocessing-configuration audit for all scenarios
#
# This cell creates two output types:
#
# 1. **Audit all unique preprocessing configurations** from baseline, single preprocessing, combined preprocessing, and tuning. This answers which imputer, scaler, target transform, outlier handling, remaining train/test rows, and encoded-feature count are used by each scenario.
# 2. **Preprocessing matrices for key scenarios**, namely baseline, best combined per model, tuning, and best final per model. Full matrices for all combinations can be enabled with `EXPORT_ALL_UNIQUE_PREPROCESSING_MATRICES = True`, but the default is disabled because the full grid can generate many files.


# %% [cell 34]
# ============================================================
# CELL 13D - PREPROCESSING AUDIT FOR ALL SCENARIOS AND KEY MATRIX EXPORTS
# ============================================================
# This does not modify model results. It only reads scenario-result tables and creates preprocessing audits and post-preprocessing snapshots.

PREPROCESSING_CONFIG_COLS = [
    "Imputer_Mode", "Scaler_Type", "Target_Transform", "Target_Outlier_Mode", "Feature_Outlier_Mode"
]

available_result_frames = []
for frame_name in ["baseline_df", "separate_preprocessing_df", "combined_preprocessing_df", "tuning_df"]:
    if frame_name in globals():
        temp = globals()[frame_name].copy()
        temp["Source_Frame"] = frame_name
        available_result_frames.append(temp)

if len(available_result_frames) == 0:
    raise RuntimeError("No scenario-result tables are available yet for audit.")

scenario_preprocessing_map_df = pd.concat(available_result_frames, ignore_index=True, sort=False)
for col in PREPROCESSING_CONFIG_COLS:
    if col not in scenario_preprocessing_map_df.columns:
        scenario_preprocessing_map_df[col] = np.nan

map_cols = [
    "Source_Frame", "Stage", "Scenario_Name", "Model",
    *PREPROCESSING_CONFIG_COLS,
    "Selected_Feature_Count", "Tuned", "Model_Params",
    "CV_Mean_R2_Original", "CV_Std_R2_Original", "Test_R2_Original", "Test_RMSE_Original", "Test_MAE_Original", "Error"
]
map_cols = [c for c in map_cols if c in scenario_preprocessing_map_df.columns]
scenario_preprocessing_map_df = scenario_preprocessing_map_df[map_cols].copy()

# Add a unique configuration ID so each scenario can be traced to the preprocessing audit.
config_key_df = scenario_preprocessing_map_df[PREPROCESSING_CONFIG_COLS].fillna("missing").astype(str)
scenario_preprocessing_map_df["Preprocessing_Config_ID"] = config_key_df.apply(
    lambda r: reviewer_safe_filename("__".join([f"{c}-{r[c]}" for c in PREPROCESSING_CONFIG_COLS]), max_len=200),
    axis=1
)

unique_preprocessing_configs_df = (
    scenario_preprocessing_map_df[["Preprocessing_Config_ID", *PREPROCESSING_CONFIG_COLS]]
    .drop_duplicates()
    .reset_index(drop=True)
)

preprocessing_audit_rows = []
for _, cfg_row in unique_preprocessing_configs_df.iterrows():
    cfg = {
        "imputer_mode": cfg_row["Imputer_Mode"],
        "scaler_type": cfg_row["Scaler_Type"],
        "target_transform": cfg_row["Target_Transform"],
        "target_outlier_mode": cfg_row["Target_Outlier_Mode"],
        "feature_outlier_mode": cfg_row["Feature_Outlier_Mode"],
    }
    save_prefix = None
    export_matrix = False
    if EXPORT_ALL_UNIQUE_PREPROCESSING_MATRICES:
        save_prefix = f"all_unique_{cfg_row['Preprocessing_Config_ID']}"
        export_matrix = True

    try:
        meta, _ = export_preprocessing_snapshot(
            cfg,
            stage="unique_preprocessing_config_audit",
            scenario_name=cfg_row["Preprocessing_Config_ID"],
            model_name="not_model_specific",
            save_prefix=save_prefix,
            export_full_matrices=export_matrix,
            export_preview=False,
        )
        meta["Preprocessing_Config_ID"] = cfg_row["Preprocessing_Config_ID"]
        meta["Audit_Error"] = ""
    except Exception as exc:
        meta = cfg.to_dict() if hasattr(cfg, "to_dict") else dict(cfg)
        meta.update({
            "Preprocessing_Config_ID": cfg_row["Preprocessing_Config_ID"],
            "Stage": "unique_preprocessing_config_audit",
            "Scenario_Name": cfg_row["Preprocessing_Config_ID"],
            "Audit_Error": str(exc),
        })
    preprocessing_audit_rows.append(meta)

preprocessing_config_audit_df = pd.DataFrame(preprocessing_audit_rows)

save_table(scenario_preprocessing_map_df, "16b_scenario_to_preprocessing_config_map.csv")
save_table(unique_preprocessing_configs_df, "16c_unique_preprocessing_config_list.csv")
save_table(preprocessing_config_audit_df, "16d_unique_preprocessing_config_data_flow_audit.csv")

print("Number of mapped scenario rows:", len(scenario_preprocessing_map_df))
print("Number of unique preprocessing configurations:", len(unique_preprocessing_configs_df))
display(preprocessing_config_audit_df.head(20))

# Export full matrices for key scenarios: baseline, best combined, tuning, and best final.
key_frames = []
if "baseline_df" in globals():
    key_frames.append(baseline_df.assign(Key_Source="baseline_all_models"))
if "best_preprocessing_per_model" in globals():
    key_frames.append(best_preprocessing_per_model.assign(Key_Source="best_combined_per_model"))
if "tuning_df" in globals():
    key_frames.append(tuning_df.assign(Key_Source="tuning_per_model"))
if "best_final_per_model" in globals():
    key_frames.append(best_final_per_model.assign(Key_Source="best_final_per_model"))

key_export_rows = []
if EXPORT_SELECTED_PREPROCESSED_MATRICES and len(key_frames) > 0:
    key_scenario_df = pd.concat(key_frames, ignore_index=True, sort=False)
    key_scenario_df = key_scenario_df.drop_duplicates(
        subset=["Key_Source", "Model", "Stage", "Scenario_Name", *PREPROCESSING_CONFIG_COLS]
    ).reset_index(drop=True)

    for _, row in key_scenario_df.iterrows():
        cfg = {
            "imputer_mode": row.get("Imputer_Mode", "median"),
            "scaler_type": row.get("Scaler_Type", "none"),
            "target_transform": row.get("Target_Transform", "none"),
            "target_outlier_mode": row.get("Target_Outlier_Mode", "none"),
            "feature_outlier_mode": row.get("Feature_Outlier_Mode", "none"),
        }
        prefix = reviewer_safe_filename(
            f"{row.get('Key_Source','key')}_{row.get('Stage','stage')}_{row.get('Model','model')}_{row.get('Scenario_Name','scenario')}",
            max_len=180
        )
        try:
            meta, _ = export_preprocessing_snapshot(
                cfg,
                stage=row.get("Stage", "key_scenario"),
                scenario_name=row.get("Scenario_Name", "scenario"),
                model_name=row.get("Model", "model"),
                save_prefix=prefix,
                export_full_matrices=True,
                export_preview=True,
            )
            meta["Key_Source"] = row.get("Key_Source", "")
            meta["File_Prefix"] = prefix
            meta["Export_Error"] = ""
        except Exception as exc:
            meta = {
                "Key_Source": row.get("Key_Source", ""),
                "Stage": row.get("Stage", ""),
                "Scenario_Name": row.get("Scenario_Name", ""),
                "Model": row.get("Model", ""),
                "File_Prefix": prefix,
                "Export_Error": str(exc),
            }
        key_export_rows.append(meta)

key_preprocessed_matrix_export_log_df = pd.DataFrame(key_export_rows)
save_table(key_preprocessed_matrix_export_log_df, "16e_key_scenario_preprocessed_matrix_export_log.csv")
display(key_preprocessed_matrix_export_log_df)



# %% [markdown]
# ## Additional analysis: data condition after best preprocessing and evidence that preprocessing is not universal
#
# This additional cell addresses reviewer/manuscript needs:
#
# 1. Save train/test data after processing by the **best preprocessing combination for each model**.
# 2. Compare data conditions before and after the best preprocessing.
# 3. Test whether the best preprocessing of one model remains best when applied to another model.
# 4. Provide data evidence that dominant preprocessing is not automatically the universal best preprocessing.


# %% [cell 13E]
# ============================================================
# CELL 13E - DATA AFTER BEST COMBINED PREPROCESSING PER MODEL
# ============================================================
# Main outputs:
# - tables/30a_best_combined_preprocessing_data_flow_by_model.csv
# - tables/30b_best_combined_numeric_feature_stats_before_vs_after.csv
# - tables/30c_best_combined_preprocessing_config_by_model.csv
# - data_snapshots/best_combined_preprocessing_<model>_*_after_preprocessing_unencoded.csv
# - preprocessed_matrices/best_combined_preprocessing_<model>_*_after_preprocessing_encoded.csv
#
# Methodological notes:
# - The preprocessor is fitted on training data and used to transform test data.
# - This output is for audit and reporting, not for modifying the model-selection pipeline.

BEST_PREPROCESSING_EXPORT_DIR = OUTPUT_DIR / "best_preprocessing_by_model"
BEST_PREPROCESSING_EXPORT_DIR.mkdir(parents=True, exist_ok=True)

best_combined_snapshot_rows = []
best_combined_numeric_stats_rows = []
best_combined_sample_rows = []

baseline_train_rows_after = baseline_preprocessing_meta.get("Train_Rows_After_Feature_Outlier", np.nan)
baseline_test_rows_after = baseline_preprocessing_meta.get("Test_Rows_After_Feature_Outlier", np.nan)
baseline_encoded_feature_count = baseline_preprocessing_meta.get("Encoded_Feature_Count", np.nan)

for _, best_cfg_row in best_preprocessing_per_model.iterrows():
    model_name = best_cfg_row["Model"]
    model_slug = reviewer_safe_filename(model_name)

    cfg = {
        "imputer_mode": best_cfg_row["Imputer_Mode"],
        "scaler_type": best_cfg_row["Scaler_Type"],
        "target_transform": best_cfg_row["Target_Transform"],
        "target_outlier_mode": best_cfg_row["Target_Outlier_Mode"],
        "feature_outlier_mode": best_cfg_row["Feature_Outlier_Mode"],
    }

    prefix = f"best_combined_preprocessing_{model_slug}"

    meta, obj = export_preprocessing_snapshot(
        cfg,
        stage="best_combined_preprocessing_data_audit",
        scenario_name=best_cfg_row.get("Scenario_Name", "best_combined_preprocessing"),
        model_name=model_name,
        selected_features=FEATURE_COLS,
        save_prefix=prefix,
        export_full_matrices=True,
        export_preview=True,
        preview_rows=50,
    )

    meta["Model"] = model_name
    meta["Best_CV_R2_Original"] = best_cfg_row.get("CV_Mean_R2_Original", np.nan)
    meta["Best_CV_Std_R2_Original"] = best_cfg_row.get("CV_Std_R2_Original", np.nan)
    meta["Best_Holdout_R2_Original"] = best_cfg_row.get("Test_R2_Original", np.nan)
    meta["Best_Holdout_RMSE_Original"] = best_cfg_row.get("Test_RMSE_Original", np.nan)
    meta["Best_Holdout_MAE_Original"] = best_cfg_row.get("Test_MAE_Original", np.nan)
    meta["Baseline_Train_Rows_After_Preprocessing"] = baseline_train_rows_after
    meta["Baseline_Test_Rows_After_Preprocessing"] = baseline_test_rows_after
    meta["Baseline_Encoded_Feature_Count"] = baseline_encoded_feature_count
    meta["Delta_Train_Rows_vs_Baseline"] = meta.get("Train_Rows_After_Feature_Outlier", np.nan) - baseline_train_rows_after
    meta["Delta_Test_Rows_vs_Baseline"] = meta.get("Test_Rows_After_Feature_Outlier", np.nan) - baseline_test_rows_after
    meta["Delta_Encoded_Feature_Count_vs_Baseline"] = meta.get("Encoded_Feature_Count", np.nan) - baseline_encoded_feature_count

    # Important file locations for easier retrieval.
    meta["Train_After_Unencoded_File"] = str(DATA_SNAPSHOT_DIR / f"{prefix}_train_after_preprocessing_unencoded.csv")
    meta["Test_After_Unencoded_File"] = str(DATA_SNAPSHOT_DIR / f"{prefix}_test_after_preprocessing_unencoded.csv")
    meta["Train_After_Encoded_File"] = str(PREPROCESSED_MATRIX_DIR / f"{prefix}_train_after_preprocessing_encoded.csv")
    meta["Test_After_Encoded_File"] = str(PREPROCESSED_MATRIX_DIR / f"{prefix}_test_after_preprocessing_encoded.csv")
    best_combined_snapshot_rows.append(meta)

    # Numeric statistics before vs after encoded preprocessing.
    X_train_encoded = obj["X_train_encoded"].copy()
    X_test_encoded = obj["X_test_encoded"].copy()

    for col in NUMERIC_COLS:
        before_train = pd.to_numeric(X_train[col], errors="coerce")
        before_test = pd.to_numeric(X_test[col], errors="coerce")
        encoded_col = f"num__{col}"
        if encoded_col not in X_train_encoded.columns:
            # fallback if the sklearn version produces a different name.
            matches = [c for c in X_train_encoded.columns if c.endswith(col)]
            encoded_col = matches[0] if matches else None

        if encoded_col is not None:
            after_train = pd.to_numeric(X_train_encoded[encoded_col], errors="coerce")
            after_test = pd.to_numeric(X_test_encoded[encoded_col], errors="coerce")
        else:
            after_train = pd.Series(dtype=float)
            after_test = pd.Series(dtype=float)

        best_combined_numeric_stats_rows.append({
            "Model": model_name,
            "Numeric_Feature": col,
            "Encoded_Feature_Name": encoded_col,
            "Imputer_Mode": cfg["imputer_mode"],
            "Scaler_Type": cfg["scaler_type"],
            "Feature_Outlier_Mode": cfg["feature_outlier_mode"],
            "Before_Train_Missing_Count": int(before_train.isna().sum()),
            "After_Train_Missing_Count": int(after_train.isna().sum()) if len(after_train) else np.nan,
            "Before_Test_Missing_Count": int(before_test.isna().sum()),
            "After_Test_Missing_Count": int(after_test.isna().sum()) if len(after_test) else np.nan,
            "Before_Train_Min": float(before_train.min()) if before_train.notna().any() else np.nan,
            "Before_Train_Max": float(before_train.max()) if before_train.notna().any() else np.nan,
            "Before_Train_Mean": float(before_train.mean()) if before_train.notna().any() else np.nan,
            "Before_Train_Std": float(before_train.std()) if before_train.notna().any() else np.nan,
            "After_Train_Min": float(after_train.min()) if len(after_train) and after_train.notna().any() else np.nan,
            "After_Train_Max": float(after_train.max()) if len(after_train) and after_train.notna().any() else np.nan,
            "After_Train_Mean": float(after_train.mean()) if len(after_train) and after_train.notna().any() else np.nan,
            "After_Train_Std": float(after_train.std()) if len(after_train) and after_train.notna().any() else np.nan,
        })

    # Example of the first 20 rows: numeric values before and after encoding.
    before_sample = X_train[NUMERIC_COLS].head(20).reset_index().rename(columns={"index": "Original_Index"})
    before_sample.insert(0, "Model", model_name)
    after_numeric_cols = []
    for col in NUMERIC_COLS:
        encoded_col = f"num__{col}"
        if encoded_col in X_train_encoded.columns:
            after_numeric_cols.append(encoded_col)
    after_sample = X_train_encoded[after_numeric_cols].head(20).reset_index(drop=True)
    after_sample.columns = [c.replace("num__", "After_") for c in after_sample.columns]
    sample_df = pd.concat([before_sample.reset_index(drop=True), after_sample.reset_index(drop=True)], axis=1)
    best_combined_sample_rows.append(sample_df)

best_combined_data_flow_df = pd.DataFrame(best_combined_snapshot_rows)
best_combined_numeric_stats_df = pd.DataFrame(best_combined_numeric_stats_rows)
best_combined_sample_before_after_df = pd.concat(best_combined_sample_rows, ignore_index=True) if best_combined_sample_rows else pd.DataFrame()

config_cols_for_article = [
    "Model", "Best_CV_R2_Original", "Best_CV_Std_R2_Original", "Best_Holdout_R2_Original",
    "Best_Holdout_RMSE_Original", "Best_Holdout_MAE_Original",
    "Imputer_Mode", "Scaler_Type", "Target_Transform", "Target_Outlier_Mode", "Feature_Outlier_Mode",
    "Train_Rows_Before_Preprocessing", "Train_Rows_After_Missing_Handling", "Train_Rows_After_Target_Outlier",
    "Train_Rows_After_Feature_Outlier", "Test_Rows_Before_Preprocessing", "Test_Rows_After_Feature_Outlier",
    "Train_Missing_Cells_Before_Preprocessing", "Encoded_Missing_Cells_Train", "Encoded_Feature_Count",
    "Feature_Outlier_Cell_Capped_Train", "Delta_Train_Rows_vs_Baseline", "Delta_Test_Rows_vs_Baseline",
    "Train_After_Unencoded_File", "Train_After_Encoded_File"
]
config_cols_for_article = [c for c in config_cols_for_article if c in best_combined_data_flow_df.columns]

save_table(best_combined_data_flow_df, "30a_best_combined_preprocessing_data_flow_by_model.csv")
save_table(best_combined_data_flow_df[config_cols_for_article], "30c_best_combined_preprocessing_config_by_model.csv")
save_table(best_combined_numeric_stats_df, "30b_best_combined_numeric_feature_stats_before_vs_after.csv")
save_table(best_combined_sample_before_after_df, "30d_best_combined_sample_numeric_before_after.csv")

best_preprocessing_note = """
Data after best preprocessing are saved per model because the best combination is not universal. The unencoded file shows data after the missing-row policy, target-outlier handling, and feature-outlier handling. The encoded file shows the final numeric matrix after imputation, scaling, and one-hot encoding that actually enters the model. To avoid leakage, the preprocessor is fitted on train data and used to transform test data.
""".strip()
save_text(best_preprocessing_note, "30_best_combined_preprocessing_data_note.txt")

print("Data after best preprocessing for each model have been exported.")
display(best_combined_data_flow_df[config_cols_for_article])
display(best_combined_numeric_stats_df.head(30))


# %% [cell 13F]
# ============================================================
# CELL 13F - DATA EVIDENCE: THE BEST PREPROCESSING FOR MODEL A IS NOT NECESSARILY BEST FOR MODEL B
# ============================================================
# This cell creates a transfer matrix: the best preprocessing from each model is applied to all other models and then compared with the best preprocessing
# the evaluated model. This supports the discussion that preprocessing is
# model-specific.

RUN_CROSS_MODEL_BEST_PREPROCESSING_TRANSFER = True

cross_transfer_rows = []
cross_transfer_fold_rows = []

if RUN_CROSS_MODEL_BEST_PREPROCESSING_TRANSFER:
    for _, source_row in best_preprocessing_per_model.iterrows():
        source_model = source_row["Model"]
        source_cfg = {
            "imputer_mode": source_row["Imputer_Mode"],
            "scaler_type": source_row["Scaler_Type"],
            "target_transform": source_row["Target_Transform"],
            "target_outlier_mode": source_row["Target_Outlier_Mode"],
            "feature_outlier_mode": source_row["Feature_Outlier_Mode"],
        }

        for target_model in MODEL_NAMES:
            scenario_name = f"transfer_best_preprocessing_from_{reviewer_safe_filename(source_model)}_to_{reviewer_safe_filename(target_model)}"
            try:
                row, pipe, pred_df, fold_df = evaluate_scenario_with_cv(
                    model_name=target_model,
                    scenario_name=scenario_name,
                    stage="10_cross_model_preprocessing_transfer",
                    imputer_mode=source_cfg["imputer_mode"],
                    scaler_type=source_cfg["scaler_type"],
                    target_transform=source_cfg["target_transform"],
                    target_outlier_mode=source_cfg["target_outlier_mode"],
                    feature_outlier_mode=source_cfg["feature_outlier_mode"],
                    model_params=None,
                    selected_features=FEATURE_COLS,
                    tuned=False,
                    cv_k=CV_TUNING,
                )
                row["Preprocessing_Source_Model"] = source_model
                row["Evaluated_Model"] = target_model
                row["Is_Own_Best_Preprocessing_Source"] = bool(source_model == target_model)
                row["Transfer_Error"] = ""
                fold_df["Preprocessing_Source_Model"] = source_model
                fold_df["Evaluated_Model"] = target_model
                cross_transfer_fold_rows.append(fold_df)
            except Exception as exc:
                row = {
                    "Stage": "10_cross_model_preprocessing_transfer",
                    "Scenario_Name": scenario_name,
                    "Preprocessing_Source_Model": source_model,
                    "Evaluated_Model": target_model,
                    "Model": target_model,
                    "Is_Own_Best_Preprocessing_Source": bool(source_model == target_model),
                    "Imputer_Mode": source_cfg["imputer_mode"],
                    "Scaler_Type": source_cfg["scaler_type"],
                    "Target_Transform": source_cfg["target_transform"],
                    "Target_Outlier_Mode": source_cfg["target_outlier_mode"],
                    "Feature_Outlier_Mode": source_cfg["feature_outlier_mode"],
                    "CV_Mean_R2_Original": np.nan,
                    "CV_Std_R2_Original": np.nan,
                    "Test_R2_Original": np.nan,
                    "Test_RMSE_Original": np.nan,
                    "Test_MAE_Original": np.nan,
                    "Transfer_Error": str(exc),
                }
            cross_transfer_rows.append(row)

cross_model_preprocessing_transfer_df = pd.DataFrame(cross_transfer_rows)
if len(cross_transfer_fold_rows) > 0:
    cross_model_preprocessing_transfer_folds_df = pd.concat(cross_transfer_fold_rows, ignore_index=True)
else:
    cross_model_preprocessing_transfer_folds_df = pd.DataFrame()

own_best_cv_map = best_preprocessing_per_model.set_index("Model")["CV_Mean_R2_Original"].to_dict()
own_best_rmse_map = best_preprocessing_per_model.set_index("Model")["Test_RMSE_Original"].to_dict()

cross_model_preprocessing_transfer_df["Own_Best_CV_R2_Original"] = cross_model_preprocessing_transfer_df["Evaluated_Model"].map(own_best_cv_map)
cross_model_preprocessing_transfer_df["Delta_CV_R2_vs_Evaluated_Model_Own_Best"] = (
    cross_model_preprocessing_transfer_df["CV_Mean_R2_Original"]
    - cross_model_preprocessing_transfer_df["Own_Best_CV_R2_Original"]
)
cross_model_preprocessing_transfer_df["Own_Best_RMSE_Original"] = cross_model_preprocessing_transfer_df["Evaluated_Model"].map(own_best_rmse_map)
cross_model_preprocessing_transfer_df["Delta_RMSE_vs_Evaluated_Model_Own_Best"] = (
    cross_model_preprocessing_transfer_df["Test_RMSE_Original"]
    - cross_model_preprocessing_transfer_df["Own_Best_RMSE_Original"]
)

transfer_matrix_cv_r2_df = cross_model_preprocessing_transfer_df.pivot_table(
    index="Preprocessing_Source_Model",
    columns="Evaluated_Model",
    values="CV_Mean_R2_Original",
    aggfunc="first"
).reset_index()

transfer_delta_matrix_cv_r2_df = cross_model_preprocessing_transfer_df.pivot_table(
    index="Preprocessing_Source_Model",
    columns="Evaluated_Model",
    values="Delta_CV_R2_vs_Evaluated_Model_Own_Best",
    aggfunc="first"
).reset_index()

save_table(cross_model_preprocessing_transfer_df, "31a_cross_model_best_preprocessing_transfer_results.csv")
save_table(transfer_matrix_cv_r2_df, "31b_cross_model_best_preprocessing_transfer_cv_r2_matrix.csv")
save_table(transfer_delta_matrix_cv_r2_df, "31c_cross_model_best_preprocessing_transfer_delta_cv_r2_matrix.csv")
if len(cross_model_preprocessing_transfer_folds_df) > 0:
    save_table(cross_model_preprocessing_transfer_folds_df, "31d_cross_model_best_preprocessing_transfer_fold_details.csv")

print("Best-preprocessing transfer matrix across models:")
display(transfer_matrix_cv_r2_df)
print("Delta against the best preprocessing of each evaluated model:")
display(transfer_delta_matrix_cv_r2_df)


# %% [cell 13G]
# ============================================================
# CELL 13G - DOMINANT PREPROCESSING METHODS AND DATA-BASED REASONS
# ============================================================
# Dominant = frequently appears in the best configuration per model.
# Dominant is not the same as universally best. Universality is tested through the transfer matrix.

component_cols = ["Imputer_Mode", "Scaler_Type", "Target_Transform", "Target_Outlier_Mode", "Feature_Outlier_Mode"]

dominant_method_rows = []
for comp in component_cols:
    counts = best_preprocessing_per_model[comp].value_counts(dropna=False)
    for method, count in counts.items():
        models_using = best_preprocessing_per_model.loc[best_preprocessing_per_model[comp] == method, "Model"].tolist()
        dominant_method_rows.append({
            "Component": comp,
            "Method": method,
            "Count_In_Best_Config": int(count),
            "Percentage_In_Best_Config": float(count / len(best_preprocessing_per_model) * 100),
            "Models_Using_As_Best": safe_json(models_using),
            "Is_Dominant_In_Component": bool(count == counts.max())
        })

dominant_method_frequency_df = pd.DataFrame(dominant_method_rows).sort_values(
    ["Component", "Count_In_Best_Config"], ascending=[True, False]
)
save_table(dominant_method_frequency_df, "32a_dominant_preprocessing_method_frequency.csv")

# For each dominant method, check the best performance that uses that method for each model.
dominant_effect_rows = []
for _, dom_row in dominant_method_frequency_df[dominant_method_frequency_df["Is_Dominant_In_Component"]].iterrows():
    comp = dom_row["Component"]
    method = dom_row["Method"]
    for model_name in MODEL_NAMES:
        own_best = best_preprocessing_per_model[best_preprocessing_per_model["Model"] == model_name].iloc[0]
        subset = combined_preprocessing_df[
            (combined_preprocessing_df["Model"] == model_name) &
            (combined_preprocessing_df[comp] == method) &
            np.isfinite(combined_preprocessing_df["CV_Mean_R2_Original"])
        ].copy()
        if len(subset) == 0:
            dominant_effect_rows.append({
                "Component": comp,
                "Dominant_Method": method,
                "Model": model_name,
                "Best_CV_R2_With_Dominant_Method": np.nan,
                "Own_Best_CV_R2": own_best["CV_Mean_R2_Original"],
                "Delta_vs_Own_Best": np.nan,
                "Dominant_Method_Is_Model_Best": False,
                "Best_Config_With_Dominant_Method": "not_available"
            })
            continue
        best_with_method = subset.sort_values("CV_Mean_R2_Original", ascending=False).iloc[0]
        dominant_effect_rows.append({
            "Component": comp,
            "Dominant_Method": method,
            "Model": model_name,
            "Best_CV_R2_With_Dominant_Method": best_with_method["CV_Mean_R2_Original"],
            "Own_Best_CV_R2": own_best["CV_Mean_R2_Original"],
            "Delta_vs_Own_Best": best_with_method["CV_Mean_R2_Original"] - own_best["CV_Mean_R2_Original"],
            "Dominant_Method_Is_Model_Best": bool(best_with_method["Scenario_Name"] == own_best["Scenario_Name"]),
            "Best_Config_With_Dominant_Method": best_with_method["Scenario_Name"],
            "Own_Best_Config": own_best["Scenario_Name"],
            "Own_Best_Imputer_Mode": own_best["Imputer_Mode"],
            "Own_Best_Scaler_Type": own_best["Scaler_Type"],
            "Own_Best_Target_Transform": own_best["Target_Transform"],
            "Own_Best_Target_Outlier_Mode": own_best["Target_Outlier_Mode"],
            "Own_Best_Feature_Outlier_Mode": own_best["Feature_Outlier_Mode"],
        })

dominant_method_effect_by_model_df = pd.DataFrame(dominant_effect_rows)
save_table(dominant_method_effect_by_model_df, "32b_dominant_preprocessing_method_effect_by_model.csv")

# Tuning-parameter analysis: which parameter values have large effects.
parameter_effect_rows = []
parameter_summary_rows = []

for path in sorted(TABLE_DIR.glob("13_tuning_candidates_*.csv")):
    try:
        cand_df = pd.read_csv(path)
    except Exception:
        continue
    if cand_df.empty or "Params" not in cand_df.columns:
        continue
    model_name = cand_df["Model"].iloc[0] if "Model" in cand_df.columns and len(cand_df) else path.stem.replace("13_tuning_candidates_", "")
    parsed_params = []
    for params_text in cand_df["Params"].fillna("{}").astype(str):
        try:
            params = json.loads(params_text)
        except Exception:
            params = {}
        parsed_params.append({k.replace("model__", ""): v for k, v in params.items()})
    params_df = pd.DataFrame(parsed_params)
    candidate_aug_df = pd.concat([cand_df.reset_index(drop=True), params_df.reset_index(drop=True)], axis=1)
    metric_col = "Selection_Score" if "Selection_Score" in candidate_aug_df.columns else "CV_Mean_R2_Original"
    param_cols = list(params_df.columns)
    for param in param_cols:
        grouped = (
            candidate_aug_df
            .groupby(param, dropna=False)[metric_col]
            .agg(["count", "mean", "max", "min", "std"])
            .reset_index()
            .rename(columns={param: "Parameter_Value", "mean": "Mean_Selection_Score", "max": "Max_Selection_Score", "min": "Min_Selection_Score", "std": "Std_Selection_Score"})
        )
        grouped["Model"] = model_name
        grouped["Parameter"] = param
        grouped["Metric_Used"] = metric_col
        grouped["Best_Value_For_Parameter"] = grouped.loc[grouped["Mean_Selection_Score"].idxmax(), "Parameter_Value"] if len(grouped) else np.nan
        for _, grow in grouped.iterrows():
            parameter_effect_rows.append(grow.to_dict())
        if len(grouped) > 1:
            parameter_summary_rows.append({
                "Model": model_name,
                "Parameter": param,
                "Metric_Used": metric_col,
                "Best_Value_By_Mean": grouped.loc[grouped["Mean_Selection_Score"].idxmax(), "Parameter_Value"],
                "Worst_Value_By_Mean": grouped.loc[grouped["Mean_Selection_Score"].idxmin(), "Parameter_Value"],
                "Mean_Score_Range": grouped["Mean_Selection_Score"].max() - grouped["Mean_Selection_Score"].min(),
                "Max_Score_Range": grouped["Max_Selection_Score"].max() - grouped["Max_Selection_Score"].min(),
                "Candidate_File": str(path.name)
            })

parameter_effect_by_value_df = pd.DataFrame(parameter_effect_rows)
parameter_effect_summary_df = pd.DataFrame(parameter_summary_rows)
if len(parameter_effect_summary_df) > 0:
    parameter_effect_summary_df = parameter_effect_summary_df.sort_values(["Model", "Mean_Score_Range"], ascending=[True, False])

save_table(parameter_effect_by_value_df, "33a_tuning_parameter_effect_by_value.csv")
save_table(parameter_effect_summary_df, "33b_tuning_parameter_effect_summary.csv")

explanation_lines = []
explanation_lines.append("Interpretation summary of dominant and non-universal preprocessing")
explanation_lines.append("==========================================================")
explanation_lines.append("1. Dominant means frequently appearing in the best configurations of several models; it does not mean always optimal for every model.")
explanation_lines.append("2. If the delta in 31c is negative, the source model preprocessing is worse than the target model own best preprocessing.")
explanation_lines.append("3. Feature-outlier handling is often dominant because numeric features contain extreme values; capping can limit the effect of extremes before model learning.")
explanation_lines.append("4. Scaling is not universal: SVR strongly depends on scale because it is distance/kernel-based, whereas tree models are relatively more scale-robust but can still change due to imputation and outlier distributions.")
explanation_lines.append("5. The imputer is not universal: constant_zero can signal 'not available' to tree models, but in linear/kernel models it can alter feature geometry and coefficients differently.")
explanation_lines.append("6. Hyperparameter values also interact with preprocessing results. For example, max_depth, min_samples_leaf, C, epsilon, and learning_rate can make a model more complex or more regularized depending on the data shape after preprocessing.")
save_text("\n".join(explanation_lines), "32_preprocessing_dominance_interpretation_note.txt")

print("Frequency of dominant preprocessing methods:")
display(dominant_method_frequency_df)
print("Effect of dominant methods per model:")
display(dominant_method_effect_by_model_df.head(60))
print("Effect of tuning-parameter values:")
display(parameter_effect_summary_df.head(30))


# %% [cell 35]
# ============================================================
# CELL 13B - SUMMARY OF MODEL COUNTS THAT IMPROVED/DECREASED:
# BASELINE vs PREPROCESSING vs TUNING
# ============================================================
# Tujuan:
# Purpose:
# Question:
# "Do preprocessing and hyperparameter tuning mostly improve or decrease model performance compared with baseline?"

EPS_IMPROVEMENT = 1e-9

metric_directions = {
    "CV_Mean_R2_Original": "higher_better",
    "Test_R2_Original": "higher_better",
    "CV_Mean_R2_Transformed": "higher_better",
    "R2_Test_Transformed": "higher_better",
    "Test_RMSE_Original": "lower_better",
    "Test_MAE_Original": "lower_better"
}

metric_display_names = {
    "CV_Mean_R2_Original": "CV R2 Original",
    "Test_R2_Original": "Holdout R2 Original",
    "CV_Mean_R2_Transformed": "CV R2 Transformed",
    "R2_Test_Transformed": "Holdout R2 Transformed",
    "Test_RMSE_Original": "RMSE Original",
    "Test_MAE_Original": "MAE Original"
}

baseline_compare_df = (
    baseline_df[baseline_df["Stage"] == "01_baseline_default"]
    .copy()
    .sort_values(PRIMARY_SELECTION_METRIC, ascending=False)
    .groupby("Model", as_index=False)
    .head(1)
)

preprocessing_compare_df = best_preprocessing_per_model.copy()
tuning_compare_df = tuning_df.copy()

baseline_compare_df["Comparison_Stage"] = "Baseline"
preprocessing_compare_df["Comparison_Stage"] = "Combined Preprocessing"
tuning_compare_df["Comparison_Stage"] = "Hyperparameter Tuning"

needed_cols = [
    "Model", "Comparison_Stage", "Stage", "Tuned",
    "Imputer_Mode", "Scaler_Type", "Target_Transform",
    "Target_Outlier_Mode", "Feature_Outlier_Mode"
] + list(metric_directions.keys())

baseline_compare_df = baseline_compare_df[[col for col in needed_cols if col in baseline_compare_df.columns]]
preprocessing_compare_df = preprocessing_compare_df[[col for col in needed_cols if col in preprocessing_compare_df.columns]]
tuning_compare_df = tuning_compare_df[[col for col in needed_cols if col in tuning_compare_df.columns]]

stage_performance_df = pd.concat(
    [baseline_compare_df, preprocessing_compare_df, tuning_compare_df],
    ignore_index=True,
    sort=False
)

print("=" * 120)
print("BASELINE, PREPROCESSING, AND TUNING PERFORMANCE DATA")
print("=" * 120)
display(stage_performance_df.sort_values(["Model", "Comparison_Stage"]))

save_table(
    stage_performance_df,
    "18a_stage_performance_baseline_preprocessing_tuning.csv"
)

def classify_performance_change(delta, direction, eps=EPS_IMPROVEMENT):
    if pd.isna(delta):
        return "Not available"

    if direction == "higher_better":
        if delta > eps:
            return "Improved"
        elif delta < -eps:
            return "Decreased"
        else:
            return "Unchanged"

    if direction == "lower_better":
        if delta < -eps:
            return "Improved"
        elif delta > eps:
            return "Decreased"
        else:
            return "Unchanged"

    return "Not available"


def calculate_stage_delta(reference_df, comparison_df, reference_label, comparison_label):
    rows = []

    reference_by_model = reference_df.set_index("Model")
    comparison_by_model = comparison_df.set_index("Model")

    common_models = sorted(set(reference_by_model.index).intersection(set(comparison_by_model.index)))

    for model_name in common_models:
        ref_row = reference_by_model.loc[model_name]
        comp_row = comparison_by_model.loc[model_name]

        if isinstance(ref_row, pd.DataFrame):
            ref_row = ref_row.iloc[0]
        if isinstance(comp_row, pd.DataFrame):
            comp_row = comp_row.iloc[0]

        result_row = {
            "Model": model_name,
            "Reference_Stage": reference_label,
            "Comparison_Stage": comparison_label
        }

        for metric, direction in metric_directions.items():
            ref_value = ref_row.get(metric, np.nan)
            comp_value = comp_row.get(metric, np.nan)
            delta = comp_value - ref_value

            result_row[f"{metric}_Reference"] = ref_value
            result_row[f"{metric}_Comparison"] = comp_value
            result_row[f"{metric}_Delta"] = delta
            result_row[f"{metric}_Status"] = classify_performance_change(delta, direction)

        rows.append(result_row)

    return pd.DataFrame(rows)


delta_baseline_to_preprocessing_df = calculate_stage_delta(
    reference_df=baseline_compare_df,
    comparison_df=preprocessing_compare_df,
    reference_label="Baseline",
    comparison_label="Combined Preprocessing"
)

delta_baseline_to_tuning_df = calculate_stage_delta(
    reference_df=baseline_compare_df,
    comparison_df=tuning_compare_df,
    reference_label="Baseline",
    comparison_label="Hyperparameter Tuning"
)

delta_preprocessing_to_tuning_df = calculate_stage_delta(
    reference_df=preprocessing_compare_df,
    comparison_df=tuning_compare_df,
    reference_label="Combined Preprocessing",
    comparison_label="Hyperparameter Tuning"
)

performance_delta_df = pd.concat(
    [
        delta_baseline_to_preprocessing_df,
        delta_baseline_to_tuning_df,
        delta_preprocessing_to_tuning_df
    ],
    ignore_index=True,
    sort=False
)

print("\n" + "=" * 120)
print("DELTA PERFORMA PER MODEL")
print("=" * 120)

main_delta_cols = [
    "Model", "Reference_Stage", "Comparison_Stage",

    "CV_Mean_R2_Original_Reference",
    "CV_Mean_R2_Original_Comparison",
    "CV_Mean_R2_Original_Delta",
    "CV_Mean_R2_Original_Status",

    "Test_R2_Original_Reference",
    "Test_R2_Original_Comparison",
    "Test_R2_Original_Delta",
    "Test_R2_Original_Status",

    "CV_Mean_R2_Transformed_Reference",
    "CV_Mean_R2_Transformed_Comparison",
    "CV_Mean_R2_Transformed_Delta",
    "CV_Mean_R2_Transformed_Status",

    "R2_Test_Transformed_Reference",
    "R2_Test_Transformed_Comparison",
    "R2_Test_Transformed_Delta",
    "R2_Test_Transformed_Status",

    "Test_RMSE_Original_Reference",
    "Test_RMSE_Original_Comparison",
    "Test_RMSE_Original_Delta",
    "Test_RMSE_Original_Status",

    "Test_MAE_Original_Reference",
    "Test_MAE_Original_Comparison",
    "Test_MAE_Original_Delta",
    "Test_MAE_Original_Status"
]

available_main_delta_cols = [col for col in main_delta_cols if col in performance_delta_df.columns]

display(performance_delta_df[available_main_delta_cols])

save_table(
    performance_delta_df,
    "18b_performance_delta_per_model_baseline_preprocessing_tuning.csv"
)

summary_rows = []

for comparison_name, group_df in performance_delta_df.groupby(["Reference_Stage", "Comparison_Stage"]):
    reference_stage, comparison_stage = comparison_name

    for metric, direction in metric_directions.items():
        status_col = f"{metric}_Status"

        if status_col not in group_df.columns:
            continue

        status_counts = group_df[status_col].value_counts(dropna=False)

        total_model = len(group_df)
        improved_count = int(status_counts.get("Improved", 0))
        decreased_count = int(status_counts.get("Decreased", 0))
        unchanged_count = int(status_counts.get("Unchanged", 0))
        not_available_count = int(status_counts.get("Not available", 0))

        if improved_count > decreased_count:
            dominant_effect = "Dominant improvement"
        elif decreased_count > improved_count:
            dominant_effect = "Dominant decrease"
        elif improved_count == decreased_count and improved_count > 0:
            dominant_effect = "Balanced improvement and decrease"
        else:
            dominant_effect = "Dominant unchanged / no change"

        summary_rows.append({
            "Reference_Stage": reference_stage,
            "Comparison_Stage": comparison_stage,
            "Metric": metric,
            "Metric_Display": metric_display_names.get(metric, metric),
            "Metric_Direction": direction,
            "Total_Model": total_model,
            "Number_Improved": improved_count,
            "Number_Decreased": decreased_count,
            "Number_Unchanged": unchanged_count,
            "Number_Not_Available": not_available_count,
            "Percent_Improved_%": improved_count / total_model * 100 if total_model > 0 else np.nan,
            "Percent_Decreased_%": decreased_count / total_model * 100 if total_model > 0 else np.nan,
            "Percent_Unchanged_%": unchanged_count / total_model * 100 if total_model > 0 else np.nan,
            "Dominant_Conclusion": dominant_effect
        })

performance_change_summary_df = pd.DataFrame(summary_rows)

print("\n" + "=" * 120)
print("SUMMARY OF THE NUMBER OF MODELS THAT IMPROVED/DECREASED/REMAINED UNCHANGED")
print("=" * 120)

display(performance_change_summary_df)

save_table(
    performance_change_summary_df,
    "18c_summary_count_model_improved_decreased_unchanged.csv"
)

article_focus_metrics = [
    "CV_Mean_R2_Original",
    "Test_R2_Original",
    "CV_Mean_R2_Transformed",
    "R2_Test_Transformed",
    "Test_RMSE_Original",
    "Test_MAE_Original"
]

article_summary_df = performance_change_summary_df[
    performance_change_summary_df["Metric"].isin(article_focus_metrics)
].copy()

article_summary_df = article_summary_df.sort_values(
    ["Reference_Stage", "Comparison_Stage", "Metric"]
).reset_index(drop=True)

print("\n" + "=" * 120)
print("MANUSCRIPT-READY SUMMARY FOR DISCUSSION")
print("=" * 120)

display(article_summary_df[[
    "Reference_Stage",
    "Comparison_Stage",
    "Metric_Display",
    "Total_Model",
    "Number_Improved",
    "Number_Decreased",
    "Number_Unchanged",
    "Percent_Improved_%",
    "Percent_Decreased_%",
    "Dominant_Conclusion"
]])

save_table(
    article_summary_df,
    "18d_article_summary_preprocessing_tuning_effect_direction.csv"
)

def print_dominant_conclusion(summary_df, reference_stage, comparison_stage, metric):
    row = summary_df[
        (summary_df["Reference_Stage"] == reference_stage) &
        (summary_df["Comparison_Stage"] == comparison_stage) &
        (summary_df["Metric"] == metric)
    ]

    if len(row) == 0:
        print(f"- {reference_stage} -> {comparison_stage} for {metric}: data are not available.")
        return

    row = row.iloc[0]

    print(
        f"- {reference_stage} -> {comparison_stage} based on {row['Metric_Display']}: "
        f"{row['Number_Improved']} models improved, "
        f"{row['Number_Decreased']} models decreased, "
        f"{row['Number_Unchanged']} models unchanged "
        f"({row['Dominant_Conclusion']})."
    )

print("\nAUTOMATIC CONCLUSION:")

for ref_stage, comp_stage in [
    ("Baseline", "Combined Preprocessing"),
    ("Baseline", "Hyperparameter Tuning"),
    ("Combined Preprocessing", "Hyperparameter Tuning")
]:
    print_dominant_conclusion(performance_change_summary_df, ref_stage, comp_stage, "CV_Mean_R2_Original")
    print_dominant_conclusion(performance_change_summary_df, ref_stage, comp_stage, "Test_R2_Original")


# %% [cell 36]
# ============================================================
# CELL 13C - PUBLICATION MODEL FAMILY COMPARISON VS RANDOM FOREST
# ============================================================
# Purpose:
# Compare the final selected model families against the Random Forest reference used in the manuscript.

PUBLICATION_MODEL_GROUPS = {
    "MLR": "Linear/kernel reference models",
    "Ridge Regression": "Linear/kernel reference models",
    "SVR": "Linear/kernel reference models",
    "Random Forest Regression": "Main Random Forest comparator",
    "XGBoost": "Additional ensemble models",
    "Extra Trees Regression": "Additional ensemble models",
}

publication_model_comparison_df = best_final_per_model.copy()
publication_model_comparison_df["Model_Group"] = publication_model_comparison_df["Model"].map(PUBLICATION_MODEL_GROUPS).fillna("Additional ensemble models")

publication_model_comparison_df = publication_model_comparison_df.sort_values(
    PRIMARY_SELECTION_METRIC,
    ascending=False
).reset_index(drop=True)

publication_model_comparison_df["Rank_Overall"] = np.arange(1, len(publication_model_comparison_df) + 1)

print("=" * 120)
print("PUBLICATION MODEL FAMILY COMPARISON VS RANDOM FOREST")
print("=" * 120)
print("Selection metric:", PRIMARY_SELECTION_METRIC)

display(publication_model_comparison_df[[
    "Rank_Overall",
    "Model_Group",
    "Model",
    "Stage",
    "Tuned",
    "CV_Mean_R2_Original",
    "Test_R2_Original",
    "CV_Mean_R2_Transformed",
    "R2_Test_Transformed",
    "Test_RMSE_Original",
    "Test_MAE_Original",
    "Imputer_Mode",
    "Scaler_Type",
    "Target_Transform",
    "Target_Outlier_Mode",
    "Feature_Outlier_Mode"
]])

save_table(
    publication_model_comparison_df,
    "18e_publication_model_family_vs_random_forest_comparison.csv"
)

reference_models_only_df = publication_model_comparison_df[
    publication_model_comparison_df["Model_Group"] == "Linear/kernel reference models"
].copy()

rf_only_df = publication_model_comparison_df[
    publication_model_comparison_df["Model"] == "Random Forest Regression"
].copy()

best_reference_model = reference_models_only_df.sort_values(
    PRIMARY_SELECTION_METRIC,
    ascending=False
).iloc[0]

print("\nBEST LINEAR/KERNEL REFERENCE MODEL:")
display(pd.DataFrame([best_reference_model]))

if len(rf_only_df) > 0:
    rf_row = rf_only_df.iloc[0]

    reference_vs_rf_summary_df = pd.DataFrame([{
        "Best_Reference_Model": best_reference_model["Model"],
        "Best_Reference_CV_R2_Original": best_reference_model["CV_Mean_R2_Original"],
        "Best_Reference_Holdout_R2_Original": best_reference_model["Test_R2_Original"],
        "Best_Reference_RMSE_Original": best_reference_model["Test_RMSE_Original"],
        "Best_Reference_MAE_Original": best_reference_model["Test_MAE_Original"],

        "Random_Forest_CV_R2_Original": rf_row["CV_Mean_R2_Original"],
        "Random_Forest_Holdout_R2_Original": rf_row["Test_R2_Original"],
        "Random_Forest_RMSE_Original": rf_row["Test_RMSE_Original"],
        "Random_Forest_MAE_Original": rf_row["Test_MAE_Original"],

        "Delta_CV_R2_Original_RF_minus_BestReference": rf_row["CV_Mean_R2_Original"] - best_reference_model["CV_Mean_R2_Original"],
        "Delta_Holdout_R2_Original_RF_minus_BestReference": rf_row["Test_R2_Original"] - best_reference_model["Test_R2_Original"],
        "Delta_RMSE_Original_RF_minus_BestReference": rf_row["Test_RMSE_Original"] - best_reference_model["Test_RMSE_Original"],
        "Delta_MAE_Original_RF_minus_BestReference": rf_row["Test_MAE_Original"] - best_reference_model["Test_MAE_Original"]
    }])

    print("\nSUMMARY OF BEST REFERENCE MODEL VS RANDOM FOREST:")
    display(reference_vs_rf_summary_df)

    save_table(
        reference_vs_rf_summary_df,
        "18f_best_reference_model_vs_random_forest_summary.csv"
    )

    if rf_row[PRIMARY_SELECTION_METRIC] > best_reference_model[PRIMARY_SELECTION_METRIC]:
        print("\nAutomatic conclusion: Random Forest outperforms the best linear/kernel reference model based on", PRIMARY_SELECTION_METRIC)
    elif rf_row[PRIMARY_SELECTION_METRIC] < best_reference_model[PRIMARY_SELECTION_METRIC]:
        print("\nAutomatic conclusion: The best linear/kernel reference model outperforms Random Forest based on", PRIMARY_SELECTION_METRIC)
    else:
        print("\nAutomatic conclusion: Random Forest and the best linear/kernel reference model are equivalent based on", PRIMARY_SELECTION_METRIC)
else:
    print("\nRandom Forest Regression was not found in final results.")


# %% [cell 37]
# ============================================================
# CELL 14 - VISUALISASI PERBANDINGAN TRAIN-TEST SPLIT
# ============================================================

def plot_bar_by_stage_model(df, metric, title, filename):
    plot_df = df.copy()
    labels = plot_df["Stage"] + "\n" + plot_df["Model"]
    values = plot_df[metric].values

    plt.figure(figsize=(max(10, len(plot_df) * 0.55), 5))
    plt.bar(range(len(plot_df)), values)
    plt.xticks(range(len(plot_df)), labels, rotation=75, ha="right")
    plt.ylabel(metric)
    plt.title(title)
    plt.tight_layout()
    path = PLOT_DIR / filename
    plt.savefig(path, dpi=180)
    plt.show()
    return path

plot_bar_by_stage_model(
    main_comparison_df.sort_values(["Model", "Stage"]),
    metric="Test_R2_Original",
    title="Perbandingan R2 Holdout Train-Test Split: Baseline vs Preprocessing vs Tuning",
    filename="comparison_holdout_test_r2.png"
)

plot_bar_by_stage_model(
    main_comparison_df.sort_values(["Model", "Stage"]),
    metric="Test_RMSE_Original",
    title="Perbandingan RMSE Holdout Train-Test Split: Baseline vs Preprocessing vs Tuning",
    filename="comparison_holdout_test_rmse.png"
)

plot_bar_by_stage_model(
    main_comparison_df.sort_values(["Model", "Stage"]),
    metric="Test_MAE_Original",
    title="Perbandingan MAE Holdout Train-Test Split: Baseline vs Preprocessing vs Tuning",
    filename="comparison_holdout_test_mae.png"
)

# %% [markdown]
# ## Grafik tambahan: Baseline vs Combined Preprocessing vs Hyperparameter Tuning
#
# The following cell adds an R² comparison between cross-validation and holdout on both original and transformed scales. This figure clarifies performance changes from baseline to best preprocessing and hyperparameter tuning.


# %% [cell 39]
# ============================================================
# CELL 14B - R2 CV vs HOLDOUT FIGURE: BASELINE vs PREPROCESSING vs TUNING
# ============================================================
# This figure compares R2 performance for:
# 1) original scale: CV validation R2 vs Holdout R2
# 2) transformed scale: CV validation R2 vs Holdout R2
# Compared stages: Baseline, Combined Preprocessing, and Hyperparameter Tuning.

stage_frames = []

# Complete-case default baseline without preprocessing.
baseline_stage_df = baseline_df[baseline_df["Stage"] == "01_baseline_default"].copy()
baseline_stage_df["Comparison_Stage"] = "Baseline"
baseline_stage_df["Comparison_Order"] = 1
stage_frames.append(baseline_stage_df)

# Best combined preprocessing per model.
combined_stage_df = best_preprocessing_per_model.copy()
combined_stage_df["Comparison_Stage"] = "Combined Preprocessing"
combined_stage_df["Comparison_Order"] = 2
stage_frames.append(combined_stage_df)

# Hyperparameter tuning on the best preprocessing configuration.
tuning_stage_df = tuning_df.copy()
tuning_stage_df["Comparison_Stage"] = "Hyperparameter Tuning"
tuning_stage_df["Comparison_Order"] = 3
stage_frames.append(tuning_stage_df)

stage_comparison_df = pd.concat(stage_frames, ignore_index=True, sort=False)
stage_comparison_df = stage_comparison_df.sort_values(
    ["Model", "Comparison_Order"]
).reset_index(drop=True)

stage_r2_cv_holdout_df = stage_comparison_df[[
    "Comparison_Stage", "Comparison_Order", "Stage", "Model", "Tuned",

    # Original scale.
    "CV_Mean_Train_R2_Original",
    "CV_Mean_R2_Original",
    "CV_Mean_Train_Val_Gap_R2_Original",
    "Train_R2_Original",
    "Test_R2_Original",
    "Train_Test_Gap_R2_Original",

    # Transformed scale.
    "CV_Mean_Train_R2_Transformed",
    "CV_Mean_R2_Transformed",
    "CV_Mean_Train_Val_Gap_R2_Transformed",
    "R2_Train_Transformed",
    "R2_Test_Transformed",
    "Train_Test_Gap_R2_Transformed",

    "Test_RMSE_Original", "Test_MAE_Original",
    "Imputer_Mode", "Scaler_Type", "Target_Transform",
    "Target_Outlier_Mode", "Feature_Outlier_Mode"
]].rename(columns={
    "CV_Mean_Train_R2_Original": "CV_Train_R2_Original",
    "CV_Mean_R2_Original": "CV_Val_R2_Original",
    "CV_Mean_Train_Val_Gap_R2_Original": "CV_Train_Val_Gap_R2_Original",
    "Train_R2_Original": "Train_R2_Original",
    "Test_R2_Original": "Holdout_R2_Original",
    "Train_Test_Gap_R2_Original": "Train_Holdout_Gap_R2_Original",

    "CV_Mean_Train_R2_Transformed": "CV_Train_R2_Transformed",
    "CV_Mean_R2_Transformed": "CV_Val_R2_Transformed",
    "CV_Mean_Train_Val_Gap_R2_Transformed": "CV_Train_Val_Gap_R2_Transformed",
    "R2_Train_Transformed": "Train_R2_Transformed",
    "R2_Test_Transformed": "Holdout_R2_Transformed",
    "Train_Test_Gap_R2_Transformed": "Train_Holdout_Gap_R2_Transformed",

    "Test_RMSE_Original": "RMSE_Original",
    "Test_MAE_Original": "MAE_Original"
})

display(stage_r2_cv_holdout_df)
save_table(stage_r2_cv_holdout_df, "21_r2_cv_holdout_baseline_preprocessing_tuning.csv")

def plot_cv_vs_holdout_r2(plot_df, cv_col, holdout_col, title, ylabel, filename):
    plot_df = plot_df.sort_values(["Model", "Comparison_Order"]).reset_index(drop=True)
    labels = plot_df["Model"] + "\n" + plot_df["Comparison_Stage"]
    x = np.arange(len(plot_df))
    width = 0.38

    plt.figure(figsize=(max(12, len(plot_df) * 0.65), 6))
    plt.bar(x - width/2, plot_df[cv_col], width, label="CV Validation")
    plt.bar(x + width/2, plot_df[holdout_col], width, label="Holdout Test")
    plt.axhline(0, linewidth=0.8)
    plt.xticks(x, labels, rotation=75, ha="right")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    path = PLOT_DIR / filename
    plt.savefig(path, dpi=180)
    plt.show()
    print("Saved:", path)

plot_cv_vs_holdout_r2(
    stage_r2_cv_holdout_df,
    cv_col="CV_Val_R2_Original",
    holdout_col="Holdout_R2_Original",
    title="Original R2 Comparison: CV Validation vs Holdout for Baseline, Preprocessing, and Tuning",
    ylabel="R2 Original",
    filename="r2_original_cv_vs_holdout_baseline_preprocessing_tuning.png"
)

plot_cv_vs_holdout_r2(
    stage_r2_cv_holdout_df,
    cv_col="CV_Val_R2_Transformed",
    holdout_col="Holdout_R2_Transformed",
    title="Transformed R2 Comparison: CV Validation vs Holdout for Baseline, Preprocessing, and Tuning",
    ylabel="R2 Transformed",
    filename="r2_transformed_cv_vs_holdout_baseline_preprocessing_tuning.png"
)


# %% [cell 40]
# ============================================================
# CELL 15 - K-FOLD CROSS VALIDATION FINAL SEBAGAI PEMBANDING TRAIN-TEST SPLIT
# ============================================================
# Final K-Fold is run on training data for the final model of each algorithm.
# The test data are still used only for final holdout evaluation to match the study workflow.

kfold_final_rows = []
kfold_final_fold_rows = []

for _, row in best_final_per_model.iterrows():
    model_name = row["Model"]
    params = {}
    try:
        params = json.loads(row.get("Model_Params", "{}")) if isinstance(row.get("Model_Params", "{}"), str) else {}
    except Exception:
        params = {}

    print("\n" + "=" * 100)
    print("K-FOLD FINAL | Model:", model_name)
    print("=" * 100)

    for k in K_VALUES_FINAL:
        if len(X_all) <= k:
            continue

        cv_summary, fold_df = kfold_cv_score_for_config(
            model_name=model_name,
            X_df=X_train,
            y_series=y_train,
            imputer_mode=row["Imputer_Mode"],
            scaler_type=row["Scaler_Type"],
            target_transform=row["Target_Transform"],
            target_outlier_mode=row["Target_Outlier_Mode"],
            feature_outlier_mode=row["Feature_Outlier_Mode"],
            model_params=params,
            n_splits=k
        )

        out_row = {
            "Model": model_name,
            "K": k,
            "Holdout_Test_R2_Original": row["Test_R2_Original"],
            "Holdout_Test_RMSE_Original": row["Test_RMSE_Original"],
            "Holdout_Test_MAE_Original": row["Test_MAE_Original"],
            **cv_summary,
            "Imputer_Mode": row["Imputer_Mode"],
            "Scaler_Type": row["Scaler_Type"],
            "Target_Transform": row["Target_Transform"],
            "Target_Outlier_Mode": row["Target_Outlier_Mode"],
            "Feature_Outlier_Mode": row["Feature_Outlier_Mode"]
        }
        out_row["Delta_KFoldMean_vs_Holdout_R2_Original"] = out_row["CV_Mean_R2_Original"] - out_row["Holdout_Test_R2_Original"]
        out_row["Delta_KFoldMean_vs_Holdout_RMSE_Original"] = out_row["CV_Mean_RMSE_Original"] - out_row["Holdout_Test_RMSE_Original"]
        out_row["Delta_KFoldMean_vs_Holdout_MAE_Original"] = out_row["CV_Mean_MAE_Original"] - out_row["Holdout_Test_MAE_Original"]

        kfold_final_rows.append(out_row)

        fold_df["Model"] = model_name
        fold_df["K"] = k
        kfold_final_fold_rows.append(fold_df)

        print(f"K={k} | CV Mean R2 original={out_row['CV_Mean_R2_Original']:.4f} ± {out_row['CV_Std_R2_Original']:.4f}")
        print(f"     | Holdout Test R2 original={out_row['Holdout_Test_R2_Original']:.4f}")

kfold_final_df = pd.DataFrame(kfold_final_rows)
kfold_final_folds_df = pd.concat(kfold_final_fold_rows, ignore_index=True)

display(kfold_final_df[[
    "Model", "K",

    # K-Fold training and validation.
    "CV_Mean_Train_R2_Original",
    "CV_Std_Train_R2_Original",
    "CV_Mean_R2_Original",
    "CV_Std_R2_Original",
    "CV_Mean_Train_Val_Gap_R2_Original",

    # Holdout.
    "Holdout_Test_R2_Original",
    "Delta_KFoldMean_vs_Holdout_R2_Original",

    "Holdout_Test_RMSE_Original", "CV_Mean_RMSE_Original", "CV_Std_RMSE_Original",
    "Holdout_Test_MAE_Original", "CV_Mean_MAE_Original", "CV_Std_MAE_Original"
]].sort_values(["Model", "K"]))

save_table(kfold_final_df, "18_final_kfold_vs_holdout_summary.csv")
save_table(kfold_final_folds_df, "19_final_kfold_fold_details.csv")

plt.figure(figsize=(10, 5))
labels = kfold_final_df["Model"] + " K=" + kfold_final_df["K"].astype(str)
plt.bar(range(len(kfold_final_df)), kfold_final_df["CV_Mean_R2_Original"])
plt.errorbar(
    range(len(kfold_final_df)),
    kfold_final_df["CV_Mean_R2_Original"],
    yerr=kfold_final_df["CV_Std_R2_Original"],
    fmt="none",
    capsize=4
)
plt.xticks(range(len(kfold_final_df)), labels, rotation=60, ha="right")
plt.ylabel("Mean R2 Original")
plt.title("K-Fold Cross Validation Final per Model")
plt.tight_layout()
path = PLOT_DIR / "final_kfold_r2_mean_std.png"
plt.savefig(path, dpi=180)
plt.show()

# %% [markdown]
# ## Additional figure: final K-Fold vs holdout for final models
#
# The following cell compares final K-Fold R² with holdout R² for each final algorithm on original and transformed scales.


# %% [cell 42]
# ============================================================
# CELL 15B - FINAL K-FOLD vs HOLDOUT FIGURE FOR FINAL MODELS
# ============================================================
# This cell is run after CELL 15 because it uses kfold_final_df.

final_r2_df = kfold_final_df.merge(
    best_final_per_model[[
        "Model",
        "R2_Test_Transformed",
        "R2_Train_Transformed",
        "Train_Test_Gap_R2_Transformed"
    ]],
    on="Model",
    how="left"
)

final_r2_df = final_r2_df.assign(
    KFold_Train_R2_Original=final_r2_df["CV_Mean_Train_R2_Original"],
    KFold_Val_R2_Original=final_r2_df["CV_Mean_R2_Original"],
    KFold_Train_Val_Gap_R2_Original=final_r2_df["CV_Mean_Train_Val_Gap_R2_Original"],
    Holdout_R2_Original=final_r2_df["Holdout_Test_R2_Original"],

    KFold_Train_R2_Transformed=final_r2_df["CV_Mean_Train_R2_Transformed"],
    KFold_Val_R2_Transformed=final_r2_df["CV_Mean_R2_Transformed"],
    KFold_Train_Val_Gap_R2_Transformed=final_r2_df["CV_Mean_Train_Val_Gap_R2_Transformed"],
    Holdout_R2_Transformed=final_r2_df["R2_Test_Transformed"]
)

final_r2_df = final_r2_df[[
    "Model", "K",

    "KFold_Train_R2_Original",
    "KFold_Val_R2_Original",
    "KFold_Train_Val_Gap_R2_Original",
    "Holdout_R2_Original",

    "KFold_Train_R2_Transformed",
    "KFold_Val_R2_Transformed",
    "KFold_Train_Val_Gap_R2_Transformed",
    "Holdout_R2_Transformed",

    "Holdout_Test_RMSE_Original",
    "CV_Mean_RMSE_Original",
    "Holdout_Test_MAE_Original",
    "CV_Mean_MAE_Original"
]].sort_values(["Model", "K"]).reset_index(drop=True)

display(final_r2_df)
save_table(final_r2_df, "22_final_kfold_vs_holdout_r2_original_transformed.csv")

def plot_final_kfold_vs_holdout(plot_df, cv_col, holdout_col, title, ylabel, filename):
    plot_df = plot_df.sort_values(["Model", "K"]).reset_index(drop=True)
    labels = plot_df["Model"] + "\nK=" + plot_df["K"].astype(str)
    x = np.arange(len(plot_df))
    width = 0.38

    plt.figure(figsize=(max(10, len(plot_df) * 0.8), 5.5))
    plt.bar(x - width/2, plot_df[cv_col], width, label="K-Fold Validation")
    plt.bar(x + width/2, plot_df[holdout_col], width, label="Holdout Test")
    plt.axhline(0, linewidth=0.8)
    plt.xticks(x, labels, rotation=60, ha="right")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    path = PLOT_DIR / filename
    plt.savefig(path, dpi=180)
    plt.show()
    print("Saved:", path)

plot_final_kfold_vs_holdout(
    final_r2_df,
    cv_col="KFold_Val_R2_Original",
    holdout_col="Holdout_R2_Original",
    title="Model Final: R2 Original K-Fold Validation vs Holdout",
    ylabel="R2 Original",
    filename="final_model_r2_original_kfold_vs_holdout.png"
)

plot_final_kfold_vs_holdout(
    final_r2_df,
    cv_col="KFold_Val_R2_Transformed",
    holdout_col="Holdout_R2_Transformed",
    title="Model Final: R2 Transformed K-Fold Validation vs Holdout",
    ylabel="R2 Transformed",
    filename="final_model_r2_transformed_kfold_vs_holdout.png"
)


# %% [cell 43]
# ============================================================
# CELL 16 - BEST MODEL DETAILS: ACTUAL VS PREDICTED VALUES AND RESIDUALS
# ============================================================

best_overall_model_name = best_overall["Model"]
best_overall_stage = best_overall["Stage"]

try:
    best_params = json.loads(best_overall.get("Model_Params", "{}")) if isinstance(best_overall.get("Model_Params", "{}"), str) else {}
except Exception:
    best_params = {}

best_row, best_pipe, best_pred_df = fit_evaluate_train_test(
    model_name=best_overall_model_name,
    scenario_name="best_overall_refit_holdout",
    X_train_df=X_train,
    y_train_series=y_train,
    X_test_df=X_test,
    y_test_series=y_test,
    imputer_mode=best_overall["Imputer_Mode"],
    scaler_type=best_overall["Scaler_Type"],
    target_transform=best_overall["Target_Transform"],
    target_outlier_mode=best_overall["Target_Outlier_Mode"],
    feature_outlier_mode=best_overall["Feature_Outlier_Mode"],
    model_params=best_params,
    tuned=bool(best_overall.get("Tuned", False)),
    stage="best_overall_refit"
)

print("BEST OVERALL MODEL BASED ON TRAINING CV")
display(pd.DataFrame([best_row]))

save_table(best_pred_df, "20_best_overall_holdout_predictions.csv")

plt.figure(figsize=(6, 6))
plt.scatter(best_pred_df[PRED_ACTUAL_COL], best_pred_df[PRED_PRED_COL], alpha=0.75)
min_v = min(best_pred_df[PRED_ACTUAL_COL].min(), best_pred_df[PRED_PRED_COL].min())
max_v = max(best_pred_df[PRED_ACTUAL_COL].max(), best_pred_df[PRED_PRED_COL].max())
plt.plot([min_v, max_v], [min_v, max_v], linestyle="--")
plt.xlabel(f"Actual {TARGET_DISPLAY_NAME} ({TARGET_UNIT})")
plt.ylabel(f"Predicted {TARGET_DISPLAY_NAME} ({TARGET_UNIT})")
plt.title(f"Actual vs Predicted - {best_overall_model_name}")
plt.tight_layout()
path = PLOT_DIR / "best_model_actual_vs_predicted.png"
plt.savefig(path, dpi=180)
plt.show()

plt.figure(figsize=(8, 4))
plt.scatter(best_pred_df[PRED_PRED_COL], best_pred_df["Residual_nm"], alpha=0.75)
plt.axhline(0, linestyle="--")
plt.xlabel(f"Predicted {TARGET_DISPLAY_NAME} ({TARGET_UNIT})")
plt.ylabel(f"Residual ({TARGET_UNIT})")
plt.title(f"Residual Plot - {best_overall_model_name}")
plt.tight_layout()
path = PLOT_DIR / "best_model_residual_plot.png"
plt.savefig(path, dpi=180)
plt.show()

# Save the best model.
Xtr_pkg = X_train[FEATURE_COLS].copy()
ytr_pkg = np.asarray(y_train, dtype=float)

Xtr_pkg, ytr_pkg, missing_pkg_info = apply_missing_drop_to_training(
    Xtr_pkg, ytr_pkg, best_overall["Imputer_Mode"]
)

keep_pkg, ytr_pkg_after_outlier, target_pkg_info = apply_target_outlier_to_training(
    ytr_pkg, best_overall["Target_Outlier_Mode"]
)

if len(keep_pkg) == len(Xtr_pkg) and not np.all(keep_pkg):
    Xtr_pkg = Xtr_pkg.loc[keep_pkg].copy()

Xtr_pkg, _, feature_pkg_info = apply_feature_outlier_fit_transform(
    Xtr_pkg, X_train[FEATURE_COLS].copy(), best_overall["Feature_Outlier_Mode"]
)

ytr_pkg_t = transform_target(ytr_pkg_after_outlier, best_overall["Target_Transform"])

best_pipe_for_save = build_full_pipeline(
    model_name=best_overall_model_name,
    imputer_mode=best_overall["Imputer_Mode"],
    scaler_type=best_overall["Scaler_Type"],
    n_rows_for_quantile=len(Xtr_pkg),
    selected_features=FEATURE_COLS
)

if best_params:
    best_pipe_for_save.set_params(**best_params)

best_pipe_for_save.fit(Xtr_pkg, ytr_pkg_t)

model_path = MODEL_DIR / "best_overall_model.joblib"

joblib.dump({
    "pipeline": best_pipe_for_save,
    "metadata": best_row,
    "feature_outlier_params": feature_pkg_info.get("Feature_Outlier_Params", {}),
    "missing_info": missing_pkg_info,
    "target_outlier_info": target_pkg_info,
    "feature_cols": FEATURE_COLS,
    "categorical_cols": CATEGORICAL_COLS,
    "numeric_cols": NUMERIC_COLS,
    "target_col": TARGET_COL,
    "target_display_name": TARGET_DISPLAY_NAME,
    "target_unit": TARGET_UNIT
}, model_path)

print("Best model saved:", model_path)
print("Feature-outlier parameters saved:", len(feature_pkg_info.get("Feature_Outlier_Params", {})))



# %% [markdown]
# ## Additional analysis: inverse-system concept / inverse design
#
# The main study model is a forward model: synthesis parameters are inputs and core size is the output. The following cell adds a **candidate-search-based inverse design** mode: the user enters a target size, then the system ranks synthesis-parameter combinations predicted to be closest to that size.
#
# This is not a single mathematical inversion because many parameter combinations can produce similar sizes. Outputs should be treated as laboratory-experiment candidates, not guaranteed final recipes.


# %% [cell 16A0]
# ============================================================
# CELL 16A0 - INVERSE DESIGN: INPUT TARGET SIZE, OUTPUT SYNTHESIS-PARAMETER CANDIDATES
# ============================================================
# Konsep:
# - Forward model: X(synthesis parameters) -> y(core size)
# - ML-based inverse design: target core size -> search for X candidates with the closest predictions
# - The default candidate pool comes from parameter combinations observed in the clean dataset.
# - For laboratory experiments, these results are used as an initial candidate list that requires validation.

INVERSE_DESIGN_DIR = OUTPUT_DIR / "inverse_design"
INVERSE_DESIGN_DIR.mkdir(parents=True, exist_ok=True)

# Modify this list according to the target sizes to be searched.
# Example: request parameter recommendations for 10 nm, 20 nm, and 50 nm targets.
INVERSE_DESIGN_TARGET_SIZES_NM = [10, 20, 30, 50, 100]
INVERSE_DESIGN_TOP_N = 50


def build_inverse_design_candidate_pool_from_observed_data():
    """Safe candidate pool: synthesis-parameter combinations already observed in the data."""
    pool = df_clean[FEATURE_COLS].drop_duplicates().reset_index(drop=True).copy()
    pool["Candidate_ID"] = np.arange(1, len(pool) + 1)
    return pool


def predict_core_size_for_candidate_pool(candidate_pool_df):
    """Predict core size for the candidate pool using the best overall fitted model."""
    X_candidate = candidate_pool_df[FEATURE_COLS].copy()

    # Apply feature-outlier bounds from the best model training data if available.
    feature_bounds = feature_pkg_info.get("Feature_Outlier_Params", {}) if "feature_pkg_info" in globals() else {}
    X_candidate_model = apply_feature_outlier_params(X_candidate, feature_bounds)

    pred_transformed = best_pipe_for_save.predict(X_candidate_model)
    pred_nm = inverse_transform_target(pred_transformed, best_overall["Target_Transform"])

    out = candidate_pool_df.copy()
    out["Predicted_Core_Size_nm"] = pred_nm
    return out


def rank_inverse_design_candidates(desired_core_size_nm, candidate_predictions_df, top_n=50, constraints=None):
    """
    Rank synthesis-parameter candidates based on prediction closeness to the target size.

    optional constraints in dict form, example:
    constraints = {
        "Process": ["Biological"],
        "Treatment": ["No"],
        "Temp_Celcius": (20, 90)
    }
    For categorical variables, use a list of allowed values.
    For numeric variables, use a tuple of (min, max).
    """
    df = candidate_predictions_df.copy()

    if constraints:
        for col, allowed in constraints.items():
            if col not in df.columns:
                continue
            if isinstance(allowed, tuple) and len(allowed) == 2:
                df = df[(pd.to_numeric(df[col], errors="coerce") >= allowed[0]) & (pd.to_numeric(df[col], errors="coerce") <= allowed[1])]
            elif isinstance(allowed, list):
                df = df[df[col].isin(allowed)]
            else:
                df = df[df[col] == allowed]

    df["Desired_Core_Size_nm"] = float(desired_core_size_nm)
    df["Absolute_Error_to_Desired_nm"] = (df["Predicted_Core_Size_nm"] - float(desired_core_size_nm)).abs()
    df["Relative_Error_to_Desired_%"] = df["Absolute_Error_to_Desired_nm"] / max(abs(float(desired_core_size_nm)), 1e-9) * 100
    df = df.sort_values(["Absolute_Error_to_Desired_nm", "Predicted_Core_Size_nm"]).reset_index(drop=True)
    df["Inverse_Design_Rank"] = np.arange(1, len(df) + 1)
    return df.head(top_n)


candidate_pool_df = build_inverse_design_candidate_pool_from_observed_data()
candidate_predictions_df = predict_core_size_for_candidate_pool(candidate_pool_df)

candidate_pool_path = INVERSE_DESIGN_DIR / "inverse_design_candidate_pool_observed_parameter_combinations.csv"
candidate_predictions_path = INVERSE_DESIGN_DIR / "inverse_design_candidate_pool_with_predictions.csv"
candidate_pool_df.to_csv(candidate_pool_path, index=False)
candidate_predictions_df.to_csv(candidate_predictions_path, index=False)

inverse_design_summary_rows = []
for desired_size in INVERSE_DESIGN_TARGET_SIZES_NM:
    ranked_df = rank_inverse_design_candidates(
        desired_core_size_nm=desired_size,
        candidate_predictions_df=candidate_predictions_df,
        top_n=INVERSE_DESIGN_TOP_N,
        constraints=None,
    )
    out_path = INVERSE_DESIGN_DIR / f"inverse_design_top_{INVERSE_DESIGN_TOP_N}_candidates_for_{str(desired_size).replace('.', '_')}_nm.csv"
    ranked_df.to_csv(out_path, index=False)
    best_candidate = ranked_df.iloc[0].to_dict() if len(ranked_df) else {}
    inverse_design_summary_rows.append({
        "Desired_Core_Size_nm": desired_size,
        "Top_N": INVERSE_DESIGN_TOP_N,
        "Best_Candidate_ID": best_candidate.get("Candidate_ID", np.nan),
        "Best_Predicted_Core_Size_nm": best_candidate.get("Predicted_Core_Size_nm", np.nan),
        "Best_Absolute_Error_nm": best_candidate.get("Absolute_Error_to_Desired_nm", np.nan),
        "Best_Relative_Error_%": best_candidate.get("Relative_Error_to_Desired_%", np.nan),
        "Output_File": str(out_path),
    })

inverse_design_summary_df = pd.DataFrame(inverse_design_summary_rows)
inverse_design_summary_path = INVERSE_DESIGN_DIR / "inverse_design_summary.csv"
inverse_design_summary_df.to_csv(inverse_design_summary_path, index=False)

inverse_design_readme = f"""
Inverse design / reverse-system concept
================================

The main study model is a forward model: synthesis parameters -> core size (nm).
The inverse system does not mean the model automatically knows one unique recipe from the target size.
This is a many-to-one problem: several parameter combinations can produce similar sizes.

The approach used in this code is candidate search:
1. Take synthesis-parameter combinations observed in the clean data as the candidate pool.
2. Predict core size for each candidate using the best model.
3. The user enters a target size, for example 20 nm.
4. The system ranks candidates based on the absolute difference between prediction and target.
5. Top candidates are used as initial laboratory-experiment recommendations.

Best model used:
- Model: {best_overall_model_name}
- Stage: {best_overall_stage}
- Preprocessing: imputer={best_overall['Imputer_Mode']}; scaler={best_overall['Scaler_Type']}; target_transform={best_overall['Target_Transform']}; target_outlier={best_overall['Target_Outlier_Mode']}; feature_outlier={best_overall['Feature_Outlier_Mode']}

Limitations:
- Recommendations are valid only within a data space similar to the training data.
- Candidates must be checked for chemical/laboratory feasibility.
- Predictions are not causal proof and do not guarantee that the actual size will be identical.
- For new experimental designs, the candidate pool should be extended with chemically reasonable numeric ranges and categories.
""".strip()

(INVERSE_DESIGN_DIR / "README_inverse_design.txt").write_text(inverse_design_readme + "\n", encoding="utf-8")
save_text(inverse_design_readme, "34_inverse_design_concept_note.txt")

print("Inverse design candidate pool:", candidate_pool_path)
print("Inverse design predictions:", candidate_predictions_path)
print("Inverse design summary:", inverse_design_summary_path)
display(inverse_design_summary_df)



# %% [cell 44]
# ============================================================
# CELL 16A - SHAP FEATURE-IMPORTANCE ANALYSIS
# ============================================================
# Purpose:
# 1. Compute feature ranking using mean absolute SHAP values.
# 2. Aggregate one-hot encoded SHAP values back to the original feature names.
# 3. Export SHAP tables and a publication-ready SHAP feature-importance figure.
#
# SHAP is used only for interpretation. This script does not perform SHAP feature-subset
# feature-selection retraining.

# ============================================================
# SHAP CONFIGURATION
# ============================================================

# To keep SHAP runtime manageable, only a subset of the training data is used.
# Increase this value for a more stable but slower SHAP computation.
SHAP_SAMPLE_SIZE = 300

# Tree-based source model priority for SHAP interpretation.
# Set FORCE_SHAP_SOURCE_MODEL to "Random Forest Regression" if the manuscript
# must force SHAP to follow the final Random Forest model.
FORCE_SHAP_SOURCE_MODEL = None

TREE_MODEL_PRIORITY = [
    "Random Forest Regression",
    "XGBoost",
    "Extra Trees Regression"
]

print("SHAP sample size:", SHAP_SAMPLE_SIZE)

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def parse_model_params_from_row(row):
    value = row.get("Model_Params", "{}")

    if isinstance(value, dict):
        return value

    if isinstance(value, str):
        try:
            return json.loads(value)
        except Exception:
            return {}

    return {}


def parse_selected_features_from_row(row):
    value = row.get("Selected_Features", None)

    if isinstance(value, list):
        return value

    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return parsed
        except Exception:
            pass

    return FEATURE_COLS


def select_shap_source_row(best_final_df):
    df = best_final_df.copy()

    if FORCE_SHAP_SOURCE_MODEL is not None:
        forced = df[df["Model"] == FORCE_SHAP_SOURCE_MODEL].copy()
        if len(forced) > 0:
            print("Forced SHAP source model:", FORCE_SHAP_SOURCE_MODEL)
            return forced.iloc[0]

        print("The forced SHAP source model was not found; reverting to automatic selection.")

    available_tree = df[df["Model"].isin(TREE_MODEL_PRIORITY)].copy()

    if len(available_tree) > 0:
        if PRIMARY_SELECTION_METRIC in available_tree.columns:
            available_tree = available_tree.sort_values(PRIMARY_SELECTION_METRIC, ascending=False)

        return available_tree.iloc[0]

    print("No tree-based model was found. SHAP uses best_overall.")
    return best_overall


def fit_pipeline_for_shap(row, selected_features):
    """
    Refit the pipeline on training data to compute SHAP.
    Preprocessing follows the final model configuration.
    SHAP is computed only on training data to avoid data leakage.
    """
    model_name = row["Model"]
    model_params = parse_model_params_from_row(row)

    imputer_mode = row["Imputer_Mode"]
    scaler_type = row["Scaler_Type"]
    target_transform = row["Target_Transform"]
    target_outlier_mode = row["Target_Outlier_Mode"]
    feature_outlier_mode = row["Feature_Outlier_Mode"]

    Xtr = X_train[selected_features].copy()
    ytr_original = np.asarray(y_train, dtype=float)

    # 1. Missing handling.
    if imputer_mode == "drop":
        train_mask = ~Xtr.isna().any(axis=1)
        Xtr = Xtr.loc[train_mask].copy()
        ytr_original = ytr_original[train_mask.values]
    else:
        Xtr, ytr_original, _ = apply_missing_drop_to_training(
            Xtr,
            ytr_original,
            imputer_mode
        )

    # 2. Target-outlier handling is applied only to training data.
    keep_mask, ytr_after_target_outlier, _ = apply_target_outlier_to_training(
        ytr_original,
        target_outlier_mode
    )

    if len(keep_mask) == len(Xtr) and not np.all(keep_mask):
        Xtr = Xtr.loc[keep_mask].copy()

    # 3. Feature-outlier threshold from training data.
    Xtr, _, _ = apply_feature_outlier_fit_transform(
        Xtr,
        Xtr.copy(),
        feature_outlier_mode
    )

    # 4. Target transform.
    ytr_transformed = transform_target(
        ytr_after_target_outlier,
        target_transform
    )

    # 5. Fit pipeline.
    pipe = build_full_pipeline(
        model_name=model_name,
        imputer_mode=imputer_mode,
        scaler_type=scaler_type,
        n_rows_for_quantile=len(Xtr),
        selected_features=selected_features
    )

    if model_params:
        pipe.set_params(**model_params)

    pipe.fit(Xtr, ytr_transformed)

    return pipe, Xtr, ytr_transformed


def get_original_feature_mapping(fitted_preprocessor, selected_features):
    """
    Create a mapping from preprocessed features to original features.
    Example:
    cat__NEW_Capping agent_plant -> NEW_Capping agent
    num__Temp_Celcius -> Temp_Celcius
    """
    transformed_feature_names = fitted_preprocessor.get_feature_names_out()

    selected_features_sorted = sorted(
        list(selected_features),
        key=len,
        reverse=True
    )

    original_map = []

    for transformed_name in transformed_feature_names:
        clean_name = transformed_name.split("__", 1)[-1]

        matched_col = None

        for original_col in selected_features_sorted:
            if clean_name == original_col or clean_name.startswith(original_col + "_"):
                matched_col = original_col
                break

        if matched_col is None:
            matched_col = clean_name

        original_map.append(matched_col)

    return list(transformed_feature_names), original_map


def compute_shap_ranking_original_features(source_row):
    """
    Compute mean absolute SHAP and aggregate one-hot results back to original features.
    """
    source_model_name = source_row["Model"]

    print("=" * 120)
    print("COMPUTING SHAP FEATURE IMPORTANCE")
    print("SHAP source model:", source_model_name)
    print("Source model target transform:", source_row["Target_Transform"])
    print("=" * 120)

    # SHAP is computed from all 14 input features so the interpretation covers the full manuscript feature space.
    selected_features_for_shap = FEATURE_COLS

    shap_pipe, Xtr_for_shap, _ = fit_pipeline_for_shap(
        source_row,
        selected_features=selected_features_for_shap
    )

    preprocessor = shap_pipe.named_steps["preprocessor"]
    model = shap_pipe.named_steps["model"]

    Xtr_transformed = preprocessor.transform(Xtr_for_shap)

    # Convert sparse output safely when needed.
    try:
        Xtr_transformed = Xtr_transformed.toarray()
    except Exception:
        pass

    transformed_feature_names, original_feature_map = get_original_feature_mapping(
        preprocessor,
        selected_features_for_shap
    )

    # Sample rows so SHAP runtime is not too long.
    n_sample = min(SHAP_SAMPLE_SIZE, Xtr_transformed.shape[0])
    rng = np.random.default_rng(RANDOM_STATE)
    sample_idx = rng.choice(
        np.arange(Xtr_transformed.shape[0]),
        size=n_sample,
        replace=False
    )

    X_shap = Xtr_transformed[sample_idx]

    tree_model_names = [
        "Random Forest Regression",
        "Extra Trees Regression",
        "XGBoost"
    ]

    start = time.time()

    if source_model_name in tree_model_names:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_shap)

        if isinstance(shap_values, list):
            shap_values = shap_values[0]

    else:
        # Fallback for non-tree models.
        background_size = min(100, Xtr_transformed.shape[0])
        background_idx = rng.choice(
            np.arange(Xtr_transformed.shape[0]),
            size=background_size,
            replace=False
        )
        background = Xtr_transformed[background_idx]

        explainer = shap.Explainer(
            model.predict,
            background,
            feature_names=transformed_feature_names
        )

        shap_exp = explainer(X_shap)
        shap_values = shap_exp.values

    shap_seconds = time.time() - start

    if len(shap_values.shape) == 3:
        shap_values = shap_values[:, :, 0]

    mean_abs_shap = np.abs(shap_values).mean(axis=0)

    shap_transformed_df = pd.DataFrame({
        "Transformed_Feature": transformed_feature_names,
        "Original_Feature": original_feature_map,
        "MeanAbsSHAP": mean_abs_shap
    }).sort_values("MeanAbsSHAP", ascending=False)

    shap_original_df = (
        shap_transformed_df
        .groupby("Original_Feature", as_index=False)["MeanAbsSHAP"]
        .sum()
        .sort_values("MeanAbsSHAP", ascending=False)
        .reset_index(drop=True)
    )

    shap_original_df["Rank"] = np.arange(1, len(shap_original_df) + 1)
    shap_original_df["SHAP_Source_Model"] = source_model_name
    shap_original_df["SHAP_Sample_Size"] = n_sample
    shap_original_df["SHAP_Seconds"] = shap_seconds

    print("SHAP completed.")
    print("SHAP sample size:", n_sample)
    print("SHAP time:", round(shap_seconds, 2), "seconds")

    display(shap_original_df)

    save_table(
        shap_transformed_df,
        "21a_shap_importance_transformed_onehot_features.csv"
    )

    save_table(
        shap_original_df,
        "21b_shap_importance_original_features.csv"
    )

    # Plot Top-14 to match the number of initial features.
    plot_df = shap_original_df.head(14).copy().sort_values("MeanAbsSHAP", ascending=True)

    plt.figure(figsize=(9, 6))
    plt.barh(plot_df["Original_Feature"], plot_df["MeanAbsSHAP"])
    plt.xlabel("Mean absolute SHAP value")
    plt.ylabel("Original Feature")
    plt.title(f"SHAP Feature Importance - {source_model_name}")
    plt.tight_layout()

    path = PLOT_DIR / "21_shap_feature_importance_original_top14.png"
    plt.savefig(path, dpi=180)
    plt.show()

    print("SHAP plot saved:", path)

    return shap_original_df, shap_transformed_df


# ============================================================
# SELECT SHAP SOURCE MODEL AND COMPUTE SHAP
# ============================================================

shap_source_row = select_shap_source_row(best_final_per_model)

shap_original_df, shap_transformed_df = compute_shap_ranking_original_features(
    shap_source_row
)

shap_ranked_features = shap_original_df["Original_Feature"].tolist()

print("\nFEATURE RANKING BASED ON SHAP:")
for i, f in enumerate(shap_ranked_features, start=1):
    print(f"{i}. {f}")

# ============================================================
# SHAP INTERPRETATION COMPLETED
# ============================================================
# The ranked SHAP features above are retained only as model-interpretation outputs.
# No SHAP feature-subset feature-selection retraining is performed in this publication-ready script.


# %% [cell 45]
# ============================================================
# CELL 16B - EXPLICIT HOLDOUT PREDICTION EXAMPLES:
# INPUT FEATURES -> ACTUAL TARGET -> PREDICTION -> ERROR
# ============================================================
# Purpose:
# Export explicit holdout prediction examples for manuscript reporting.

N_EXPLICIT_PREDICTION_EXAMPLES = 10

def parse_json_list_safely(value, default=None):
    if default is None:
        default = []

    if isinstance(value, list):
        return value

    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return parsed
        except Exception:
            return default

    return default


def parse_json_dict_safely(value, default=None):
    if default is None:
        default = {}

    if isinstance(value, dict):
        return value

    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            return default

    return default


def get_selected_features_from_result_row(row):
    if "Selected_Features" in row.index:
        selected = parse_json_list_safely(row.get("Selected_Features"), default=[])
        if len(selected) > 0:
            return selected

    return list(FEATURE_COLS)


def build_holdout_input_display(row, selected_features):
    selected_features = [col for col in selected_features if col in X_test.columns]

    Xte_display = X_test[selected_features].copy()
    yte_display = np.asarray(y_test, dtype=float)

    imputer_mode = row.get("Imputer_Mode", "median")

    if imputer_mode == "drop":
        test_mask = ~Xte_display.isna().any(axis=1)
        Xte_display = Xte_display.loc[test_mask].copy()
        yte_display = yte_display[test_mask.values]

    holdout_index = Xte_display.index.to_numpy()

    Xte_display = Xte_display.reset_index(drop=True)

    return Xte_display, yte_display, holdout_index


def get_prediction_source_for_explicit_table():
    """Return the final selected model used for explicit holdout prediction examples."""
    row = best_overall.copy()
    source_name = "Best Overall Model"

    selected_features = get_selected_features_from_result_row(row)

    if "best_pred_df" in globals():
        pred_df = best_pred_df.copy()
    else:
        model_params = parse_json_dict_safely(row.get("Model_Params", "{}"), default={})

        _, _, pred_df = fit_evaluate_train_test(
            model_name=row["Model"],
            scenario_name="explicit_prediction_best_overall_refit",
            X_train_df=X_train,
            y_train_series=y_train,
            X_test_df=X_test,
            y_test_series=y_test,
            imputer_mode=row["Imputer_Mode"],
            scaler_type=row["Scaler_Type"],
            target_transform=row["Target_Transform"],
            target_outlier_mode=row["Target_Outlier_Mode"],
            feature_outlier_mode=row["Feature_Outlier_Mode"],
            model_params=model_params,
            selected_features=selected_features,
            tuned=bool(row.get("Tuned", False)),
            stage="explicit_prediction_best_overall"
        )

    return row, pred_df, selected_features, source_name

explicit_row, explicit_pred_df, explicit_selected_features, explicit_source_name = (
    get_prediction_source_for_explicit_table()
)

Xte_display, yte_display, holdout_index = build_holdout_input_display(
    explicit_row,
    explicit_selected_features
)

n_rows = min(len(Xte_display), len(explicit_pred_df))

if len(Xte_display) != len(explicit_pred_df):
    print("WARNING: X_test display length and prediction length differ.")
    print("X_test display:", len(Xte_display))
    print("Predictions:", len(explicit_pred_df))
    print("The table will be truncated to the minimum length:", n_rows)

Xte_display = Xte_display.iloc[:n_rows].copy()
explicit_pred_df = explicit_pred_df.iloc[:n_rows].copy()
holdout_index = holdout_index[:n_rows]

actual_values = explicit_pred_df[PRED_ACTUAL_COL].to_numpy(dtype=float)
predicted_values = explicit_pred_df[PRED_PRED_COL].to_numpy(dtype=float)

residual_values = actual_values - predicted_values
absolute_error_values = np.abs(residual_values)

percentage_error_values = np.where(
    np.abs(actual_values) > 1e-12,
    absolute_error_values / np.abs(actual_values) * 100,
    np.nan
)

explicit_prediction_df = Xte_display.copy()

explicit_prediction_df.insert(0, "Holdout_Row_Index", holdout_index)
explicit_prediction_df.insert(1, "Prediction_Source", explicit_source_name)
explicit_prediction_df.insert(2, "Model", explicit_row["Model"])
explicit_prediction_df.insert(3, "Feature_Set", f"All_{len(explicit_selected_features)}")
explicit_prediction_df.insert(4, "Feature_Count", len(explicit_selected_features))

explicit_prediction_df["Actual_Core_Size_nm"] = actual_values
explicit_prediction_df["Predicted_Core_Size_nm"] = predicted_values
explicit_prediction_df["Residual_nm_Actual_minus_Predicted"] = residual_values
explicit_prediction_df["Absolute_Error_nm"] = absolute_error_values
explicit_prediction_df["Percentage_Error_%"] = percentage_error_values

explicit_prediction_df = explicit_prediction_df.reset_index(drop=True)

print("=" * 120)
print("EXPLICIT PREDICTION EXAMPLE")
print("=" * 120)
print("Prediction source:", explicit_source_name)
print("Model:", explicit_row["Model"])
print("Feature set:", explicit_row.get("Feature_Set", "All / Best Overall"))
print("Number of features used:", len(explicit_selected_features))
print("Features used:")
for i, col in enumerate(explicit_selected_features, start=1):
    print(f"{i}. {col}")

print("\nSource model metrics:")
metric_cols_to_show = [
    "CV_Mean_R2_Transformed",
    "R2_Test_Transformed",
    "CV_Mean_R2_Original",
    "Test_R2_Original",
    "Test_RMSE_Original",
    "Test_MAE_Original"
]

available_metric_cols = [col for col in metric_cols_to_show if col in explicit_row.index]
display(pd.DataFrame([explicit_row[available_metric_cols]]))

sample_n = min(N_EXPLICIT_PREDICTION_EXAMPLES, len(explicit_prediction_df))

explicit_prediction_examples_df = (
    explicit_prediction_df
    .sample(n=sample_n, random_state=RANDOM_STATE)
    .sort_values("Holdout_Row_Index")
    .reset_index(drop=True)
)

display_df = explicit_prediction_examples_df.copy()
numeric_cols = display_df.select_dtypes(include=[np.number]).columns
display_df[numeric_cols] = display_df[numeric_cols].round(4)

print("\nEXAMPLE TABLE: INPUT VARIABLES -> ACTUAL -> PREDICTED -> ERROR")
display(display_df)

save_table(
    explicit_prediction_df,
    "26_explicit_prediction_all_holdout_input_actual_pred_error.csv"
)

save_table(
    explicit_prediction_examples_df,
    "27_explicit_prediction_examples_for_article.csv"
)

summary_explicit_prediction = pd.DataFrame([{
    "Prediction_Source": explicit_source_name,
    "Model": explicit_row["Model"],
    "Feature_Set": explicit_row.get("Feature_Set", "All / Best Overall"),
    "Feature_Count": len(explicit_selected_features),
    "Displayed_Holdout_Data_Count": len(explicit_prediction_df),
    "Mean_Absolute_Error_nm": explicit_prediction_df["Absolute_Error_nm"].mean(),
    "Median_Absolute_Error_nm": explicit_prediction_df["Absolute_Error_nm"].median(),
    "Min_Absolute_Error_nm": explicit_prediction_df["Absolute_Error_nm"].min(),
    "Max_Absolute_Error_nm": explicit_prediction_df["Absolute_Error_nm"].max(),
    "Mean_Percentage_Error_%": explicit_prediction_df["Percentage_Error_%"].mean()
}])

display(summary_explicit_prediction)

save_table(
    summary_explicit_prediction,
    "28_explicit_prediction_error_summary.csv"
)


# %% [cell 46]
# ============================================================
# CELL 16C - AUDIT OVERFITTING MODEL FINAL
# ============================================================
# This cell is not used for model selection.
# Its only function is to discuss whether the final model indicates overfitting.

overfitting_audit_df = best_final_per_model.copy()

overfitting_audit_df["Train_Holdout_Gap_Status"] = np.where(
    overfitting_audit_df["Train_Test_Gap_R2_Original"].abs() > MAX_ALLOWED_TRAIN_HOLDOUT_GAP_R2,
    "Watch",
    "Aman"
)

overfitting_audit_df["CV_Train_Val_Gap_Status"] = np.where(
    overfitting_audit_df["CV_Mean_Train_Val_Gap_R2_Original"].abs() > MAX_ALLOWED_TRAIN_VAL_GAP_R2,
    "Watch",
    "Aman"
)

display(overfitting_audit_df[[
    "Model",
    "Stage",
    "Tuned",

    "CV_Mean_Train_R2_Original",
    "CV_Mean_R2_Original",
    "CV_Mean_Train_Val_Gap_R2_Original",
    "CV_Train_Val_Gap_Status",

    "Train_R2_Original",
    "Test_R2_Original",
    "Train_Test_Gap_R2_Original",
    "Train_Holdout_Gap_Status",

    "Test_RMSE_Original",
    "Test_MAE_Original",
    "Imputer_Mode",
    "Scaler_Type",
    "Target_Transform",
    "Target_Outlier_Mode",
    "Feature_Outlier_Mode"
]])

save_table(overfitting_audit_df, "17b_overfitting_audit_best_final_per_model.csv")

print("Interpretasi:")
print("- High Train R2 with much lower Holdout R2 indicates potential overfitting.")
print("- A large CV train-validation gap indicates that the model adapts too strongly to training data within CV folds.")
print("- This audit does not change model selection; it is used only for manuscript discussion.")


# %% [cell 47]
# ============================================================
# CELL 17 - NEW-DATA PREDICTION FUNCTION
# ============================================================

def load_best_model_package(model_path=MODEL_DIR / "best_overall_model.joblib"):
    return joblib.load(model_path)

def predict_new_experiment(new_data_dict, model_path=MODEL_DIR / "best_overall_model.joblib"):
    package = load_best_model_package(model_path)

    pipe = package["pipeline"]
    feature_cols = package["feature_cols"]
    feature_outlier_params = package.get("feature_outlier_params", {})

    new_df = pd.DataFrame([new_data_dict])

    for col in feature_cols:
        if col not in new_df.columns:
            new_df[col] = np.nan

    new_df = new_df[feature_cols].copy()

    for col in NUMERIC_COLS:
        if col in new_df.columns:
            new_df[col] = pd.to_numeric(new_df[col], errors="coerce")

    new_df = apply_feature_outlier_params(new_df, feature_outlier_params)

    pred_t = pipe.predict(new_df)

    target_transform = package["metadata"].get("Target_Transform", "none")
    pred_original = inverse_transform_target(pred_t, target_transform)

    return {
        "Prediction_Transformed": float(pred_t[0]),
        "Prediction_Core_Size_nm": float(pred_original[0]),
        "Target_Transform": target_transform
    }

print("The predict_new_experiment function is ready to use.")
print("Example:")

example_dict = {}

for c in FEATURE_COLS:
    non_missing = df_clean[c].dropna()
    if len(non_missing) == 0:
        example_val = np.nan
    elif c in CATEGORICAL_COLS:
        example_val = str(non_missing.iloc[0])
    else:
        example_val = float(non_missing.iloc[0])

    example_dict[c] = example_val

print("pred = predict_new_experiment({")
for k, v in example_dict.items():
    print(f"    {repr(k)}: {repr(v)},")
print("})")


# %% [cell 48]
# ============================================================
# CELL 18 - AUTOMATIC SUMMARY FOR THE ARTICLE
# ============================================================

print("=" * 120)
print("AUTOMATIC SUMMARY OF EXPERIMENT RESULTS")
print("=" * 120)

print("\n1. Dataset summary")
print(f"- Initial data count: {df_data.shape[0]} rows, {df_data.shape[1]} columns.")
print(f"- Data count after missing target and duplicates are cleaned: {df_clean.shape[0]} rows.")
print(f"- Number of input features: {len(FEATURE_COLS)} features; target: {TARGET_COL}.")
print("- Largest missing-value counts:")
display(missing_before.head(5))

print("\n2. Target and outliers")
display(target_outlier_summary)
print("Original and log1p target distributions have been saved in the plots folder.")

print("\n3. Train-test split biasa")
display(split_summary)

print("\n4. Main table: baseline vs preprocessing vs tuning")
display(main_comparison_df[[
    "Stage", "Model", "Tuned",

    # CV training and validation.
    "CV_Mean_Train_R2_Original",
    "CV_Mean_R2_Original",
    "CV_Mean_Train_Val_Gap_R2_Original",

    # Train-test split.
    "Train_R2_Original",
    "Test_R2_Original",
    "Train_Test_Gap_R2_Original",

    "Delta_Test_R2_Percentage_Points",
    "Test_RMSE_Original", "Test_RMSE_Improvement_Percent",
    "Test_MAE_Original", "Test_MAE_Improvement_Percent",

    "Imputer_Mode", "Scaler_Type",
    "Target_Transform", "Target_Outlier_Mode", "Feature_Outlier_Mode"
]].sort_values(["Model", "Stage"]))

print("\n5. Best final per model")
display(best_final_per_model[[
    "Model", "Stage", "Tuned", "CV_Mean_R2_Transformed", "CV_Mean_R2_Original",
    "Test_R2_Original", "Test_RMSE_Original", "Test_MAE_Original",
    "Imputer_Mode", "Scaler_Type", "Target_Transform", "Target_Outlier_Mode", "Feature_Outlier_Mode"
]].sort_values(PRIMARY_SELECTION_METRIC, ascending=False))

print("\n6. Train-test split comparison with final K-Fold")
display(kfold_final_df[[
    "Model", "K",

    # K-Fold training and validation.
    "CV_Mean_Train_R2_Original",
    "CV_Std_Train_R2_Original",
    "CV_Mean_R2_Original",
    "CV_Std_R2_Original",
    "CV_Mean_Train_Val_Gap_R2_Original",

    # Holdout.
    "Holdout_Test_R2_Original",
    "Delta_KFoldMean_vs_Holdout_R2_Original",

    "Holdout_Test_RMSE_Original", "CV_Mean_RMSE_Original", "CV_Std_RMSE_Original",
    "Holdout_Test_MAE_Original", "CV_Mean_MAE_Original", "CV_Std_MAE_Original"
]].sort_values(["Model", "K"]))

print("\nInterpretation notes:")
print("- Delta_Test_R2_Percentage_Points shows the R2 difference in percentage points, not relative percent.")
print("- Relative_Test_R2_Improvement_Percent is available in the CSV file, but it should not be used as the main claim if baseline R2 is small or negative.")
print("- For the manuscript, use a combination of R2, RMSE, MAE, delta R2, RMSE/MAE reduction percentage, and K-Fold consistency.")
print("- Combined preprocessing is selected from direct combination search based on training CV, not from a manual merge of separately best methods.")
print("- The models used include linear, regularized linear, kernel-based, boosting-tree, extremely randomized tree, and bagging tree ensemble models.")

print("\n4B. Summary of model counts that improved/decreased after preprocessing and tuning")
if "article_summary_df" in globals():
    display(article_summary_df[[
        "Reference_Stage",
        "Comparison_Stage",
        "Metric_Display",
        "Total_Model",
        "Number_Improved",
        "Number_Decreased",
        "Number_Unchanged",
        "Percent_Improved_%",
        "Percent_Decreased_%",
        "Dominant_Conclusion"
    ]])
else:
    print("article_summary_df is not yet available. Run Cell 13B first.")


print("\n4C. Perbandingan model reference/diminta reviewer vs Random Forest")
if "publication_model_comparison_df" in globals():
    display(publication_model_comparison_df[[
        "Rank_Overall",
        "Model_Group",
        "Model",
        "Stage",
        "CV_Mean_R2_Original",
        "Test_R2_Original",
        "Test_RMSE_Original",
        "Test_MAE_Original"
    ]])
else:
    print("publication_model_comparison_df is not yet available. Run Cell 13C first.")


print("\n7. Explicit prediction example: input variables -> actual -> predicted -> error")
if "explicit_prediction_examples_df" in globals():
    display_df = explicit_prediction_examples_df.copy()
    numeric_cols = display_df.select_dtypes(include=[np.number]).columns
    display_df[numeric_cols] = display_df[numeric_cols].round(4)
    display(display_df)
else:
    print("explicit_prediction_examples_df is not yet available. Run Cell 16B first.")

# %% [markdown]
# ## Export package reviewer-ready
#
# This final cell creates the README, output manifest, and output ZIP so the final structure resembles the reproducible reference repository. Numeric/figure files will be complete after the notebook is executed with the appropriate Excel dataset.


# %% [cell 50]
# ============================================================
# CELL 19A - EXPORT README, MANIFEST, AND REVIEWER-READY PACKAGE
# ============================================================

# Output manifest explaining the function of important files.
output_manifest_rows = [
    {"Folder": "data_snapshots", "File/Pattern": "00_raw_sheet_core_size_before_preprocessing.csv", "Purpose": "Raw data from the core_size sheet before feature selection and preprocessing."},
    {"Folder": "data_snapshots", "File/Pattern": "01_modeling_dataset_selected_features_before_model_preprocessing.csv", "Purpose": "Target + 14 features used before baseline/model preprocessing."},
    {"Folder": "data_snapshots", "File/Pattern": "03_train_before_any_model_preprocessing.csv", "Purpose": "Training subset before model preprocessing."},
    {"Folder": "data_snapshots", "File/Pattern": "04_test_before_any_model_preprocessing.csv", "Purpose": "Test subset before model preprocessing."},
    {"Folder": "data_snapshots", "File/Pattern": "baseline_default_complete_case_*", "Purpose": "Baseline data before and after complete-case missing-value dropping."},
    {"Folder": "preprocessed_matrices", "File/Pattern": "*_after_preprocessing_encoded.csv", "Purpose": "Numeric matrices after imputation/encoding/scaling for key scenarios."},
    {"Folder": "tables", "File/Pattern": "06c_baseline_preprocessing_data_flow_audit.csv", "Purpose": "Baseline row-flow audit before/after preprocessing."},
    {"Folder": "tables", "File/Pattern": "16b_scenario_to_preprocessing_config_map.csv", "Purpose": "Mapping of all scenarios to preprocessing configurations."},
    {"Folder": "tables", "File/Pattern": "16d_unique_preprocessing_config_data_flow_audit.csv", "Purpose": "Data-flow audit for all unique preprocessing configurations."},
    {"Folder": "tables", "File/Pattern": "16_main_baseline_preprocessing_tuning_comparison.csv", "Purpose": "Main comparison table for baseline, preprocessing, and tuning."},
    {"Folder": "tables", "File/Pattern": "17_best_final_per_model.csv", "Purpose": "Best final model per algorithm based on the main selection metric."},
    {"Folder": "json", "File/Pattern": "run_metadata.json", "Purpose": "Run metadata and global configuration."},
    {"Folder": "json", "File/Pattern": "model_configurations.json", "Purpose": "Default parameters and tuning spaces for all models."},
    {"Folder": "snippets", "File/Pattern": "protocol_summary.txt", "Purpose": "Protocol summary for manuscript/reviewer use."},
]
output_manifest_df = pd.DataFrame(output_manifest_rows)
save_table(output_manifest_df, "99_output_manifest_reviewer_ready.csv")

readme_text = f"""
# AgNP Core Size Regression - Reviewer-ready Reproducible Workflow

This folder is generated by the reviewer-ready notebook for the silver nanoparticle core-size prediction study.

## Pipeline study

The main pipeline is unchanged:

1. Load Excel sheet `core_size`.
2. Selection of target `core size (nm)` and 14 manuscript features.
3. Removal of missing targets and exact duplicates.
4. Train-test split 80:20.
5. Default complete-case/drop-missing baseline, without scaling/outlier handling/target transformation.
6. Single preprocessing.
7. Combined preprocessing search.
8. Hyperparameter tuning on the best preprocessing configuration for each model.
9. Final K-Fold validation, holdout evaluation, SHAP feature importance, explicit prediction examples, inverse design, and overfitting audit.

## Important outputs

- `tables/`: metric tables, preprocessing summaries, tuning results, K-Fold results, predictions, SHAP, and audit files.
- `data_snapshots/`: data before preprocessing and unencoded data after preprocessing.
- `preprocessed_matrices/`: matrices after preprocessing/encoding for baseline and key scenarios.
- `json/`: run metadata, package versions, and model configurations.
- `snippets/`: protocol summaries and manuscript-ready narrative snippets.
- `plots/` or `figures/`: model-comparison figures and supporting figures.

## Reviewer notes

The holdout test set is not used to select preprocessing strategies or hyperparameters. Selection is based on cross-validation on the training set. Data snapshots are exported for auditability and do not change the training process.
""".strip()

save_text(readme_text, "README_reviewer_ready.md", folder=OUTPUT_DIR)
save_text(readme_text, "README_reviewer_ready.md", folder=SNIPPET_DIR)

# Zip the output folder. The external notebook/repository package is created outside the notebook by the assistant.
if SAVE_OUTPUTS:
    reviewer_zip_base = shutil.make_archive(str(OUTPUT_DIR) + "_reviewer_ready", "zip", root_dir=OUTPUT_DIR)
    print("Reviewer-ready output ZIP dibuat:", reviewer_zip_base)
else:
    print("SAVE_OUTPUTS=False; ZIP was not created.")


# %% [cell 51]
# ============================================================
# CELL 19 - SAVE OUTPUT ZIP FOR DOWNLOAD
# ============================================================

if SAVE_OUTPUTS:
    zip_base = shutil.make_archive(str(OUTPUT_DIR), "zip", root_dir=OUTPUT_DIR)
    print("Output zip saved:", zip_base)

    try:
        from google.colab import files
        print("Mengunduh ZIP output...")
        files.download(zip_base)
    except Exception:
        print("If the file is not downloaded automatically, download it manually here:")
        print(zip_base)

print("\nNotebook completed.")
print("Tables folder:", TABLE_DIR)
print("Folder plot:", PLOT_DIR)
print("Folder model:", MODEL_DIR)
print("Predictions folder:", PRED_DIR)



# %% [script completion]
try:
    completion_text = f"""
AgNP core-size reviewer-ready workflow completed.
Output directory: {OUTPUT_DIR}
Tables: {TABLE_DIR}
Figures: {PLOT_DIR}
Models: {MODEL_DIR}
Predictions: {PRED_DIR}
JSON: {JSON_DIR}
Snippets: {SNIPPET_DIR}
Data snapshots: {DATA_SNAPSHOT_DIR}
Audit: {AUDIT_DIR}
Preprocessed matrices: {PREPROCESSED_MATRIX_DIR}
""".strip()
    (LOG_DIR / "run_completion_log.txt").write_text(completion_text + "\n", encoding="utf-8")
    print("Run completion log:", LOG_DIR / "run_completion_log.txt")
except Exception as exc:
    print("Completion log could not be written:", exc)
