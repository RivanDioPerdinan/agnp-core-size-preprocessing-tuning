# The Effect of Preprocessing and Hyperparameter Tuning on Core Size Prediction of Silver Nanoparticles

This repository contains the source code, dataset placement instructions, and generated computational outputs for a machine-learning regression study on silver nanoparticle (AgNP) core size prediction. The workflow is prepared for journal submission and reproducible public archiving.

## Short study description

The study evaluates how preprocessing strategies and hyperparameter tuning affect the predictive performance of regression models for AgNP core size. The regression target is `core size (nm)`, and the input variables describe synthesis-related conditions from the `core_size` sheet of the Furxhi et al. supplementary dataset.

The workflow includes:

- dataset loading from the `core_size` Excel sheet;
- selection of 14 input features and one regression target;
- duplicate removal and initial dataset audit;
- 80:20 train-holdout split;
- fold-wise preprocessing fitted only on the corresponding training fold to reduce preprocessing leakage risk;
- baseline model evaluation;
- single-preprocessing experiments;
- combined-preprocessing experiments;
- hyperparameter tuning on the best preprocessing configuration per model;
- final 5-, 10-, and 15-fold cross-validation checks;
- SHAP-based feature-importance interpretation;
- explicit holdout prediction examples;
- optional exploratory inverse-design candidate ranking, which is not used in the manuscript analyses or conclusions.

The evaluated regression models are:

- Multiple Linear Regression;
- Ridge Regression;
- Support Vector Regression;
- XGBoost Regression;
- Extra Trees Regression;
- Random Forest Regression.

## Repository structure

```text
AgNP_core_size_publication/
├── data/
│   ├── 1-s2.0-S2001037024000333-mmc1.xlsx
│   └── README.md
├── outputs/
│   ├── data_snapshots/
│   ├── figures/
│   ├── inverse_design/
│   ├── json/
│   ├── logs/
│   ├── models/
│   ├── predictions/
│   ├── preprocessed_matrices/
│   ├── snippets/
│   ├── tables/
│   └── README_reviewer_ready.md
├── src/
│   └── ag_np_core_size.py
├── archive/
│   ├── outputs.zip
│   └── README.md
├── manuscript/
│   ├── manuscript_with_author_details.docx
│   ├── manuscript_anonymous.docx
│   └── cover_letter.docx
├── .gitignore
├── CITATION.cff
├── environment.yml
├── LICENSE
├── requirements.txt
└── README.md
```

If some folders are not included in a public GitHub release, they can be stored in a GitHub Release or Zenodo archive instead.

## Dataset source

The dataset used in this study is the supplementary material of:

Furxhi et al. (2024), *Design rules applied to silver nanoparticles synthesis: A practical example of machine learning application*.  
DOI: `10.1016/j.csbj.2024.02.010`

The workflow uses the Excel file:

```text
data/1-s2.0-S2001037024000333-mmc1.xlsx
```

and the sheet:

```text
core_size
```

The initial `core_size` sheet contains 2,113 data points and 18 columns. After selecting the target and manuscript feature set, removing exact duplicate rows, and excluding high-missingness features, the final modelling dataset contains:

- 816 unique data points;
- 14 input features;
- 4 numerical features;
- 10 categorical features;
- 1 target variable: `core size (nm)`.

Three features are excluded because of high missingness:

- `Stiring_Speed_rpm_`;
- `Capping agent concentration (mg/mL)`;
- `Reducing agent quantity_Ml`.

The dataset uses `?` as a missing-value token, which is converted to `NaN` during preprocessing.

## How to install dependencies

Python 3.12.3 is the reference Python version used to generate the reported computational outputs.

### Option 1: using pip

From the repository root, run:

```bash
python -m venv .venv
```

Activate the environment:

```bash
# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

### Option 2: using Conda

```bash
conda env create -f environment.yml
conda activate agnp-core-size
```

## How to run the script

Run the script from the repository root.

### Fast check

Use this mode only to check whether the workflow can start and whether the environment is configured correctly.

```bash
python src/ag_np_core_size.py --data data/1-s2.0-S2001037024000333-mmc1.xlsx --sheet core_size --output outputs --fast-mode
```

### Full reproducible run

This mode runs the full computational workflow. It may take a long time because the combined preprocessing grid and full hyperparameter tuning are computationally intensive.

```bash
python src/ag_np_core_size.py --data data/1-s2.0-S2001037024000333-mmc1.xlsx --sheet core_size --output outputs
```

### Optional arguments

```bash
python src/ag_np_core_size.py \
  --data data/1-s2.0-S2001037024000333-mmc1.xlsx \
  --sheet core_size \
  --output outputs \
  --cv-tuning 5 \
  --k-values-final 5,10,15
```

## Expected input

The expected input file is:

```text
data/1-s2.0-S2001037024000333-mmc1.xlsx
```

The expected sheet name is:

```text
core_size
```

The target column is:

```text
core size (nm)
```

The 14 input features used in the modelling workflow are:

```text
Process
Steps
External_energy
Temp_Celcius
Stirring
Synthesis_Duration_h
Scale_synthesis_ml
Precurcor_conc_mM
NEW_Capping agent
NEW_capping agent_Class
NEW_reducing agent
NEW_reducing agent_class
Order of reagent_CODE
Treatment
```

## Expected output

The script writes generated outputs to:

```text
outputs/
```

The main output subfolders are:

```text
outputs/tables/                  CSV tables for manuscript and reviewer audit
outputs/figures/                 Generated figures
outputs/json/                    Run metadata, package versions, and model configurations
outputs/snippets/                Text summaries for reporting
outputs/logs/                    Run logs
outputs/data_snapshots/          Raw, cleaned, train/test, and preprocessing snapshots
outputs/audit/                   Reviewer-audit files
outputs/preprocessed_matrices/   Selected processed matrices
outputs/models/                  Serialized trained model pipeline(s)
outputs/predictions/             Holdout prediction tables
outputs/inverse_design/          Exploratory inverse-design outputs not used in manuscript analyses or conclusions
```

Important output files include:

```text
outputs/tables/08_baseline_results_with_default_params.csv
outputs/tables/11_paper_table_pure_single_preprocessing_summary.csv
outputs/tables/12_best_combined_preprocessing_per_model.csv
outputs/tables/14_hyperparameter_tuning_results_paper.csv
outputs/tables/17_best_final_per_model.csv
outputs/tables/18_final_kfold_vs_holdout_summary.csv
outputs/tables/21b_shap_importance_original_features.csv
outputs/tables/27_explicit_prediction_examples_for_article.csv
outputs/tables/99_output_manifest_reviewer_ready.csv
outputs/predictions/
outputs/inverse_design/
```

The `archive/` folder may contain zipped versions of full outputs:

```text
archive/outputs.zip
archive/outputs_reviewer_ready.zip
```

The archive/outputs.zip file contains the archived computational outputs associated with the final reproducibility package.

## Key reproduced metrics

The main manuscript-level results expected from the workflow are:

### Dataset and scenario counts

| Item | Value |
|---|---:|
| Initial `core_size` data points | 2,113 |
| Final modelling data points after duplicate removal | 816 |
| Input features | 14 |
| Numerical features | 4 |
| Categorical features | 10 |
| Target variable | `core size (nm)` |
| Total preprocessing and tuning scenarios | 18,624 |
| Scenarios improving baseline CV R² | 5,021 |
| Scenarios decreasing baseline CV R² | 13,537 |
| Scenarios unchanged from baseline CV R² | 66 |

### Baseline model examples

| Model | CV R² | Std CV R² | RMSE | MAE |
|---|---:|---:|---:|---:|
| XGBoost | 0.5211 | 0.2390 | 27.9980 | 16.0266 |
| Random Forest Regression | 0.5166 | 0.2083 | 28.8731 | 16.7757 |
| Extra Trees Regression | 0.5119 | 0.2322 | 26.3401 | 15.5012 |
| SVR | -0.0906 | 0.0420 | 41.9310 | 25.8048 |

### Best combined preprocessing results

| Model | CV R² | Std CV R² | RMSE | MAE |
|---|---:|---:|---:|---:|
| Extra Trees Regression | 0.6160 | 0.0836 | 25.2461 | 15.0600 |
| Random Forest Regression | 0.6117 | 0.0635 | 25.4519 | 15.5902 |
| XGBoost Regression | 0.6025 | 0.0755 | 25.9493 | 15.7180 |
| SVR | 0.3366 | 0.1274 | 32.4398 | 18.7987 |

### Best final model after hyperparameter tuning

The best final model selected by cross-validation is:

```text
Random Forest Regression after hyperparameter tuning
```

Expected final performance:

| Metric | Value |
|---|---:|
| CV R² | 0.6190 |
| Std CV R² | 0.0677 |
| RMSE | 26.0754 nm |
| MAE | 15.8576 nm |

### SHAP interpretation

The five most influential original features from the Random Forest SHAP interpretation are expected to include:

```text
Scale_synthesis_ml
Order of reagent_CODE
Stirring
NEW_Capping agent
NEW_reducing agent
```

These features are used for interpretation only. SHAP is not used for feature-subset retraining in the final manuscript workflow.

## Reproducibility notes

- The random state is set to `42`.
- The holdout test set is not used for selecting preprocessing configurations or hyperparameters.
- Cross-validation is performed on the training data.
- Preprocessing is fitted inside each training fold and then applied to the validation fold to reduce data leakage risk.
- Final evaluation includes 5-, 10-, and 15-fold validation summaries.
- Some numerical results may vary slightly across library versions, operating systems, and hardware.

## Citation

If you use this repository, cite the manuscript and dataset source.

### Manuscript citation

```bibtex
@article{pardede_agnp_preprocessing_tuning_2026,
  title  = {The Effect of Preprocessing and Hyperparameter Tuning on Core Size Prediction of Silver Nanoparticles},
  author = {Pardede, Jasman and Perdinan, Rivan Dio and Milandga Milenio, Rizka and Premitasari, Marisa},
  year   = {2026},
  note   = {Manuscript prepared for submission to PeerJ Computer Science}
}
```

### Dataset citation

```bibtex
@article{furxhi_2024_agnp_design_rules,
  title   = {Design rules applied to silver nanoparticles synthesis: A practical example of machine learning application},
  author  = {Furxhi, Irini and others},
  journal = {Computational and Structural Biotechnology Journal},
  year    = {2024},
  doi     = {10.1016/j.csbj.2024.02.010}
}
```

The archived reproducibility package is available on Zenodo at: https://doi.org/10.5281/zenodo.22153591.

## License note

The source code in this repository is released under the license provided in:

```text
LICENSE
```

If the included license is MIT, it applies to the source code and repository documentation created by the authors. It does not automatically apply to third-party data, third-party articles, or external supplementary materials.

## Data redistribution note

The raw Excel dataset originates from the supplementary material of Furxhi et al. (2024). Before making this repository public, verify whether the dataset license permits redistribution of the raw Excel file.

Recommended public-release options:

1. If redistribution is permitted, keep the Excel file in `data/` and cite the original source clearly.
2. If redistribution is not clearly permitted, remove the raw Excel file from the public repository and provide instructions for users to download it from the original publication DOI.
3. Generated output files derived from the workflow can be released separately, but they should still cite the original dataset source.

The manuscript associated with this repository is currently prepared for submission to PeerJ Computer Science. Authors and users should refer to the current journal submission requirements for manuscript and supplementary-file preparation.

## Contact

For submission correspondence, the corresponding author and contact details are provided in the submitted manuscript and in the journal submission system.
