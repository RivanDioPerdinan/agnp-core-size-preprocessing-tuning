# Data

The workflow uses the `core_size` sheet from the supplementary Excel dataset by Furxhi et al. The expected filename is:

```text
1-s2.0-S2001037024000333-mmc1.xlsx
```

For local reproducibility, place the file in this folder and run:

```bash
python src/ag_np_core_size.py --data data/1-s2.0-S2001037024000333-mmc1.xlsx --sheet core_size --output outputs
```

For a public GitHub release, verify whether redistribution of the raw Excel file is allowed. If redistribution is not allowed, remove the raw Excel file from this folder and cite the original supplementary material in the manuscript and repository README.

## Data handling summary

- Original sheet used: `core_size`.
- Initial sheet size: 2,113 rows and 18 columns.
- Target variable: `core size (nm)`.
- Feature set used in the model: 14 input features.
- High-missingness features excluded from modelling: `Stiring_Speed_rpm_`, `Capping agent concentration (mg/mL)`, and `Reducing agent quantity_Ml`.
- Missing token: `?` is converted to `NaN`.
- Exact duplicate rows are removed after selecting the target and article-style feature set.
- Final modelling dataset after duplicate removal: 816 rows.
