# Steel Energy MLOps

**Industrial electricity consumption prediction with reproducible model training, serving, and monitoring.**

This project uses measurements from a South Korean steel manufacturing facility to estimate electricity consumption (`Usage_kWh`). It connects my manufacturing experience with practical machine learning engineering: tracking experiments, validating model changes, serving predictions, and monitoring changes in incoming data.

The repository supports an MLOps semester project at Óbuda University and will also serve as a portfolio project. The coursework uses a fixed 5,000-record sample; analysis of the complete dataset is a later extension within this same repository.

> **Status: initial repository setup.** Dataset selection and sampling are complete. Model training, the REST API, containerized services, automated quality checks, and the monitoring dashboard are still to be implemented. No model scores or running-system results are reported yet.

## Project objective

Build a workflow that can:

1. Validate and version the data used for each experiment.
2. Train a regression pipeline and record its parameters, metrics, and artifacts.
3. Compare candidate models with the current model before promotion.
4. Expose training and prediction through a REST API and record requests and predictions.
5. Report data drift in a dashboard and support investigation or retraining when appropriate.
6. Start the complete system with Docker Compose and verify its behavior with automated tests.

The focus is reproducibility and reliable model operation. MAE and RMSE will be used for evaluation; results will be added after the implementation has been tested.

## Dataset

| Item | Description |
|---|---|
| Dataset | [Steel Industry Energy Consumption](https://archive.ics.uci.edu/dataset/851/steel+industry+energy+consumption) |
| Source | UCI Machine Learning Repository |
| Original CSV | 35,040 records, 11 columns |
| Coursework sample | 5,000 records, the same 11 columns |
| Target | `Usage_kWh` - electricity consumption in kWh |
| Task | Supervised regression |
| Data license | [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/) |

The sample was drawn without replacement using `pandas.DataFrame.sample(n=5000, replace=False, random_state=42)` and then sorted by parsed timestamp. Original column names and selected cell values were preserved. The CSV is UTF-8 with a BOM and has a header row.

The existing sampling validation found no missing cells. Maximum differences in month, hour, weekday, and load-type shares were below 0.75 percentage points; mean electricity consumption differed by approximately +0.525%. These checks describe sample representativeness, not model performance.

**Modeling considerations**

- The sampled rows have irregular time gaps. The preceding row must not be treated as the measurement from 15 minutes earlier.
- Predictor availability must be checked before claiming future forecasting. Contemporaneous electrical measurements may be appropriate for estimation but unavailable for a future prediction.
- The availability of `Load_Type` before prediction is not established; treating it as a known production schedule requires an explicit assumption.
- The CO2 column has inconsistent unit metadata between its header and the UCI description. Its unit should not be inferred silently.
- Data drift alone does not establish prediction error. Assessing performance on new observations requires observed target values.

**Dataset attribution**

V E, S., Shin, C., & Cho, Y. (2021). *Steel Industry Energy Consumption* [Dataset]. UCI Machine Learning Repository. [https://doi.org/10.24432/C52G8C](https://doi.org/10.24432/C52G8C).

The coursework CSV is a sampled and date-sorted adaptation of this dataset. The dataset license applies to the source data and its adaptation.

## Repository layout

```text
steel-energy-mlops/
├── .dvc/                    # Data version-control configuration
├── data/
│   ├── raw/                 # Complete source CSV; tracked with DVC
│   ├── coursework/          # Fixed 5,000-row sample; tracked with DVC
│   └── processed/           # Reserved for model-ready data
├── src/steel_energy_mlops/   # Reserved for application and pipeline logic
├── tests/                   # Reserved for automated tests
├── dashboard/               # Reserved for the Streamlit entry point
├── dags/                    # Reserved for optional Airflow workflows
├── docs/screenshots/        # Reserved for screenshots of the running system
├── requirements-dev.txt     # Repository tooling
└── README.md
```

Only the repository foundation and data tracking are present. Reserved directories contain placeholders, not implemented services. Submission PDFs and ZIP archives are kept locally in the ignored `submission/` directory.

## Data version control

Git tracks source files and DVC metadata. CSV files and the DVC cache are excluded from Git.

Repository tooling uses Python 3.12:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
dvc status
```

The data cache is currently local. A shared DVC remote and an automated data preparation command are not configured yet; a fresh clone retrieves the dataset metadata, not the CSV files. The original dataset is available from the UCI link above.

## Implementation scope

| Area | Intended implementation | Status |
|---|---|---|
| Data | Source attribution, fixed coursework sample, DVC metadata | Initialized |
| Modeling | Validation, preprocessing, time-aware evaluation, baseline models | Planned |
| Experiment tracking | MLflow pipelines, experiment artifacts, model registry | Planned |
| Model management | Candidate comparison and promotion safeguards | Planned |
| Serving | Training and prediction endpoints with request logging | Planned |
| Monitoring | Evidently reports displayed in Streamlit | Planned |
| Runtime | Docker images and a single Compose startup command | Planned |
| Quality | Tests, coverage, Flake8, Pyright, pre-commit, GitHub Actions | Planned |
| Optional extension | Airflow orchestration and full-dataset analysis | Planned |

The coursework quality targets include at least 95% coverage of application and test code, no Pyright standard-mode errors in `src/`, and compliance with the specified Flake8 rules. These are acceptance targets, not results already achieved.

## Development workflow

After the initial repository bootstrap, development will use feature branches and pull requests merged into `main` with merge commits so that branch history is retained. Commits will represent actual implementation and validation work.

## AI assistance

AI assistance was used for dataset selection, sampling validation, and repository organization. Generated changes and technical claims are reviewed against the source data and verified before being presented as completed results.
