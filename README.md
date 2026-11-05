# Steel Energy MLOps

**Industrial electricity consumption prediction with reproducible model training, serving, and monitoring.**

This project uses measurements from a South Korean steel manufacturing facility to estimate electricity consumption (`Usage_kWh`). It connects my manufacturing experience with practical machine learning engineering: tracking experiments, validating model changes, serving predictions, and monitoring changes in incoming data.

The repository supports an MLOps semester project at Óbuda University and will also serve as a portfolio project. The coursework uses a fixed 5,000-record sample; analysis of the complete dataset is a later extension within this same repository.

> **Status: data foundation implemented.** CSV validation, calendar feature generation, chronological evaluation partitions, DVC preparation, and automated quality checks are implemented. Model training, the REST API, containerized services, and the monitoring dashboard are still to be implemented. No model performance results are reported yet.

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

## Initial feature and evaluation contract

The first baseline uses **only the requested timestamp** to construct seven predictors: sine/cosine pairs for time of day, weekday, and month, plus a weekend indicator. This requires no fitted preprocessing statistics and uses no observed electricity consumption or contemporaneous electrical measurements as inputs. `Load_Type` is also excluded until its availability before prediction is established.

The target is `Usage_kWh`. The prepared CSVs retain `timestamp` as audit metadata; model inputs must be selected explicitly using `FEATURE_NAMES` from `steel_energy_mlops.features`. Neither the target nor the audit timestamp is an additional model feature.

The existing sample is sorted by timestamp and split without shuffling:

| Partition | Records | First timestamp | Last timestamp | Purpose |
|---|---:|---|---|---|
| Train | 3,500 (70%) | 2018-01-01 01:15 | 2018-09-14 12:00 | Fit models and any learned preprocessing |
| Validation | 750 (15%) | 2018-09-14 14:30 | 2018-11-10 05:00 | Compare candidates and decide promotion |
| Test | 750 (15%) | 2018-11-10 05:45 | 2018-12-31 20:45 | Final evaluation only; never tune or promote on this partition |

Timestamps remain timezone-naive as supplied by the dataset. The preparation code validates the original 11-column schema, nonempty cells, dates, finite non-negative measurements, power-factor bounds, calendar labels, NSM consistency, unique timestamps, and the expected record count. Invalid records cause an error instead of being silently dropped.

Each preparation run writes `train.csv`, `validation.csv`, `test.csv`, and a machine-readable `manifest.json` containing source/output hashes, feature names, partition sizes, and date boundaries. It creates no trained model and uses no row-based lag features.

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
│   └── processed/           # Generated features, partitions, and provenance manifest
├── src/steel_energy_mlops/   # Validation, features, splitting, preparation, and CLI
├── tests/                   # Synthetic unit and integration tests; no data download needed
├── dashboard/               # Reserved for the Streamlit entry point
├── dags/                    # Reserved for optional Airflow workflows
├── docs/screenshots/        # Reserved for screenshots of the running system
├── .github/workflows/       # Automated quality checks on pushes and pull requests
├── dvc.yaml / dvc.lock      # Reproducible preparation stage and artifact hashes
├── pyproject.toml           # Package, CLI, pytest, coverage, and Pyright configuration
├── requirements.txt         # Pinned modeling dependencies
├── requirements-dev.txt     # Pinned development tools
└── README.md
```

The dashboard, Airflow, and screenshot directories remain placeholders. Submission PDFs and ZIP archives are kept locally in the ignored `submission/` directory.

## Data version control

Git tracks source files and DVC metadata. CSV files and the DVC cache are excluded from Git.

Repository tooling uses Python 3.12:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt -e .
python -m pre_commit install
dvc repro
```

`dvc repro` validates the fixed coursework sample, generates model-ready partitions, and records them in the local DVC cache. To run the same preparation function directly:

```bash
steel-energy prepare-data
```

The data cache is currently local. A shared DVC remote is not configured yet; a fresh clone retrieves metadata and code, not the CSV files. To prepare the real dataset, restore the original coursework CSV to `data/coursework/Steel_industry_data_5000.csv` first. Tests use synthetic observations and do not require the CSV or remote storage. The complete original dataset is available from the UCI link above.

## Development checks

Activate the virtual environment, then run:

```bash
python -m pre_commit run --all-files
```

The same checks run automatically before local commits and in GitHub Actions: Flake8 for Python files, Pyright standard mode for `src/`, pytest with branch coverage, and separate minimum 95% coverage gates for `src/` and `tests/`. No application or test files are omitted from coverage.

The tests cover invalid data, duplicate timestamps, chronological isolation, irregular time gaps, feature leakage, reproducible outputs, preservation of source/existing files on validation failures, and CLI behavior. Tests do not train or select models using final evaluation data.

## Implementation scope

| Area | Intended implementation | Status |
|---|---|---|
| Data | Source attribution, fixed sample, validation, DVC preparation and provenance | Implemented; shared storage pending |
| Feature/evaluation preparation | Calendar-only predictors and chronological train/validation/test split | Implemented |
| Modeling | Fit and compare baseline regressors | Planned |
| Experiment tracking | MLflow pipelines, experiment artifacts, model registry | Planned |
| Model management | Candidate comparison and promotion safeguards | Planned |
| Serving | Training and prediction endpoints with request logging | Planned |
| Monitoring | Evidently reports displayed in Streamlit | Planned |
| Runtime | Docker images and a single Compose startup command | Planned |
| Quality | Tests, coverage, Flake8, Pyright, pre-commit, GitHub Actions | Implemented for the data foundation |
| Optional extension | Airflow orchestration and full-dataset analysis | Planned |

The data foundation meets the local coverage, type-checking, and lint gates. The same standards must be maintained as model training, APIs, container builds, and monitoring are added; those later components are not covered by the current implementation.

## Development workflow

After the initial repository bootstrap, development will use feature branches and pull requests merged into `main` with merge commits so that branch history is retained. Commits will represent actual implementation and validation work.

## AI assistance

AI assistance was used for dataset selection, sampling validation, repository organization, data preparation code, and tests. Generated changes and technical claims are checked against the source data and automated validation before being presented as completed results.
