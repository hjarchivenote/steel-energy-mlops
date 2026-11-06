# Steel Energy MLOps

**Industrial electricity consumption prediction with reproducible model training, serving, and monitoring.**

This project uses measurements from a South Korean steel manufacturing facility to estimate electricity consumption (`Usage_kWh`). It connects my manufacturing experience with practical machine learning engineering: tracking experiments, validating model changes, serving predictions, and monitoring changes in incoming data.

The repository supports an MLOps semester project at Óbuda University and will also serve as a portfolio project. The coursework uses a fixed 5,000-record sample; analysis of the complete dataset is a later extension within this same repository.

> **Status: local training and prediction workflow implemented.** Data preparation, model comparison, MLflow experiment tracking and registry, validation-based promotion, logged CLI predictions, and quality checks are implemented. The REST API, containerized services, and monitoring dashboard are still to be implemented. Reported model errors below are validation results; final test evaluation remains reserved.

## Project objective

Build a workflow that can:

1. Validate and version the data used for each experiment.
2. Train a regression pipeline and record its parameters, metrics, and artifacts.
3. Compare candidate models with the current model before promotion.
4. Expose training and prediction through a REST API and record requests and predictions.
5. Report data drift in a dashboard and support investigation or retraining when appropriate.
6. Start the complete system with Docker Compose and verify its behavior with automated tests.

The focus is reproducibility and reliable model operation. MAE and RMSE quantify prediction errors in kWh. The intended use is estimating a facility's expected consumption pattern and supporting investigation of changes, with explicit limits on what calendar-only inputs can explain.

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

## Model comparison and practical meaning

The first experiment uses the fixed coursework sample, seed 42, and the same seven calendar predictors for every model. Hyperparameters and the winning learned family are selected **only inside the 3,500-row training partition**, using three expanding windows with comparable elapsed-time validation periods. Row counts can differ because the sample is irregularly spaced. Scaling is fitted inside each Ridge pipeline's CV fold.

| Model | Training CV MAE (kWh) | Validation MAE (kWh) | Validation RMSE (kWh) |
|---|---:|---:|---:|
| Training mean | 29.85 | 27.86 | 30.95 |
| Training median | 24.16 | 20.57 | 35.66 |
| Ridge, square-root target | 17.22 | 15.93 | 22.50 |
| **Random forest** | **11.45** | **10.58** | **19.65** |
| Histogram gradient boosting, Poisson loss | 12.39 | 11.26 | 20.31 |

The selected random forest uses 100 trees, maximum depth 6, and minimum leaf size 2. The bounded search evaluates 13 configurations across the two trivial baselines and three learned families. Selection is based on training CV MAE, not on choosing the best-looking validation or final test result.

On the separate 750-row validation partition, random forest's MAE is approximately **48.6% lower than the stronger median baseline**, and 62.0% lower than the mean baseline. An MAE of 10.58 kWh means an average absolute error of that size per observation in this validation period. The higher RMSE highlights the importance of larger errors. These results do not demonstrate energy savings, performance at another facility, or accuracy on future operating regimes. The final 750 test records have not been scored by the training workflow.

Calendar information cannot describe unplanned shutdowns or production changes. The model therefore provides an expected consumption profile, with measured errors and an upgrade path, rather than a validated plant-control system. Production schedules or historical measurements can be considered later only after their availability at prediction time is established. Full-dataset analysis remains a separate extension after coursework requirements are met.

## Training, registry, and prediction

After restoring the coursework CSV and running `dvc repro`, activate the virtual environment and use:

```bash
steel-energy train
steel-energy predict --timestamp 2019-01-07T10:00:00
```

The prediction timestamp is a demonstration of the interface, not an observation with a verified 2019 target. Inputs must be timezone-naive and lie on the source's 15-minute grid. A request can contain up to 1,000 timestamps by repeating `--timestamp`. One explicit champion version is used for the entire request.

The default local tracking backend is `artifacts/mlflow/mlflow.db`; artifacts are stored locally and excluded from Git. Open the tracking UI with:

```bash
mlflow ui --backend-store-uri sqlite:///artifacts/mlflow/mlflow.db --host 127.0.0.1 --port 5001 --workers 1
```

Each training operation records:

- A parent run and one nested run per model family, including CV trials, parameters, validation metrics, and predictions.
- Serialized, reloadable sklearn pipelines with input examples, signatures, feature contracts, and exact model dependency versions.
- The preparation manifest, data hashes, training reference data, Git revision/dirty state, and a snapshot of the application source.
- The selected candidate's registry version and a machine-readable promotion decision. Every selected candidate is retained, including rejected ones.

Ridge uses a square-root target with a square inverse transformation; random forests average non-negative targets; boosted trees use a Poisson loss. The workflow checks all outputs for finite, non-negative consumption values. No final test targets are loaded by training or candidate selection.

**Promotion policy, fixed before the first real experiment**

1. The CV-selected candidate must improve validation MAE by at least 1% over the better mean/median baseline, while its RMSE is no worse than the mean baseline's RMSE.
2. When a champion exists, it is reloaded and re-evaluated on exactly the same current validation observations. The candidate must also improve its MAE by at least 1%, with no RMSE degradation.
3. Feature compatibility and training cutoff provenance must match the evaluation protocol. Comparison fails closed if the incumbent has already trained on the proposed validation period.
4. Only after training, validation, artifact logging, and registration finish can the `champion` alias change. A rejected candidate leaves the current champion in place.

The 1% threshold is an operational gate, not a statistical significance claim or a guarantee of future performance. Repeated experimentation must not turn the fixed validation partition into an unlimited tuning set. The untouched final test partition is reserved for the eventual final assessment.

For example, requesting a Ridge-only candidate after the random forest has become champion exercises the rejection path:

```bash
steel-energy train --families ridge
```

Prediction operations record their exact timestamp request, generated features, response, and model version in MLflow. Tracking failures produce an error instead of an unrecorded successful response. Local training uses a workspace lock to prevent overlapping local promotion jobs; distributed deployment coordination remains future work.

**MLflow Projects entry points**

`MLproject` exposes `prepare`, `train`, and `predict`, sharing the same application functions as the CLI. For example, after the normal environment setup:

```bash
export MLFLOW_TRACKING_URI=sqlite:///artifacts/mlflow/mlflow.db
mlflow run . -e train --env-manager local --experiment-name steel-energy-training
```

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
├── MLproject                # Shared prepare/train/predict entry points
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

The tests cover invalid data, chronological isolation, feature leakage, reproducible outputs, CV windows, promotion acceptance/rejection, real SQLite tracking and registry behavior, saved-model round trips, request artifacts, locking, and failure handling. Tests use synthetic observations and never train or select models using the real final evaluation data.

## Implementation scope

| Area | Intended implementation | Status |
|---|---|---|
| Data | Source attribution, fixed sample, validation, DVC preparation and provenance | Implemented; shared storage pending |
| Feature/evaluation preparation | Calendar-only predictors and chronological train/validation/test split | Implemented |
| Modeling | Train-only CV and comparison with mean/median baselines | Implemented; final test evaluation reserved |
| Experiment tracking | MLflow Projects, artifacts, reloadable models, registry | Implemented locally |
| Model management | Candidate comparison and promotion safeguards | Implemented locally |
| Serving | Version-pinned, logged CLI prediction; REST endpoints | CLI implemented; REST API planned |
| Monitoring | Evidently reports displayed in Streamlit | Planned |
| Runtime | Docker images and a single Compose startup command | Planned |
| Quality | Tests, coverage, Flake8, Pyright, pre-commit, GitHub Actions | Implemented for current application code |
| Optional extension | Airflow orchestration and full-dataset analysis | Planned |

The implemented application must maintain separate 95% coverage gates for source and test code, Pyright standard mode, and Flake8 compliance. REST, container builds, and monitoring require their own implementation and validation as they are added.

## Development workflow

After the initial repository bootstrap, development will use feature branches and pull requests merged into `main` with merge commits so that branch history is retained. Commits will represent actual implementation and validation work.

## AI assistance

AI assistance was used for dataset selection, sampling validation, repository organization, application code, model comparison, tracking integration, and tests. Generated changes and technical claims are checked against the source data, recorded experiments, and automated validation before being presented as completed results.
