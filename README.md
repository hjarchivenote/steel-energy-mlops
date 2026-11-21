# Steel Energy MLOps

**Industrial electricity consumption prediction with reproducible model training, serving, and monitoring.**

This project uses measurements from a South Korean steel manufacturing facility to estimate electricity consumption (`Usage_kWh`). It connects my manufacturing experience with practical machine learning engineering: tracking experiments, validating model changes, serving predictions, and monitoring changes in incoming data.

The repository supports an MLOps semester project at Óbuda University and will also serve as a portfolio project. The coursework uses a fixed 5,000-record sample; analysis of the complete dataset is a later extension within this same repository.

> **Status: training and prediction APIs packaged with Docker Compose.** Data preparation, model comparison, MLflow tracking and registry, validation-based promotion, asynchronous training jobs, logged predictions, and a persistent container runtime are implemented. The monitoring dashboard remains to be implemented. Reported model errors below are validation results; final test evaluation remains reserved.

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

### Candidate inputs for a 15-minute forecast

The candidate task estimates `Usage_kWh` at a requested timestamp using information available 15 minutes earlier. It uses the calendar features and the latest available historical consumption, reactive-energy and power-factor readings, with the measurement age and a history-availability indicator. The input builder requires measurement time ≤ arrival time ≤ forecast origin and retains the selected measurement and arrival timestamps for inspection.

The [UCI variable descriptions](https://archive.ics.uci.edu/dataset/851/steel+industry+energy+consumption) do not establish when these readings reach a forecasting system. The following audit simulates a 15-minute reporting delay and examines only the 3,500 training observations.

```bash
steel-energy audit-inputs --reporting-delay-minutes 15
```

Only 14.2% of successive training observations are exactly 15 minutes apart. Under the simulated delay, the selected historical reading has a median age of 75 minutes at forecast origin and a maximum age of 825 minutes. When no measurement has arrived, the historical fields are zero with `history_available=0`.

A same-row calculation using lagging reactive energy and power factor approximately reconstructs the target, with a median absolute difference of 0.0010 kWh among 2,786 eligible training rows. CO₂ has a training-target correlation of 0.991 and ambiguous units. These diagnostics motivate excluding same-row electrical measurements and CO₂. `Load_Type` remains excluded pending evidence of its derivation and availability before the forecast.

MLflow records the input decisions, source hashes and timestamps for the candidate training features. The next evaluation requires a rolling backtest against the calendar reference and a forecast that repeats the latest available consumption reading. Candidate adoption depends on performance under the declared availability rule, including high-usage errors. The current results below describe the calendar model.

## Model comparison and practical meaning

The first experiment uses the fixed coursework sample, seed 42, and the same seven calendar predictors for every model. Hyperparameters and the winning learned family are selected **only inside the 3,500-row training partition**, using three expanding windows with comparable elapsed-time validation periods. Row counts can differ because the sample is irregularly spaced. Scaling is fitted inside each Ridge pipeline's CV fold.

| Model | Training CV MAE (kWh) | Validation MAE (kWh) | Validation RMSE (kWh) |
|---|---:|---:|---:|
| Training mean | 29.85 | 27.86 | 30.95 |
| Training median | 24.16 | 20.57 | 35.66 |
| Ridge, square-root target | 17.22 | 15.93 | 22.50 |
| **Random forest** | **11.45** | **10.58** | **19.65** |
| Histogram gradient boosting, Poisson loss | 12.39 | 11.26 | 20.31 |

The initial selected random forest used squared-error splits, 100 trees, maximum depth 6, and minimum leaf size 2. That initial bounded search evaluated 13 configurations across the two trivial baselines and three learned families. Selection is based on training CV MAE, not on choosing the best-looking validation or final test result.

On the separate 750-row validation partition, random forest's MAE is approximately **48.6% lower than the better of the two constant baselines**, and 62.0% lower than the mean baseline. An MAE of 10.58 kWh means an average absolute error of that size per observation in this validation period. The higher RMSE highlights the importance of larger errors. The schedule-aware comparison below places this improvement in context. These results do not demonstrate energy savings, performance at another facility, or accuracy on future operating regimes. The final 750 test records have not been scored by the training or diagnostics workflows.

Calendar information cannot describe unplanned shutdowns or production changes. The model therefore provides an expected consumption profile, with measured errors and an upgrade path, rather than a validated plant-control system. Production schedules or historical measurements can be considered later only after their availability at prediction time is established. Full-dataset analysis remains a separate extension after coursework requirements are met.

### Schedule-aware reference and error diagnosis

Run `steel-energy evaluate` after preparation and initial training. This compares one pinned champion version with fixed weekday/hour **mean and median** profiles, fitted only on training observations. Both profiles are also refitted inside the same three expanding training CV windows; their reference is selected by training CV MAE. Hourly grouping and the fallback order (weekday/hour, weekday-or-weekend/hour, hour, global statistic) are fixed rather than tuned against validation errors. All 750 real validation observations have a matching weekday/hour training cell.

| Model | Training CV MAE (kWh) | Validation MAE (kWh) | Validation RMSE (kWh) | Validation 95th-percentile absolute error (kWh) |
|---|---:|---:|---:|---:|
| Random forest, registered version 1 | 11.45 | 10.58 | **19.65** | **44.99** |
| Weekday/hour mean | 16.16 | 12.99 | 21.28 | 51.67 |
| Weekday/hour median, CV-selected reference | 13.64 | **10.37** | 20.81 | 53.68 |

The original squared-error forest **does not beat the schedule median on validation MAE**: its error is 0.207 kWh higher per observation, although its RMSE and 95th-percentile absolute error are lower. It was the winner under the original training-CV selection rule. These descriptive validation differences do not establish statistical significance or justify silently changing the selected model after seeing validation results. The 48.6% improvement over a constant median does not demonstrate incremental MAE value over a schedule-aware reference.

The first experiments used a gate against constant baselines and the incumbent. After these diagnostics, promotion was explicitly revised to **`calendar-reference-v2`**: a candidate must improve MAE by at least 1% and have no greater RMSE against **each** of the four fixed references (constant mean/median and weekday/hour mean/median), plus the incumbent when present. MAE remains primary and RMSE acts as a guard against worse large errors. The diagnostic command itself does not register models or change the serving alias. Version 1 remains stored as a legacy model without retrospective certification; the later MAE-aligned champion is described below.

The observed high-consumption band, defined as actual usage above the **training** 75th percentile of 53.97 kWh, contains 144 validation observations. The forest's MAE in this band is **22.10 kWh**, and its average signed error is **-11.61 kWh** (prediction minus actual): it underestimates this band on average. Across all validation observations, its largest absolute error is 80.67 kWh. These cases warrant investigation; calendar inputs alone cannot identify their operational cause.

Each diagnostic run records a machine-readable summary, all validation predictions, the 20 largest champion errors, and error slices by hour, weekday, weekday/weekend, month, date, and observed usage band in the local `steel-energy-diagnostics` MLflow experiment. Every slice includes its observation count, MAE, RMSE, signed bias, 95th-percentile and maximum absolute error. The lower and upper usage cutoffs are fitted on training targets only. Observed usage bands require known targets and are **offline diagnostics**, not prediction inputs or online alert rules. Date groups contain sampled observations, not complete daily energy totals; small groups are descriptive rather than reliable rankings.

Diagnostics verify that the champion's feature contract, training/validation hashes, and training cutoff match the fixed evaluation data. They save the preparation manifest and application source alongside the results, never open the final test partition, and leave both registry versions and the serving alias unchanged.

### MAE-aligned model improvement

The next accepted change aligns forest training with the primary metric. The current search compares `squared_error` and `absolute_error` criteria inside the same chronological training CV. Absolute-error trees use median leaf estimates. Both criteria retain the original depth `[6, None]` and leaf-size `[2, 8]` grid, seed 42, 100 trees, and seven calendar inputs. The full search now contains 17 configurations across the same five model families, including the two constant baselines. No additional source measurements or lagged targets are introduced.

Before inspecting the new candidate's fixed validation results, its training-CV MAE had to improve by at least 1% over the best squared-error control. The selected absolute-error forest reduced CV MAE from **11.45 to 10.87 kWh**, an improvement of approximately **5.1%**. Depth 6 and leaf size 2 were selected by CV; the original promotion thresholds were retained.

| Model | Validation MAE (kWh) | Validation RMSE (kWh) |
|---|---:|---:|
| Previous squared-error champion | 10.5807 | 19.6482 |
| Weekday/hour median reference | 10.3739 | 20.8081 |
| **New absolute-error forest** | **9.9233** | **19.1692** |

The new forest reduces validation MAE by **6.2%** against the previous champion and **4.3%** against the schedule median, while reducing RMSE by **2.4%** against the previous champion. It clears **every** unchanged `calendar-reference-v2` MAE and RMSE check and becomes host registry champion version **7**. Reproducing the same fixed search through the Docker training API also passes every gate and promotes container version **3**, with MAE **9.9241** and RMSE **19.1703 kWh**. These separately fitted models have small platform-dependent numerical differences; their registry version numbers belong to separate stores.

There is an observed error trade-off. The validation 95th-percentile absolute error increases from **44.99 to 48.83 kWh**, while the maximum absolute error decreases from **80.67 to 76.95 kWh**. In the same 144-observation high-usage band, MAE improves from **22.10 to 21.27 kWh**, but signed bias becomes more negative (**-11.61 to -14.37 kWh**). The model passes the pre-existing MAE/RMSE gate and improves average consumption estimates; high-usage underestimation remains a limitation for peak-related decisions. No threshold is changed after seeing these diagnostics.

The training-only screen, all eight forest trials, selected parameters, promotion comparisons, source snapshots, data hashes, and validation predictions are recorded in MLflow. The screen run is linked from the accepted training run. A prior median-schedule residual-correction experiment was rejected at training CV (13.38 versus the forest control's 11.45 kWh); it never progressed to fixed-validation scoring or a registry change.

These are validation improvements following earlier validation diagnostics, rather than a new untouched estimate or a statistical significance claim. The final test partition remains reserved. Calendar-only estimates still cannot explain unplanned operating changes or establish actual energy savings.

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
- The selected candidate's registry version, promotion policy version, and a machine-readable decision. Every selected candidate is retained, including rejected ones. `promotion_decision.json` records every reference's errors, allowed limits, and both pass/fail checks; the first failed check supplies the concise rejection reason.
- Train-only weekday/hour reference predictions, their validation errors, and fallback counts. These deterministic references are mandatory checks, not additional selectable learned candidates.

Ridge uses a square-root target with a square inverse transformation; random forests aggregate non-negative mean or median leaf estimates; boosted trees use a Poisson loss. The workflow checks all outputs for finite, non-negative consumption values. No final test targets are loaded by training or candidate selection.

**Current promotion policy: `calendar-reference-v2`**

1. The learned candidate is still selected only by training CV MAE. Its validation MAE must be at least 1% lower than **each** constant mean, constant median, weekday/hour mean, and weekday/hour median reference. Its RMSE must not exceed any reference's RMSE. Both schedule references are fitted only on training observations with the same fixed backoff as diagnostics; none is selected or dropped based on validation performance.
2. When a champion exists, it is reloaded and re-evaluated on exactly the same current validation observations. The candidate must additionally improve its MAE by at least 1%, with no RMSE degradation. Every reference and incumbent check is recorded, even if another check already rejects the candidate.
3. Feature compatibility and training cutoff provenance must match the evaluation protocol. Comparison fails closed if the incumbent has already trained on the proposed validation period.
4. Only after training, validation, artifact logging, and registration finish can the `champion` alias change. A rejected candidate leaves the current champion in place.
5. The 1% improvement is a policy floor: `--minimum-improvement` may make the requirement stricter but must be at least `0.01` and less than `1`. Exact 1% improvement and equal RMSE pass. A zero-MAE reference cannot be strictly improved, so it prevents promotion. Missing references and invalid errors cause failure rather than a partial comparison.

The original constant-baseline policy was fixed before the first real experiment; version 2 is an explicit revision following validation diagnostics, not a claim that this stronger rule was predeclared. Each new run and candidate version records its policy ID and threshold. Existing untagged champions are reported as `legacy-v1` and remain available until a new candidate passes; the code does not retroactively mark them as version-2 approvals.

Under version 2, the observed schedule-median MAE of 10.3739 kWh requires candidate MAE of at most **10.270161 kWh**, together with all other checks. The original squared-error forest's 10.5807 kWh therefore does not qualify for a new promotion; the MAE-aligned forest's 9.9233 kWh clears this limit and all remaining checks. A first candidate may also be registered without receiving a `champion` alias; subsequent training can retry normally, and prediction reports that no validated champion is available.

The 1% threshold is an operational gate, not a statistical significance claim or a guarantee of future performance. Repeated experimentation must not turn the fixed validation partition into an unlimited tuning set. The untouched final test partition is reserved for the eventual final assessment.

For example, requesting a Ridge-only candidate after the random forest has become champion exercises the rejection path:

```bash
steel-energy train --families ridge
```

Prediction operations record their exact timestamp request, generated features, response, and model version in MLflow. Tracking failures produce an error instead of an unrecorded successful response. Local training uses a workspace lock to prevent overlapping local promotion jobs; distributed deployment coordination remains future work.

### Local training and prediction REST API

Start the service from the project root in the activated environment:

```bash
steel-energy serve
```

The default address is `http://127.0.0.1:8000`. Open [interactive API documentation](http://127.0.0.1:8000/docs) to inspect the contracts and submit prediction or training requests. The server also accepts `--workspace`, `--model-name`, `--host`, `--port`, and `--data` (default `data/processed`, used for training); model names and filesystem paths are server configuration, not request-body inputs.

```bash
curl --fail-with-body http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"timestamps":["2019-01-07T10:00:00","2019-01-07T10:15:00"]}'
```

The dates demonstrate the interface; no 2019 target values are available to assess their accuracy. Successful responses contain the request ID, MLflow run ID, one explicit model version, feature contract, policy at model promotion, units (`kWh`), and one prediction per timestamp in input order. The MAE-aligned champion reports `calendar-reference-v2`; older untagged models disclose `legacy-v1`. Readiness checks compatibility and loadability, so its policy field must also be inspected when assessing approval status.

| Endpoint | Purpose | Normal and failure responses |
|---|---|---|
| `GET /health` | HTTP liveness, independent of model availability | `200`; audit storage failure produces `503` |
| `GET /ready` | Resolve the champion, check its contract, and load the saved model | `200` with model metadata; `503` if unavailable or incompatible |
| `POST /predict` | Predict with one pinned model version and record the result | `200`; `422` for invalid input; `503` for model/tracking/audit failures; `500` for unexpected errors |
| `POST /train` | Start one isolated training job using a fixed snapshot | `202` with job ID and status URL; `409` if training is active; `422` for invalid options; `503` if admission fails |
| `GET /train/{job_id}` | Read job state, final training result, and promotion checks | `200`; `404` for an unknown job; `503` for unavailable status storage |

The JSON body accepts only `timestamps`: a list of 1-1,000 ISO datetime **strings**, without a timezone and on the 15-minute grid. Empty lists, numeric dates, date-only strings, timezone offsets, nonzero seconds/microseconds, and extra fields are rejected before model access. Timestamps are canonicalized for prediction artifacts; duplicates and original input order are retained. Error responses expose a request ID and actionable error category without internal storage paths or stack traces.

Every response receives a server-generated `X-Request-ID`. Application requests, including invalid input and failed predictions, append a locked JSON-lines audit record to `artifacts/mlflow/http_requests.jsonl`: UTC time, request ID, method, path, status, duration, and model/run IDs when available. Request headers and arbitrary raw payloads are not copied into the audit log. Successful predictions save canonical inputs, generated features, and the returned prediction payload in MLflow; the request ID also appears in the MLflow run tags. If HTTP audit storage fails, the application returns `503` rather than an unrecorded success. Such a failed HTTP response can follow a completed MLflow prediction run.

Prediction and readiness use one API worker and serialize model/tracking operations within that process because fluent MLflow configuration and native thread limits are shared state. HTTP validation and liveness remain responsive during inference; integration tests exercise concurrent requests across separate model stores. Training runs in a separate Python process, so it does not take this prediction-operation lock or overwrite the web process's fluent tracking configuration. Distributed coordination and load testing remain later runtime work.

### Submit and inspect a training job

```bash
curl --fail-with-body http://127.0.0.1:8000/train \
  -H 'Content-Type: application/json' \
  -d '{"families":["ridge","random_forest","hist_gradient_boosting"],"forest_trees":100,"minimum_improvement":0.01,"seed":42}'
```

An empty JSON object uses those defaults. The response is `202 Accepted` and contains `job_id`, `training_request_id`, and `status_url`; its `Location` header points to the same status URL. Query that URL, for example `GET /train/<job_id>`, until `status` becomes `succeeded`, `failed`, or `interrupted`. While training runs, status is `running`; percentage progress is not estimated. Each status query has its own request ID while retaining the original training request ID.

The training options are bounded: one to three unique supported families, 1-500 forest trees, an improvement threshold in `[0.01, 1)`, and an integer seed in `[0, 2**32-1]`. Numeric strings, booleans in integer fields, extra fields, client-provided paths, target values, and model names are rejected. The endpoint keeps the existing train-only model selection and `calendar-reference-v2` promotion policy.

`succeeded` means training and result recording completed. The returned `result.promoted` separately reports whether the champion changed; `false` includes the rejection reason and every comparison. Failed workers record an error category and exit code. Detailed worker output stays in the local job directory rather than being exposed as a server traceback.

Admission copies only `train.csv`, `validation.csv`, and `manifest.json` into `artifacts/mlflow/training_jobs/<job_id>/data` and verifies their checksums before spawning the worker. The final test CSV is neither copied nor loaded. This immutable snapshot prevents later preparation runs from changing an admitted job's inputs. Job metadata records the canonical training options, data hashes, and original HTTP request ID before execution; the training run adds matching job/request tags and `http_training_request.json` in MLflow.

A workspace admits one API training job at a time, without an unbounded queue. A cross-process admission lock rejects duplicates across API instances, and admission checks the existing CLI training lock. The child takes the normal training lock as well; if another CLI job wins a later race, that attempt fails instead of overlapping promotion. The current champion can still serve predictions during training.

Job state is atomically persisted in `job.json`, while the worker's result and diagnostic log stay in `result.json` and `training.log`. Completed jobs remain queryable after restarting the API. Graceful shutdown waits for its active worker to finish. A record left by an abruptly stopped API can be reconciled when the recorded worker PID is no longer alive: a valid saved result becomes `succeeded`, otherwise it becomes `interrupted`. This local PID-based recovery assumes a single host and does not provide a distributed queue or strong identity across PID reuse.

HTTP audit records include the training job ID and, after completion, its MLflow run ID. If HTTP audit writing fails after a job has already started, the response is `503` **with the admitted job ID and status URL**; inspect that job rather than assuming training did not start and submitting another job. The durable job record remains the admission record in this case.

### Docker Compose runtime

The portable runtime ZIP contains the project source, the prepared training and validation partitions, and the approved champion export. Extract it into a new directory and run the following from the extracted `steel-energy-mlops` directory with Docker and Compose installed.

```bash
docker compose up --build -d --wait
```

The initial build needs access to the Python base image and dependency packages. The initializer checks the data and model checksums and restores the approved model into a fresh registry. The ZIP supplies the runtime data and model files required by this command. It leaves the final test partition and local experiment history in the development workspace.

For development from a Git checkout, the API and MLflow UI share one application image and a named volume. Restore the prepared data in `data/processed` and export an existing champion before starting. Git tracks the data metadata. Shared DVC/model artifact storage is still pending.

For the current local experiment store, create the portable model export once in the activated Python environment:

```bash
steel-energy export-model
```

This writes `artifacts/deployment/champion`, including the saved pipeline, file checksums, original model/run IDs, feature contract, training/validation hashes, and original promotion policy. Export refuses to overwrite an existing directory. To export a later champion, choose a new parent, for example `steel-energy export-model --output artifacts/deployment-next/champion`, and set `STEEL_ENERGY_MODEL_SEEDS=./artifacts/deployment-next` for Compose. If no champion exists, export fails; a candidate must first qualify through normal training and promotion.

With those prerequisites available, start the container services with one command:

```bash
docker compose up --build -d --wait
```

- [Container API and interactive documentation](http://127.0.0.1:18000/docs)
- [Container MLflow UI](http://127.0.0.1:15001)

The container API exposes the same `/health`, `/ready`, `/predict`, `/train`, and `/train/{job_id}` contracts described above. Ports bind to the local computer. `STEEL_ENERGY_API_PORT` and `STEEL_ENERGY_MLFLOW_PORT` can override their defaults. A second installation on the same computer can use `docker compose -p steel-energy-demo up --build -d --wait` with distinct ports. The Compose project name scopes the image tag, containers, network, and volume.

| Service | Responsibility |
|---|---|
| `volume-init` | One-shot ownership setup for the application volume |
| `initialize` | Verify and copy prepared train/validation files, then restore the exported champion if none exists |
| `api` | Serve predictions and isolated asynchronous training jobs as user 10001 |
| `mlflow` | Display the container experiment store and registry as user 10001 |

Initialization mounts the host's prepared data and model export read-only. It copies only `train.csv`, `validation.csv`, and `manifest.json`; the final test CSV is neither copied nor opened. Copying the existing files preserves the exact hashes used by the model: recomputing trigonometric features on a different operating system can change the last floating-point digit and therefore the CSV checksum. Feature values, file checksums, source cohort size, training cutoff, and model provenance are checked before accepting runtime data. Changed inputs fail initialization without overwriting an existing valid runtime dataset.

The portable export transfers the **existing champion only**, rather than the entire host experiment history. Initialization reloads that fitted pipeline and records a restoration run in the separate container registry, preserving the original promotion policy and source version/run IDs. It does not retrain or award a new policy approval. A legacy export retains `legacy-v1`; `/train` jobs must pass `calendar-reference-v2` to replace it. The MAE-aligned forest has passed that gate through normal training. Runtime registry version numbers belong to the new store and may differ from host version numbers. A separate portable export of host version 7 is available locally at `artifacts/deployment-v7/champion`; choose `STEEL_ENERGY_MODEL_SEEDS=./artifacts/deployment-v7` when initializing a fresh volume from it. An existing volume retains its own champion.

The named volume `steel-energy-mlops_runtime` holds prepared training data, the SQLite database, models, prediction artifacts, HTTP audit records, and training-job results. API and UI use the same `/runtime` path so stored artifact locations remain valid in both containers. Restart initialization validates the data against the existing champion and retains that champion without importing another model version. Host artifacts and the host SQLite store are separate.

```bash
docker compose ps --all
docker compose logs --tail 50 api mlflow initialize
docker compose down
```

`down` removes this project's containers and network while retaining its named volume. Starting again restores access to persisted models and completed jobs. Adding `--volumes` would delete that state. API shutdown allows up to 120 seconds for its worker to finish; an interrupted job can be reconciled from its durable record after startup.

The image uses a digest-pinned Python 3.12 base and the pinned application dependencies. Only the one-shot volume ownership service runs as root. This is a local, single-host SQLite deployment with one API worker and one admitted training job; distributed coordination, a shared artifact store, and the monitoring services remain future work.

**MLflow Projects entry points**

`MLproject` exposes `prepare`, `train`, `predict`, and `evaluate`, sharing the same application functions as the CLI. For example, after the normal environment setup:

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
├── MLproject                # Shared prepare/train/predict/evaluate entry points
├── Dockerfile / .dockerignore # Application image and restricted build context
├── compose.yaml             # Initialization, API, MLflow UI, and persistent volume
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

The tests cover invalid data, chronological isolation, feature leakage, reproducible outputs, CV windows, promotion acceptance/rejection, real SQLite tracking and registry behavior, saved-model round trips, request artifacts, HTTP schemas and errors, audit failures, concurrent prediction isolation, isolated training subprocesses, immutable training snapshots, duplicate admission, persisted job recovery, portable model restoration, initialization idempotency, data/artifact tampering, and failure handling. Prediction integration tests serve saved models after removing the prepared synthetic CSVs. Training API and runtime integration tests remove the final test CSV before execution. Tests use synthetic observations and never train or select models using the real final evaluation data.

## Implementation scope

| Area | Intended implementation | Status |
|---|---|---|
| Data | Source attribution, fixed sample, validation, DVC preparation and provenance | Implemented; shared storage pending |
| Feature/evaluation preparation | Calendar-only predictors and chronological train/validation/test split | Implemented |
| Modeling | Train-only CV, constant and schedule-aware comparisons, validation error slices | Implemented; final test evaluation reserved |
| Experiment tracking | MLflow Projects, artifacts, reloadable models, registry | Implemented locally |
| Model management | Candidate comparison and promotion safeguards | Implemented locally |
| Serving | Version-pinned prediction, isolated training jobs, status queries, request records | Training and prediction APIs implemented locally |
| Monitoring | Evidently reports displayed in Streamlit | Planned |
| Runtime | Docker image, single Compose startup, persistent model/data/job state | Implemented locally for API and MLflow; monitoring services pending |
| Quality | Tests, coverage, Flake8, Pyright, pre-commit, GitHub Actions | Implemented for current application code |
| Optional extension | Airflow orchestration and full-dataset analysis | Planned |

The implemented application must maintain separate 95% coverage gates for source and test code, Pyright standard mode, and Flake8 compliance. Container startup, HTTP training/prediction, and state retention are checked against the running local stack. Monitoring requires its own implementation and validation when added.

## Development workflow

Development uses local feature branches and meaningful commits. Changes are pushed and reviewed in batches when requested or when preparing a submission or shared milestone; routine local iterations do not require GitHub uploads or remote CI runs. Published pull requests are merged into `main` with merge commits so that branch history is retained.

## AI assistance

AI assistance was used for dataset selection, sampling validation, repository organization, application code, model comparison, tracking integration, the prediction service, and tests. Generated changes and technical claims are checked against the source data, recorded experiments, and automated validation before being presented as completed results.
