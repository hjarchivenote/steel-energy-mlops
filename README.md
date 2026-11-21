# Steel Energy MLOps

**Industrial electricity consumption prediction with reproducible model training, serving, and monitoring.**

This project uses measurements from a South Korean steel manufacturing facility to estimate electricity consumption (`Usage_kWh`). It connects my manufacturing experience with practical machine learning engineering: tracking experiments, validating model changes, serving predictions, and monitoring changes in incoming data.

The repository supports an MLOps semester project at Óbuda University and will also serve as a portfolio project. The coursework uses a fixed 5,000-record target sample. A separate offline forecast experiment supplements it with earlier measurements from the original continuous source; it does not expand the target cohort or change its partitions.

> **Status: training, prediction, and monitoring packaged with Docker Compose.** Data preparation, model comparison, MLflow tracking and registry, validation-based promotion, asynchronous training jobs, logged predictions, a persistent container runtime, and version-scoped Evidently reports displayed in Streamlit are implemented locally. Reported model errors below are validation results; final test evaluation remains reserved.

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

MLflow records the input decisions, source hashes and timestamps for the candidate training features. The continuous-history experiment below now compares this input design with the calendar reference and a forecast that repeats the latest available consumption reading. The deployed service continues to use the calendar model.

### Continuous history: frozen offline validation

The fixed 5,000 targets and their train/validation/test boundaries remain unchanged. Original measurements supplement **past context only**. The experiment assumes a 15-minute reporting delay and predicts 15 minutes after the forecast origin, so a normal prediction uses measurements from 30 minutes before its target. Arrival times are simulated, not measured by UCI. Earlier observations from the validation period may supply later forecasts only after their declared arrival; the model is never refitted on validation labels.

The original file places each date's `00:00` row after its `23:45` row. The file does not resolve the measurement-time interpretation. Midnight measurements are therefore excluded from history and fitting; their dates are never shifted. All evaluation targets are retained, and midnight requests use the same calendar-forest fallback for every comparison, with no short-horizon forecasting claim for those requests. Training fits 3,457 non-midnight labels; validation retains all 750 targets, including nine common-fallback targets.

Seven configurations were screened using three expanding windows inside the training partition, purging labels not available at each window's first forecast origin. Before fitting, acceptance required at least 1% lower MAE than **both** the matched calendar forest and latest-usage persistence, no worse RMSE/P95/high-usage MAE, and at least 10% lower absolute high-usage bias than both controls. The high-usage threshold is computed from each training window only. Among eligible candidates, squared-error histogram boosting had the lowest mean CV MAE (7.3297 kWh versus persistence 8.6731). Its configuration was then frozen for one check on the existing validation partition; no validation search was performed.

| Fixed validation, 750 targets | MAE (kWh) | RMSE (kWh) | P95 absolute error (kWh) | High-usage MAE (kWh) | High-usage signed bias (kWh) |
|---|---:|---:|---:|---:|---:|
| Matched calendar forest | 10.0739 | 19.4556 | 49.1754 | 21.8321 | -14.2442 |
| Latest available usage | 7.8716 | 17.3429 | 39.3100 | 20.8091 | -11.3837 |
| Frozen history candidate | **6.7300** | **13.0324** | **29.1749** | **17.3278** | **-3.7196** |

All predeclared checks pass. MAE is 14.5% lower than persistence and 33.2% lower than the matched calendar control. The candidate also improves MAE over persistence in both temporal halves (5.8777 versus 6.5711, and 7.5823 versus 9.1722). High usage means more than the training-only Q75 of 54.22 kWh, covering 140 validation targets. This matched calendar control excludes midnight fitting labels, so its score differs from the deployed calendar model's 9.9241 kWh. These results use additional past measurements and support continuing the forecast work; they do not establish statistical significance, energy savings, or performance under a different arrival delay.

Machine-readable [selection](experiments/history_15min/selection.json) and [validation](experiments/history_15min/validation.json) snapshots preserve the frozen plans, hashes, all screened candidates and acceptance checks. The original source and availability/prediction traces are stored in MLflow experiment `steel-energy-dense-history-screen`: selection run `5fc6a173bd9c4d3b8bc6c2e8b603e030`, validation run `3ff1b50d2cc74358b0ef5889e89039b4`. With the original source and prepared data restored, replay the frozen validation without tuning:

```bash
python experiments/history_15min/replay_validation.py
```

The replay verifies original, training and validation hashes, logs to a separate local store, and never opens the final test CSV or registers a model. Tests use synthetic data to check availability, midnight fallback, future-value isolation, target integrity and rejection of unavailable labels. The running API, champion and monitoring still use the calendar contract. The next implementation step is an explicit forecast request contract with historical observations, arrival-time validation and fallback behavior before any deployment decision. The final test remains unscored.

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

### Prospective protection for large and high-usage errors

Future training jobs use `calendar-reference-v3`. All existing v2 MAE/RMSE comparisons and the minimum 1% MAE improvement remain mandatory. When an incumbent exists, promotion additionally requires no increase in the 95th-percentile absolute error, high-usage MAE, or absolute high-usage signed bias. The high-usage cutoff is the training-target 75th percentile; candidate and incumbent are measured on the same validation observations. The complete comparison is stored as `tail_guard` in the promotion artifact.

Without an incumbent, the tail comparison is explicitly unavailable and the four fixed MAE/RMSE references still govern initial approval. If no validation observations exceed the training cutoff, high-usage metrics are null rather than invented zero-error results; the P95 comparison still applies. These descriptive safeguards do not establish statistical significance or an operational guarantee. Existing host version 7 and container version 3 retain their original v2 approval and performance measurements; they are not retrospectively certified under v3.

Two bounded training-only screens tested 12 loss-aligned boosting configurations and nine calendar-basis/blending candidates. None met the prospectively recorded combined MAE, RMSE, P95, high-usage MAE, and bias requirements. The forest control's mean CV MAE was 10.8655 kWh; the best boosting candidate reached 11.6476 and the best basis/blending candidate reached 10.9892. These are CV results, not new validation scores. No candidate changed the registry or serving model. Frozen plans, trial results, and source are retained in MLflow runs `3416655050184da3b7a106c4e6390e34` and `61a89acf9e15434eaced792999517986` in `steel-energy-peak-candidate-screen`.

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

The training options are bounded: one to three unique supported families, 1-500 forest trees, an improvement threshold in `[0.01, 1)`, and an integer seed in `[0, 2**32-1]`. Numeric strings, booleans in integer fields, extra fields, client-provided paths, target values, and model names are rejected. The endpoint keeps train-only model selection and applies the prospective `calendar-reference-v3` promotion policy described above.

`succeeded` means training and result recording completed. The returned `result.promoted` separately reports whether the champion changed; `false` includes the rejection reason and every comparison. Failed workers record an error category and exit code. Detailed worker output stays in the local job directory rather than being exposed as a server traceback.

Admission copies only `train.csv`, `validation.csv`, and `manifest.json` into `artifacts/mlflow/training_jobs/<job_id>/data` and verifies their checksums before spawning the worker. The final test CSV is neither copied nor loaded. This immutable snapshot prevents later preparation runs from changing an admitted job's inputs. Job metadata records the canonical training options, data hashes, and original HTTP request ID before execution; the training run adds matching job/request tags and `http_training_request.json` in MLflow.

A workspace admits one API training job at a time, without an unbounded queue. A cross-process admission lock rejects duplicates across API instances, and admission checks the existing CLI training lock. The child takes the normal training lock as well; if another CLI job wins a later race, that attempt fails instead of overlapping promotion. The current champion can still serve predictions during training.

Job state is atomically persisted in `job.json`, while the worker's result and diagnostic log stay in `result.json` and `training.log`. Completed jobs remain queryable after restarting the API. Graceful shutdown waits for its active worker to finish. Recovery validates the saved result and checks worker identity: new records use both PID and creation time, while older records are checked against the training CLI command and the exact job's data snapshot path. A genuine worker stays `running` until it exits, including when it has just written its result. With a missing or mismatched worker, a valid result restores `succeeded`; otherwise the job becomes `interrupted`, allowing another job to be admitted. The monitor releases admission after worker exit and before publishing completion. If process information is inaccessible, recovery conservatively retains `running` for inspection. This remains a single-host subprocess workflow.

HTTP audit records include the training job ID and, after completion, its MLflow run ID. If HTTP audit writing fails after a job has already started, the response is `503` **with the admitted job ID and status URL**; inspect that job rather than assuming training did not start and submitting another job. The durable job record remains the admission record in this case.

### Prediction distribution monitoring

Generate a monitoring snapshot from the current champion's recent, successfully logged prediction requests:

```bash
steel-energy monitor --data data/processed --workspace artifacts/mlflow --lookback-hours 168 --max-runs 100
```

For the running Compose stack, use its own data and experiment store:

```bash
docker compose exec -T api steel-energy monitor --data /runtime/data/processed --workspace /runtime/mlflow
```

The operation pins one champion version, verifies its training checksum, and compares the training inputs and that same model's reference predictions with incoming `predictions.csv` artifacts. It selects only completed requests for the pinned model and version inside the request-time window. The default snapshot covers the newest 100 matching runs from the last seven days; `max-runs` is bounded at 1,000 and lookback at 720 hours. The summary identifies the selected run IDs, window, and whether the run limit was reached. Repeated target timestamps count once, retaining the newest request. Malformed artifacts fail the operation rather than being silently excluded.

Both datasets need at least 100 rows after deduplicating current timestamps. Otherwise, the saved status is `insufficient_data`, without an Evidently drift verdict or HTML report. This is a conservative operational guard, not proof of statistical power. With sufficient data, Evidently uses normalized Wasserstein distance for numeric columns and Jensen–Shannon distance for `is_weekend`, with an explicit distance threshold of 0.1. Monitoring copies are rounded to 12 decimal places to prevent floating-point noise from breaking histograms for nearly constant calendar columns; prediction serving and original artifacts are unchanged.

The `steel-energy-monitoring` experiment stores `monitoring_summary.json` and the deduplicated `current_predictions.csv`. Evaluated snapshots also contain `evidently.json` and a standalone `drift_report.html`. A detected change yields `distribution_changed`; otherwise the result is `no_drift_detected`. These refer to request schedules and prediction distributions. Calendar seasonality or deliberately selected queries can cause a signal. Without observed consumption, this operation cannot assess actual prediction error or factory-load changes. It does not retrain, promote a model, or open the validation or final test CSV. Attempts for a compatible champion are recorded before reading input data, so failures while loading prediction artifacts are visible as failed monitoring runs.

### Automatic report refresh

The Compose stack starts a dedicated `monitor` service alongside the API, MLflow, and dashboard. It checks every 60 seconds by default and creates a new snapshot when eligible prediction run IDs, the deployed model, reference-file hashes, or monitoring settings change. Unchanged successful snapshots are reused for up to 24 hours. Expired requests falling outside the rolling window also change the input set. Change checks inspect run metadata and hash the local training CSV and manifest; they do not reload every prediction artifact when the inputs are unchanged.

A workspace-wide `monitoring.lock` serializes manual and automatic reports. Busy calls skip without queuing. A restarted worker closes an abandoned `RUNNING` monitoring attempt as `KILLED` before recovering under that lock. Failed attempts with unchanged inputs wait five minutes before retrying; changed inputs can retry immediately. These decisions use persisted MLflow metadata and survive a worker restart. The service never trains a model or changes the champion alias.

`STEEL_ENERGY_MONITOR_INTERVAL_SECONDS` sets the Compose polling interval, from 1 to 3,600 seconds. The worker handles shutdown signals, records an atomic heartbeat in `monitoring_service.json`, and becomes unhealthy when its last check failed, is waiting to retry a failure, or its heartbeat is stale. The dashboard keeps its read-only mount.

For a one-off change-aware refresh in the activated local environment:

```bash
steel-energy monitor --if-changed --data data/processed --workspace artifacts/mlflow
```

Without `--if-changed`, the existing `monitor` command forces a fresh report after acquiring the same lock. This can be used after repairing an artifact. To run the local worker outside Compose:

```bash
steel-energy watch-monitoring --data data/processed --workspace artifacts/mlflow --interval-seconds 60
```

This is a single-host polling service using the shared workspace, not a distributed scheduler. Saved prediction artifacts are treated as immutable inputs identified by their run IDs. There is no automatic retraining based on a drift signal.

### Monitoring dashboard

The Compose stack includes a Streamlit dashboard at [http://127.0.0.1:18501](http://127.0.0.1:18501). It reads the latest monitoring attempt for the deployed model version and training checksum. The runtime volume is mounted read-only in this container. Viewing or refreshing the page does not generate a report, train a model, or change the registry.

The dashboard shows unique prediction times, the required sample size, reference rows, removed duplicates, and the saved snapshot time. With sufficient data, it displays per-column distances and the saved Evidently HTML report. Insufficient data, missing results, running or failed attempts, and storage errors have distinct messages. A latest failed or unfinished attempt hides earlier results; incompatible or inconsistent saved summaries are rejected. Snapshots older than 24 hours and snapshots reaching the request limit carry warnings. A lack of detected drift is not displayed as proof of prediction quality.

After collecting predictions, the monitor service updates the saved snapshot on its next check. Select **Refresh saved results** to load it; this button does not itself run monitoring. The manual `monitor` command remains available. `STEEL_ENERGY_DASHBOARD_PORT` changes the local Compose port. For development outside Docker:

```bash
STEEL_ENERGY_WORKSPACE=artifacts/mlflow streamlit run dashboard/app.py --browser.gatherUsageStats=false
```

`dashboard/app.py` is a thin entry point; result loading, validation, and presentation live in `src/steel_energy_mlops/dashboard.py`. `STEEL_ENERGY_MODEL_NAME` optionally selects the registered model name for this local view.

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
- [Container monitoring dashboard](http://127.0.0.1:18501)

The container API exposes the same `/health`, `/ready`, `/predict`, `/train`, and `/train/{job_id}` contracts described above. Ports bind to the local computer. `STEEL_ENERGY_API_PORT` and `STEEL_ENERGY_MLFLOW_PORT` can override their defaults. A second installation on the same computer can use `docker compose -p steel-energy-demo up --build -d --wait` with distinct ports. The Compose project name scopes the image tag, containers, network, and volume.

| Service | Responsibility |
|---|---|
| `volume-init` | One-shot ownership setup for the application volume |
| `initialize` | Verify and copy prepared train/validation files, then restore the exported champion if none exists |
| `api` | Serve predictions and isolated asynchronous training jobs as user 10001 |
| `mlflow` | Display the container experiment store and registry as user 10001 |

Initialization mounts the host's prepared data and model export read-only. It copies only `train.csv`, `validation.csv`, and `manifest.json`; the final test CSV is neither copied nor opened. Copying the existing files preserves the exact hashes used by the model: recomputing trigonometric features on a different operating system can change the last floating-point digit and therefore the CSV checksum. Feature values, file checksums, source cohort size, training cutoff, and model provenance are checked before accepting runtime data. Changed inputs fail initialization without overwriting an existing valid runtime dataset.

The portable export transfers the **existing champion only**, rather than the entire host experiment history. Initialization reloads that fitted pipeline and records a restoration run in the separate container registry, preserving the original promotion policy and source version/run IDs. It does not retrain or award a new policy approval. Legacy and v2 exports retain their original policy; new `/train` jobs must pass `calendar-reference-v3` to replace the incumbent. Runtime registry version numbers belong to the new store and may differ from host version numbers. A separate portable export of host version 7 is available locally at `artifacts/deployment-v7/champion`; choose `STEEL_ENERGY_MODEL_SEEDS=./artifacts/deployment-v7` when initializing a fresh volume from it. An existing volume retains its own champion.

The named volume `steel-energy-mlops_runtime` holds prepared training data, the SQLite database, models, prediction artifacts, HTTP audit records, and training-job results. API and UI use the same `/runtime` path so stored artifact locations remain valid in both containers. Restart initialization validates the data against the existing champion and retains that champion without importing another model version. Host artifacts and the host SQLite store are separate.

```bash
docker compose ps --all
docker compose logs --tail 50 api mlflow initialize
docker compose down
```

`down` removes this project's containers and network while retaining its named volume. Starting again restores access to persisted models and completed jobs. Adding `--volumes` would delete that state. API shutdown allows up to 120 seconds for its worker to finish; an interrupted job can be reconciled from its durable record after startup.

The image uses a digest-pinned Python 3.12 base and the pinned application dependencies. Only the one-shot volume ownership service runs as root. This is a local, single-host SQLite deployment with one API worker, one admitted training job, automatic monitoring, and a read-only dashboard; distributed coordination and a shared artifact store remain future work.

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
├── dashboard/               # Thin Streamlit entry point
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

The Airflow and screenshot directories remain placeholders. Submission PDFs and ZIP archives are kept locally in the ignored `submission/` directory.

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

Changes to runtime source, Docker configuration, dependencies, or the integration-test fixtures also trigger a Compose smoke test in pre-commit. GitHub Actions runs it through the same hooks. The test generates 5,000 synthetic observations, trains and exports a Ridge model, builds the application image, and starts the API and MLflow with a new volume and dynamically assigned local ports. It verifies model readiness, non-root execution, prediction values, recorded prediction artifacts, Evidently HTML generation from 100 HTTP prediction rows, and completion of a training request that retains the incumbent. It also restarts the API with a completed legacy job left running under an unrelated PID, verifies recovery, and admits another training request. The test cleans up its project containers, volume, network, and image.

Run this check directly with Docker and Compose available in the activated development environment.

```bash
python -m coverage run --append -m pytest tests/test_docker_smoke.py --run-docker -q
```

Ordinary pytest runs skip this integration test until `--run-docker` is supplied. The hook appends its execution to the unit-test coverage before checking both coverage gates. `STEEL_ENERGY_PREPARED_DATA` selects the prepared-data directory for the generated fixture.

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
| Monitoring | Version-scoped Evidently reports from prediction artifacts | Automatic refresh and Streamlit presentation implemented locally |
| Runtime | Docker image, single Compose startup, persistent model/data/job state | Implemented locally for API, MLflow, monitoring worker, and Streamlit |
| Quality | Tests, coverage, Flake8, Pyright, pre-commit, GitHub Actions | Implemented for current application code |
| Optional extension | Airflow orchestration and full-dataset analysis | Planned |

The implemented application must maintain separate 95% coverage gates for source and test code, Pyright standard mode, and Flake8 compliance. Container startup, HTTP training/prediction, and state retention are checked against the running local stack. Streamlit AppTest covers dashboard states and refresh behavior; the Compose smoke test checks the dashboard with a saved Evidently report and verifies its read-only runtime mount.

## Development workflow

Development uses local feature branches and meaningful commits. Changes are pushed and reviewed in batches when requested or when preparing a submission or shared milestone; routine local iterations do not require GitHub uploads or remote CI runs. Published pull requests are merged into `main` with merge commits so that branch history is retained.

## AI assistance

AI assistance was used for dataset selection, sampling validation, repository organization, application code, model comparison, tracking integration, the prediction service, and tests. Generated changes and technical claims are checked against the source data, recorded experiments, and automated validation before being presented as completed results.
