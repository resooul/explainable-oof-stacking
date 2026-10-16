# Explainable Out-of-Fold Stacking with Nested Temporal Validation

Reference implementation of the stacking methodology described in

> R. Özdemir, M. Taşyürek, V. Aslantaş. *Leveraging an Explainable Stacking Ensemble Model for Near-Real-Time
> Decision Support in Same-Day Delivery Shipments.* Manuscript under review.

The repository reproduces every analysis of the article on the public
[NYC Taxi Trip Duration](https://www.kaggle.com/competitions/nyc-taxi-trip-duration) dataset. The proprietary
delivery dataset used in the article cannot be redistributed; the identical pipeline is applied to it through the
generic dataset adapter, and its configuration is published in `configs/ssd.yaml`. Both experiments of the article
were run with version 1.4.2 under the same tuning and stacking protocol (see [Reproducing the article](#reproducing-the-article)).

## Methodology

```
raw data ─► cleaning & feature engineering ─► chronological holdout (train │ test)
                                                        │
            ┌───────────────────────────────────────────┘
            ▼  training partition only
  Level-0: K temporal blocks ─► nested HPO per block ─► out-of-fold predictions
           (blocked K-fold; forward-chaining as sensitivity analysis)
            │
            ├─► top-k base learners ranked on out-of-fold error
            ▼
  Level-1: EN_AVG · EN_WAVG · EN_LIN · EN_LGBM · EN_XGB · EN_CAT fitted on out-of-fold predictions
            │
            ▼  test partition, used once
  evaluation · paired statistics · target bands · periods · seeds · rolling origin · SHAP · profiling
```

1. **Chronological holdout.** The most recent `split.test_fraction` of the data forms the test partition. Cut points
   are aligned with group boundaries (`dataset.group_col`), so that correlated rows, such as the pickup and delivery
   records of one order, never fall on both sides of a split.
2. **Out-of-fold generation.** The training partition is cut into K contiguous temporal blocks. Under the *blocked*
   scheme each block is predicted by a model fitted on the remaining blocks; under the *forward-chaining* scheme
   each block is predicted by a model fitted on earlier blocks only, which removes any look-ahead through calendar
   features. An optional purge gap (`split.gap`) removes training rows adjacent to the held-out block.
3. **Nested hyperparameter optimisation.** For every out-of-fold model, and for the final refit, hyperparameters
   are selected by Optuna (TPE) on a chronological inner split of that model's own training rows. Neither the
   held-out block nor the test partition influences the selection. Under the blocked scheme the training rows of
   an interior fold contain a gap where the held-out block lies; the inner split is therefore drawn from the most
   recent *contiguous* run of training rows (`tuning.inner_split: contiguous`), so that its fitting and validation
   rows are adjacent in time and never straddle the gap. Within that run the validation rows are the most recent
   ones and the fitting rows immediately precede them (`tuning.val_position: latest`), and every search draws its
   own TPE start-up configurations (`tuning.sampler_seed: per_split`). Where and when every inner split lies is recorded in
   `tables/hyperparameters.csv`. `tuning.max_rows` caps the inner fitting and validation rows together;
   `tuning.max_val_rows` sets a separate validation cap, so that a longer validation period can be used without
   enlarging the fitting set. Boosting rounds and epochs are determined by early
   stopping on the inner validation set; boosting rounds are rescaled to the size of the refit sample.
4. **Base-learner selection** ranks all candidates by out-of-fold error on the training partition.
5. **Level-1 combiners** are fitted on out-of-fold predictions: simple averaging (`EN_AVG`), convex weighted
   averaging (`EN_WAVG`), non-negative linear stacking (`EN_LIN`), unconstrained least-squares stacking (`EN_SLR`)
   and three gradient-boosting meta-learners (`EN_LGBM`, `EN_XGB`, `EN_CAT`). The article compares six of them;
   `EN_SLR` is available but is not listed in `stacking.meta_learners` of the published configurations. The ensemble used for explanation
   and profiling is chosen on the meta-dataset, never by test performance. When the forward-chaining scheme is
   computed, the choice is made on its out-of-fold predictions (`stacking.combiner_selection: auto`), whose
   meta-rows were produced by models that never saw a later row; the chosen combiner is deployed as fitted on the
   primary scheme. By default
   (`stacking.meta_selection: rolling`) the most recent `k/(k+1)` of the chronologically ordered meta-dataset is
   cut into `k = meta_selection_windows` consecutive windows; for each window every combiner is fitted, and the
   gradient-boosting meta-learners are tuned, on the earlier rows only, and the selection score is the mean over
   the windows, so the choice does not hinge on one period. `meta_selection: holdout` uses a single hold-out, the
   most recent `meta_val_fraction` (the behaviour of version 1.1). Per-window scores are reported in
   `tables/combiner_selection_windows.csv` and summarised in `tables/combiner_selection.csv`.
6. **Evaluation** on the test partition reports RMSE, MAE, MedAE, MAPE, SMAPE, R² and NRMSE, together with paired
   Wilcoxon signed-rank tests (Holm-corrected), cluster-bootstrap confidence intervals of MAE and RMSE differences
   (resampling calendar days, which respects temporal dependence), paired Cohen's *d_z*, and performance by target
   band (stratified by the actual and, to control for regression to the mean, by the predicted value), by calendar
   period, by stratum (e.g. pickup and delivery phase, `dataset.strata_col`) and by the training-partition
   frequency of every categorical level (unseen, rare, common; the unseen group is the cold-start case). Besides
   the out-of-fold reference, further comparators can be named in `statistics.extra_references`.
7. **Robustness**: repeated fits with several random seeds (hyperparameters held fixed), comparison of the two
   out-of-fold schemes, and a rolling-origin evaluation that is separate from the main experiment. The most recent
   `rolling.n_origins × rolling.test_fraction` of the data is divided into consecutive test windows; at every origin
   the complete pipeline (tuning, out-of-fold generation, combiners) is rebuilt from the preceding data only. By
   default (`rolling.base_learners: selected`) the model families selected in the main experiment are re-tuned and
   refitted at every origin; `all` repeats the selection among every candidate at every origin; an explicit list
   adds the named families to the selected ones and repeats top-k selection over the enlarged set (the article uses
   `[LGBM, XGB, CAT, FTT]`, so that CAT and FTT are rebuilt at every origin). `tables/rolling_selection.csv` reports
   which learners were chosen. The comparator is fixed across origins
   (`rolling.reference: main`, the reference learner of the main experiment), and `tables/rolling_comparison.csv`
   additionally compares the protocol-selected ensemble with the best single base learner in hindsight at every
   origin. A one-at-a-time hyperparameter sensitivity analysis (`sensitivity`) moves each hyperparameter of the
   selected base learners away from its chosen value, re-determines the number of boosting rounds on the same
   inner split, refits and evaluates; it is descriptive and plays no role in any selection.
8. **Explainability** at three levels: exact TreeSHAP of the gradient-boosting base learners on the original
   features; attribution of each meta-learner's output to the base predictions; and model-agnostic permutation SHAP
   of the complete pipeline (raw features → base learners → meta-learner). Agreement between proxy and composite
   explanations is quantified on identical instances (Spearman ρ, Kendall τ, top-k Jaccard), and the stability of
   importance rankings across seeds is reported.
9. **Inference profiling** runs in a fresh process: single-record latency per stage and end-to-end (mean, p50, p95,
   p99) and of every Level-1 combiner, request throughput with threads sharing one process (`profiling.workers`)
   and with separate processes that each load their own copy of the model (`profiling.process_workers`), batch
   throughput, resident and peak memory, and serialised model size.

### Leakage safeguards

| Risk | Safeguard |
|---|---|
| Test information in modelling decisions | Test partition is isolated before tuning, selection, band thresholds and SHAP background sampling |
| In-sample meta-features | Level-1 is fitted exclusively on out-of-fold predictions |
| Model-selection leakage across folds | Hyperparameters are tuned inside each fold (`tuning.mode: nested`) |
| Inner tuning split across the held-out block | Inner split drawn from the most recent contiguous run (`tuning.inner_split`) |
| Combiner tuned on its own selection rows | Meta-learners tuned on the fitting part of the meta-dataset only |
| Combiner selection informed by later rows | Selection on forward-chaining out-of-fold predictions (`stacking.combiner_selection`) |
| Temporal look-ahead | Chronological holdout; forward-chaining scheme; optional purge gap |
| Correlated rows across splits | Group-aligned cut points |
| Preprocessing leakage | Encoders, imputers, scalers and quantile transforms are fitted within every training fold |
| Unseen categories | Vocabularies are fitted on training rows; unseen levels are treated as missing |

### Base learners

| Key | Model | Implementation |
|---|---|---|
| `LGBM`, `XGB`, `CAT` | LightGBM, XGBoost, CatBoost with native categorical handling | official libraries |
| `RF`, `DT` | Random forest, decision tree | scikit-learn |
| `OLS`, `RIDGE` | Linear and ridge regression (one-hot encoding, standardisation) | scikit-learn |
| `EBM` | Explainable Boosting Machine (GA²M) | `interpret` |
| `MLP`, `RESNET` | Multilayer perceptron, tabular ResNet | `rtdl_revisiting_models` (Gorishniy et al., 2021) |
| `FTT` | FT-Transformer | `rtdl_revisiting_models` (Gorishniy et al., 2021) |
| `TABNET` | TabNet | `pytorch-tabnet` (Arik & Pfister, 2021) |
| `CNN`, `LSTM` | 1-D convolutional and recurrent networks over the encoded feature vector | PyTorch |

All deep learners share one preprocessing scheme (median imputation, quantile-normal transformation, categorical
embeddings, standardised target) and one early-stopped AdamW training loop (TabNet uses its reference training
loop with the same batch size, epoch budget and patience), so that differences reflect architecture rather than
training protocol. Search spaces are printed by `python -m stackxai spaces` and exported to
`tables/search_spaces.csv`.

## Installation

Python ≥ 3.10.

```bash
git clone https://github.com/resooul/explainable-oof-stacking.git
cd explainable-oof-stacking
pip install -e ".[all,dev]"      # core, deep learners, EBM and tests
# pip install -e .               # core only: tree-based and linear learners
```

For GPU training of the deep learners, install the CUDA build of PyTorch first (see pytorch.org); the device is
selected automatically (`tuning.deep.device`).

## Data

```bash
kaggle competitions download -c nyc-taxi-trip-duration
unzip nyc-taxi-trip-duration.zip -d data/raw/nyc && unzip data/raw/nyc/train.zip -d data/raw/nyc
```

The pipeline expects `data/raw/nyc/train.csv`. Cleaning rules (duration bounds, bounding box, passenger count,
minimum distance) are set in `dataset.cleaning`, and every removal step is logged in `tables/data_flow.csv`.
Spatial separation is represented by the great-circle (haversine) distance, complemented by a Manhattan
approximation and the bearing; no external routing service is required. Additional pre-computed features can be
merged through `dataset.external_features` (`path`, `key`, `columns`).

## Usage

```bash
python -m stackxai run --config configs/nyc_smoke.yaml     # subsample, pipeline check in minutes
python -m stackxai run --config configs/nyc.yaml           # full NYC experiment -> results/nyc/
python -m stackxai run --config configs/nyc.yaml configs/<other>.yaml # several datasets, in order
python -m stackxai run --config configs/nyc.yaml --set run.n_jobs=16 tuning.mode=single
python -m stackxai profile --run-dir results/nyc           # re-run inference profiling only
python -m stackxai plots --run-dir results/nyc             # redraw figures from stored results (no recomputation)
```

### Reproducing the article

The results of both experiments in the article were produced with version 1.4.2, `configs/nyc.yaml` for the NYC
data and `configs/ssd.yaml` for the delivery data. The two configurations follow the same tuning and stacking
protocol: the inner validation rows are the most recent 90,000 rows of each training segment
(`tuning.val_position: latest`, `tuning.max_val_rows: 90000`), every hyperparameter search draws its own sampler seed
(`tuning.sampler_seed: per_split`), and the combiner is chosen on the forward-chaining out-of-fold predictions
(`stacking.combiner_selection: forward`) and deployed as fitted on the blocked ones. Apart from the dataset section,
they differ only in the target transformation (log scale with smearing for the delivery data, none for NYC), the
reporting period (months or weeks), the extra comparators and the feature labels used in the figures. The package versions of every run are recorded in
`environment.json`.

Earlier NYC results obtained with version 1.3, which used a 30,000-row inner validation set after a cut at 80% of
each segment, one shared sampler seed and combiner selection on the blocked out-of-fold predictions, are not
reported in the article. These settings remain available (`tuning.val_position: after_cut`,
`tuning.sampler_seed: shared`, `stacking.combiner_selection: blocked`) but are not used by any published
configuration.

### Running in stages

The deep-learning base learners dominate the cost and use the GPU, whereas the other learners use the CPU. The
experiment can therefore be split without changing its design:

```yaml
# configs/nyc_ml.yaml (stage 1a: tree-based, linear and GAM learners)
extends: nyc.yaml
run: {stop_after: base_learners}
base_learners: [LGBM, XGB, CAT, RF, DT, OLS, RIDGE, EBM]

# configs/nyc_dl.yaml (stage 1b: deep learners; may run in parallel with stage 1a)
extends: nyc.yaml
run: {stop_after: base_learners}
base_learners: [MLP, RESNET, CNN, LSTM, FTT, TABNET]
```

```bash
python -m stackxai run --config configs/nyc_ml.yaml    # stage 1a
python -m stackxai run --config configs/nyc_dl.yaml    # stage 1b
python -m stackxai run --config configs/nyc.yaml       # stage 2: all learners restored, then the full analysis
```

The two stage files are not part of the repository and can be created as shown. They inherit every setting from
`nyc.yaml` (`extends: nyc.yaml`) and differ only in the list of base learners and in
`run.stop_after: base_learners`. Because checkpoints do not depend on which other
learners are listed, the final run restores all fourteen base learners and performs top-k selection over the
complete candidate set, exactly as a single run would; its summary reports the number of restored units. Settings
that affect training must not be changed between the stages, otherwise the affected learners are retrained. When
both stages share one machine, `--set run.n_jobs=4` for the deep stage leaves the remaining cores to the other.

Each configuration writes to its own directory `<run.output_dir>/<run.name>/` (by default `results/nyc/` for
`configs/nyc.yaml`); `--set run.name=...` keeps variants of one dataset apart.

### Checkpointing and resumption

Model names in the configuration are validated when it is loaded, so a typing error stops the run before any
computation. Base learners are run in the order given by `base_learners`. The work of every learner is split into units (final
hyperparameter search, one unit per out-of-fold split, final refit), and each unit is checkpointed under
`results/<name>/cache/` as soon as it finishes. Re-running the same command therefore skips completed learners and
resumes an interrupted learner at its first unfinished unit; the log reports how many units were restored. Results of
a resumed run are identical to those of an uninterrupted run.

Each checkpoint is keyed by a fingerprint of exactly the inputs that affect it (data content, cleaning and split
settings, tuning settings, the learner's own trial budget, OOF scheme, seed and row ranges). Consequently, adding a
learner computes only that learner, changing one learner's budget recomputes only that learner, and changing
evaluation, explanation or profiling settings reuses all trained models. Checkpoints are written atomically, so an
interruption never leaves a corrupt file. `--no-resume` recomputes and overwrites every unit. With
`run.continue_on_error: true`, a learner that fails (for instance, a missing optional dependency or insufficient
memory) is excluded and listed in `tables/failed_learners.csv` instead of stopping the run.

## Evaluation criteria

| Criterion | Definition | Role |
|---|---|---|
| RMSE | root mean squared error | primary metric (tuning, base-learner ranking, meta selection); configurable |
| MAE, MedAE | mean and median absolute error | robustness to large errors |
| MAPE, SMAPE | mean and symmetric mean absolute percentage error (%) | relative error |
| R² | coefficient of determination | explained variance |
| NRMSE | RMSE divided by the mean target of the evaluated subset | scale-independent comparison across partitions, bands and origins |
| Wilcoxon signed-rank | paired test on absolute errors, Holm-corrected | significance against the reference base learner |
| ΔMAE, ΔRMSE with 95% CI | cluster bootstrap over calendar days | magnitude and uncertainty of the improvement |
| Cohen's *d_z* | standardised mean of paired absolute-error differences | effect size |
| Latency, throughput, memory | p50/p95/p99 per stage and end-to-end; requests/s; MB | deployability |

`tuning.metric`, `stacking.selection_metric` and `evaluation.primary_metric` accept `rmse`, `mae`, `mape`, `smape`
or `r2`.

## Outputs

Every table is written as CSV and as a LaTeX (booktabs) fragment; figures as PNG and PDF. `SUMMARY.md` collects
the main tables.

| File (`tables/`) | Content |
|---|---|
| `data_flow`, `target_distribution`, `features`, `categorical_levels`, `categorical_prevalence` | Cleaning flow; partition-wise target distribution, missingness and binary prevalence; category concentration and overlap; per-level prevalence |
| `folds`, `rolling_windows` | Temporal boundaries of every split |
| `base_learners_oof_vs_test`, `oof_by_fold` | Out-of-fold (training) versus test performance of every base learner; per held-out block |
| `test_performance`, `ensemble_weights`, `oof_correlation` | Main comparison; learned combiner weights; diversity of the base learners |
| `statistical_tests` | Wilcoxon (Holm), bootstrap CIs, Cohen's *d_z* and, separately for MAE and RMSE, whether the CI excludes zero and the direction of the effect (`inconclusive` when it does not), per stratum |
| `combiner_selection`, `combiner_selection_windows` | Combiner selection score per window, its mean (the criterion), spread and ranks |
| `duration_bands`, `performance_by_period` | Performance by target band and by calendar period |
| `performance_by_stratum`, `stratum_sizes` | Performance per stratum (e.g. phase) and stratum sizes |
| `performance_by_category_frequency` | Performance by training frequency of each categorical level (cold start) |
| `oof_scheme_comparison`, `seed_runs`, `seed_robustness`, `rolling_origin`, `rolling_origin_summary`, `rolling_comparison`, `rolling_selection` | Robustness analyses |
| `hyperparameter_sensitivity`, `hyperparameter_sensitivity_summary` | One-at-a-time hyperparameter sensitivity |
| `hyperparameters`, `search_spaces`, `training_cost` | Selected hyperparameters for every nested split; search spaces; wall-clock cost |
| `shap_feature_importance`, `shap_meta_level`, `shap_agreement`, `shap_seed_stability` | Explainability |
| `profiling_latency`, `profiling_meta`, `profiling_throughput`, `profiling_memory` | Inference cost |

`predictions/` stores test and out-of-fold predictions, `models/pipeline.joblib` the deployable ensemble, and
`config.resolved.yaml` and `environment.json` (package versions, git commit, configuration hash) document the run.

## Computational budget

Cost is controlled by `tuning.n_trials`, `tuning.max_rows` (tuning uses the most recent rows of each inner training
set), early stopping, and `tuning.mode`. When a model tuned on the inner sample is refitted on a larger training
set, its early-stopped number of boosting rounds is multiplied by the square root of the sample-size ratio
(`tuning.refit_scaling: sqrt`; `linear` and `none` are alternatives) and capped at `tuning.max_estimators`, since
the optimal number of rounds grows sub-linearly with the amount of data; epochs of deep learners are kept as
found by early stopping.

| `tuning.mode` | Hyperparameter selection | Searches per learner |
|---|---|---|
| `nested` | separately inside every out-of-fold split and for the final refit | K + 1 |
| `single` | once on the training partition, reused for all folds | 1 |
| `none` | library defaults, early stopping only | 0 |

The configuration assigns the same number of trials to every model family, so that differences between families
are not an artefact of unequal tuning effort. Warm-starting from the preceding fold is applied only under the
forward-chaining scheme, where it cannot transfer
information about the held-out block. Deep learners are markedly slower on CPU; a GPU is recommended when they are
included.

## Adapting to another dataset

**Target transformation.** `dataset.target_transform: log` trains, tunes and selects every model on the log
of the target in its original unit (before `target_scale`; e.g. log minutes), while every metric, table and
exported prediction is on the report scale (`target_unit`, e.g. hours). Because exp() of a log-scale squared-error
prediction is a conditional median, predictions are bias-corrected with Duan's smearing factor, estimated from
out-of-sample residuals on the training partition only (out-of-fold predictions for base learners,
selection-window predictions for combiners) and, by default, separately per stratum
(`dataset.target_smearing: stratum | global | none`); the factors are reported in `tables/target_smearing.csv`.
The deployable `models/pipeline.joblib` returns report-scale predictions (`predict`) and model-scale ones
(`predict_model`); SHAP explanations are on the model scale, where exp(SHAP value) is a multiplicative effect.

Features can be removed from any adapter's feature set with `dataset.drop_features`, for example the calendar
features whose test values lie outside the training range. A pre-engineered table requires only a configuration;
`configs/ssd.yaml` is a complete example for the generic adapter. Configurations of
non-public datasets can be kept next to the public ones as `configs/<name>.local.yaml`; such files are excluded from
version control by `.gitignore`.

```yaml
dataset:
  adapter: generic
  path: data/processed/my_table.parquet
  time_col: created_at
  target_col: duration_min
  categorical: [carrier, city]
  group_col: order_id     # optional
  strata_col: phase       # optional
```

Requirements for the table (CSV or Parquet):

| Column | Requirement |
|---|---|
| `time_col` | parseable timestamp of the moment at which the prediction is made; defines order and splits |
| `target_col` | numeric, strictly after `time_col`; rescaled by `target_scale` |
| features | numeric or boolean, or listed in `categorical` (any dtype); missing values allowed |
| `group_col` | optional; rows sharing a value are kept on the same side of every split |
| `strata_col` | optional; may also be a feature; tests are additionally reported per stratum |

Every column that is not the time, target or group column and not listed in `exclude` becomes a feature, unless
`features` is given explicitly. Non-numeric columns that are neither declared categorical nor excluded raise an
error, so that identifiers or timestamps cannot enter the model silently. Columns known only after the prediction
time (outcome timestamps, final statuses, anything derived from the target) must be listed in `exclude`.
`dataset.calendar` derives hour, weekday, day, week, month and weekend indicators from `time_col`, with an optional
holiday file and configurable weekend days; `dataset.geo` derives the haversine distance (and optionally the
Manhattan distance and bearing) from two coordinate pairs, treating zero coordinates as missing.

By default the most recent `split.test_fraction` of the rows is held out, as for NYC. Alternatively, training and
test data are supplied as separate files through `dataset.test_path`. Both files then pass through identical
processing, the test file is used only for the final evaluation, groups occurring in both files are rejected, and
the log and `SUMMARY.md` report whether the test period lies strictly after the training period:

```yaml
dataset:
  path: data/processed/train.parquet
  test_path: data/processed/test.parquet
```

Features and target may also be stored in separate files (`X_train`, `y_train`, `X_test`, `y_test`). The target
files are attached by row order after a row-count check, or, more safely, by an identifier column (`join_key`)
that must be unique in both files. The feature files must still contain `time_col` and, if used, `group_col`,
because chronological out-of-fold splitting, the rolling-origin evaluation and the day-level bootstrap depend on
them. Any transformation already applied to the feature files must be row-wise (for example, distances or calendar
indicators); statistics estimated on the pooled data, such as scalers or target encodings, would leak test
information and must instead be left to the pipeline, which fits them within every training fold.

```yaml
dataset:
  path: data/processed/X_train.parquet
  target_path: data/processed/y_train.parquet
  test_path: data/processed/X_test.parquet
  test_target_path: data/processed/y_test.parquet
  join_key: row_id            # optional; omit to align by row order
```

Custom cleaning and feature engineering are added by registering an adapter that returns a `Dataset`:

```python
from stackxai.data import Dataset, register, temporal_order

@register("my_data")
def load_my_data(cfg):
    frame = ...  # read, clean and engineer features
    frame = temporal_order(frame, "created_at", None)
    return Dataset(frame=frame, features=[...], categorical=[...], target="duration_min", time_col="created_at")
```

## Tests

```bash
pytest
```

The suite verifies chronology, disjointness and group integrity of all splits and the purge gap; that inner tuning
splits never straddle a held-out block; that out-of-fold predictions are never in-sample; that tree meta-learners
are tuned without the rows used to select the combiner; that categorical vocabularies are fitted on training rows
only; and runs the complete pipeline end-to-end on synthetic data with the schema of the NYC file and on a
synthetic grouped, stratified table with the generic adapter.

## Citation

See `CITATION.cff`.

## License

MIT, see `LICENSE`.
