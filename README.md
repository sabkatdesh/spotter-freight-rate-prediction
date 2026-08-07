# Spotter — Freight Rate Prediction

Machine Learning Engineer take-home assessment for Spotter. The task: predict `posted_rate` (freight shipment cost) for unseen loads, and produce a 31-day rate forecast for a fixed December lane, given a schema mismatch between the two target files.

**Full write-up:** [`Spotter_Freight_Rate_Model_Report.pdf`](./Spotter_Freight_Rate_Model_Report.pdf)
**Notebook:** [`notebooks/spotter-freight-rate-prediction.ipynb`](./notebooks/spotter-freight-rate-prediction.ipynb)
**Assessment brief:** `Freight_Rate_ML_Assessment.pdf` (provided by Spotter, not included in this repo)

---

## Problem

Two deliverables were required from the same underlying data:

1. **Point predictions** for 12,000 unlabeled loads in `validation.csv` (full 14-column schema, including live market signals).
2. **A 31-day forecast** for a single fixed lane (Lexington → Fort Wayne, Dry Van) in `december_chart_inputs.csv` — which carries only **7 columns** and is missing `market_index`, `quote_signal`, and all four `*_lat`/`*_lon` coordinate columns, because those signals aren't observable 31 days into the future.

This schema mismatch is the central design constraint of the project (see [Approach](#approach) below).

## Results

| Metric | Head A (full features) | Head B (no market signals) |
|---|---|---|
| MAPE | **5.20%** | 5.50% |
| RMSE | 653.25 | 653.83 |
| MAE | 120.61 | 124.19 |
| R² | 0.8174 | 0.8170 |

Rolling-origin cross-validation across three independent monthly holdouts (Aug/Sep/Oct 2025): **mean MAPE 5.61% ± 0.75pp**, confirming the result is stable over time rather than a lucky single split.

LightGBM was selected after head-to-head testing against CatBoost (7.74% MAPE), XGBoost (9.02%), and a CatBoost+LightGBM ensemble (6.02%) on the identical holdout — see Section 4 of the report for the full comparison and baseline ablation (distance-only linear regression, lane-mean lookup, etc.).

## Approach

**Two-headed LightGBM architecture**, rather than one model forced to handle missing columns:

- **Head A** (37 features) — trained on the full feature set, including `market_index`, `quote_signal`, and their interaction terms. Used exclusively to generate `validation_predictions.csv`.
- **Head B** (33 features) — identical rows, market-derived columns dropped entirely. Used exclusively for the December forecast, since those signals don't exist in `december_chart_inputs.csv`.

Both heads share the same cleaning pipeline, feature engineering, and time-based train/holdout split (train on data before October 2025, evaluate on October 2025 — never a random shuffle, to avoid leaking future market conditions into training).

**Feature engineering** includes: haversine distance & bearing, circuity ratio, distance/weight tiers, calendar features (day-of-week, week-of-year, days-to-nearest-US-holiday, season), leak-safe shrinkage-encoded lane/origin/destination historical rate statistics (fit only on the training fold), a coarse geographic region grid, and hub-ness counts.

Full methodology, leakage checks (lane-frequency check, `quote_signal` circularity check via correlation), error analysis, and the December extrapolation caveat are documented in the report.

## Repository structure

```
.
├── data/                                       # NOT included — see Setup steps below
│   ├── train_test.csv
│   ├── validation.csv
│   ├── validation_predictions_template.csv
│   └── december_chart_inputs.csv
├── notebooks/
│   └── spotter-freight-rate-prediction.ipynb   # full pipeline: EDA → cleaning → features → both heads → predictions
├── models/
│   ├── LGB_HEAD_A.txt                          # trained LightGBM model, full feature set
│   └── LGB_HEAD_B.txt                          # trained LightGBM model, no market signals
├── outputs/
│   ├── validation_predictions.csv              # final submission: load_id, predicted_rate (12,000 rows)
│   └── december_predictions.csv                # 31-row December forecast (Head B)
├── Spotter_Freight_Rate_Model_Report.pdf        # report (methodology, metrics, error analysis, December chart)
├── requirements.txt
└── LICENSE
```

## Reproducing this

1. Clone this repo and install dependencies:
   ```bash
   python -m pip install -r requirements.txt
   ```
2. Create a `data/` folder in the repo root and add the four files provided by Spotter as part of the assessment (not included in this repo, since it's their data):
   - `train_test.csv`
   - `validation.csv`
   - `validation_predictions_template.csv`
   - `december_chart_inputs.csv`
3. Launch the notebook and run it top to bottom — this reproduces both heads and both prediction outputs:
   ```bash
   jupyter notebook notebooks/spotter-freight-rate-prediction.ipynb
   ```

To validate the output format against Spotter's scorer:

```bash
python score.py --predictions outputs/validation_predictions.csv --december-predictions outputs/december_predictions.csv
```

**Note on the December file:** the assessment's own instructions describe filling `predicted_rate` directly into `data/december_chart_inputs.csv` and passing that same path to `--december-predictions`. This repo instead writes the filled rows to a separate `outputs/december_predictions.csv` with an identical column layout (`pickup, delivery, distance, equipment, weight, date, predicted_rate`) — same content, different filename, kept separate from the raw input files intentionally. `score.py` accepts either path; both were verified to pass validation (`Validated 31 fixed December predictions.`).

The scorer writes its chart to `scorer_results/candidate_december.png`, which is reproduced in Section 7 of the report.



## Notes / open items

- The December 28–30 dip in the forecast (see report §7.1) has two plausible explanations — genuine holiday-driven demand drop, or an extrapolation artifact since December falls entirely outside the training date range — and is flagged as unresolved rather than asserted either way.
- The ~1.7–1.8% of loads with very large errors (95th → 99th percentile jump from ~7% to ~77% MAPE) are the **same loads** across both heads, suggesting an unmodeled cost driver in the source data (expedited service, hazmat, one-off negotiated rates) rather than a gap either feature set could close.
