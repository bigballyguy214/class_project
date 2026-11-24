# Group model assignment: Nightlife Autopilot (Group 7)

Team: Meira Berlow, Abhiviraj, Ryan Mehta, Dawid Litwiniec.
Model family: **regression**. Script: [`bar_popularity_regression.py`](bar_popularity_regression.py).

This folder is the model part of the assignment: the script, the data description, every number
and chart, and the logs that show it ran. The slides are made separately by the team.

## Run it

From the repo root:

```bash
python3 -m venv .venv
.venv/bin/pip install -r group_model/requirements.txt
.venv/bin/python group_model/bar_popularity_regression.py     # fits v1, prints everything, ~10 s
.venv/bin/python -m pytest group_model/tests -q                # 3 checks
```

The first run downloads the Yelp Open Dataset business file (about 119 MB) into
`group_model/data/`. That folder is not committed because Yelp's data agreement allows course use
but not redistribution. Every run rewrites `outputs/`.

## The assignment, step by step, and where each piece is

### The question
From the point of view of the **Nightlife Autopilot ops lead opening a new neighbourhood**: before
we have Scout reports, which bars will groups ask about most, so we pay Scout bounties to keep their
live info (crowd, line, cover, music) fresh? **Today, without a model,** every bar is treated the
same until a Scout happens to report on it. Details: [`BUSINESS.md`](BUSINESS.md) §1.

### The model
Multiple linear regression (statsmodels OLS) on log(1 + Yelp review count), with 25 yes/no traits:
price, crowd-reported noise level, music (DJ, live, karaoke, jukebox), dancing, happy hour, closes
late on weekends, good for groups, outdoor seating, TVs, food, and bar type. Feature list: `TERMS`,
script line 86. Feature building: `features_from_raw()`, line 134.

### Context: decision rule and baseline
- **Decision rule:** pay a Scout bounty if predicted reviews ≥ 233 (the training 75th
  percentile). Code: `decide()`, line 201; threshold set at line 79.
- **Baseline:** predict the training mean (193.4 reviews) for every bar. It is below 233, so the
  baseline gives no bounties, which is what the product does today. Code: line 264.

### The data
Real Yelp Open Dataset: 5,406 open bars with price and noise level recorded, in 11 North American
metros (not New York), snapshot of January 2022, one row per bar. Outcome: `review_count`. Full
description and data dictionary with the outcome marked: [`DATA.md`](DATA.md). Loading and
filtering: `load_bars()`, line 168.

### Task 1: split the data
Random 70/30 split, seed 42: **3,784 training bars, 1,622 test bars**. Code: `split()`, line 189.
Evidence: [`outputs/run_log.txt`](outputs/run_log.txt) step [2], `metrics.json` → `split`.

### Task 2: fit the model and read the output
OLS fitted on the training rows only (line 268). Full statsmodels output:
[`outputs/ols_summary.txt`](outputs/ols_summary.txt). Effect of each trait on expected reviews:
[`outputs/report.md`](outputs/report.md) and [`outputs/coefficients.png`](outputs/coefficients.png).
What it means for the business is in [`BUSINESS.md`](BUSINESS.md) §2: food (+114%), outdoor
seating (+47%), cocktail bar (+24%) and happy hour (+24%) go with more popular bars. Cheap prices
(−52%), quiet rooms (−47%), TVs (−37%) and sports bars (−31%) go with less popular ones.

### Task 3: evaluate fit, baseline vs v1, on the test rows
Code: line 280. Evidence: `run_log.txt` step [5], `metrics.json` → `baseline.test`, `v1.test`.

| Test bars (1,622) | Baseline | v1 |
|---|---|---|
| Adjusted R², training fit (log scale) | | 0.344 |
| R² (log scale) | −0.26 | 0.33 |
| R² (reviews) | 0.00 | 0.04 |
| Typical error: median absolute error (reviews) | 130 | 57 |
| MAE (reviews) | 172 | 136 |
| RMSE (reviews) | 301 | 295 |

Training R² 0.349 vs test 0.332 (log scale): no overfitting. Chart:
[`outputs/baseline_vs_v1_error.png`](outputs/baseline_vs_v1_error.png).

### Task 4: make the decision for a new input
`decide()` is run on [`example_new_bar.json`](example_new_bar.json), a new Lower East Side
cocktail lounge with DJs, $$, loud, dancing, happy hour, open to 2 am. Result: **82 predicted
reviews, below 233, so no bounty.** Evidence: `run_log.txt` step [6], `metrics.json` →
`decision_example`.

### Task 5: quantify it, baseline vs v1
Code: `value_per_month()`, line 216; assumptions at line 72, each labelled ASSUMPTION.
For 200 bars in a launch neighbourhood: **baseline $0 a month, v1 +$1,813 a month** (about 19
bounties a month). Hand calculation: [`BUSINESS.md`](BUSINESS.md) §5. Evidence: `run_log.txt`
step [7], `metrics.json` → `value`.

### Task 6: present it
Done by the team as two slides plus a title slide. What each slide needs and where each fact comes
from: [`BUSINESS.md`](BUSINESS.md) §6. How each number is calculated, so everyone can explain it:
[`BUSINESS.md`](BUSINESS.md) §7.

## Files

| File | What it is |
|---|---|
| `bar_popularity_regression.py` | the standalone script: loads data, fits v1, prints the evaluation, returns a decision |
| `charts.py` | the code for every chart, one function per PNG |
| `example_new_bar.json` | the new input used for the decision |
| `DATA.md` | data source, rows, period, filter, data dictionary |
| `BUSINESS.md` | question, rule, baseline, value worksheet, slide checklist, how each number is calculated |
| `tests/test_model.py` | checks: split is reproducible and 70/30, no outcome in the features, `decide()` works |
| `outputs/run_log.txt` | full printed output of the script run that produced these files |
| `outputs/test_log.txt` | output of the test run |
| `outputs/metrics.json` | every number, machine-readable |
| `outputs/report.md` | every number as tables |
| `outputs/ols_summary.txt` | full regression output |

## Charts and the code that makes each one

| Chart | Code in `charts.py` | Shows |
|---|---|---|
| [`baseline_vs_v1_error.png`](outputs/baseline_vs_v1_error.png) | `chart_baseline_vs_v1_error()`, line 22 | test R², MAE, median error, baseline vs v1 |
| [`predicted_vs_actual.png`](outputs/predicted_vs_actual.png) | `chart_predicted_vs_actual()`, line 46 | each test bar, predicted vs actual reviews |
| [`coefficients.png`](outputs/coefficients.png) | `chart_coefficients()`, line 68 | % effect of each trait on reviews |
| [`residuals.png`](outputs/residuals.png) | `chart_residuals()`, line 84 | test residuals on the log scale |

The script calls `charts.make_all()` after fitting, so the charts always match `metrics.json`.

## Limits

- The data covers 11 other North American metros, not New York. City is not a feature, so the
  model can score a New York bar, but it has not been checked on New York bars.
- Review count is a proxy for popularity, not a live crowd count. Yelp has no live crowds, lines
  or cover charges; Scouts collect those in the app.
- All dollar figures are made-up assumptions, labelled as such.
