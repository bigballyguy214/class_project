# Business framing, value worksheet and slide checklist

Reference for the team: the facts and numbers behind the slides. Write the slides in your own words; do not copy text from here.
Every number below comes from `outputs/metrics.json` (seed 42); the key is given in brackets.

## 1. Question, decision and baseline

**Who acts on it:** the Nightlife Autopilot ops lead opening a new neighbourhood.

**Question:** before we have any Scout reports, which bars will groups ask about most, so we pay
Scout bounties to keep those bars' live info (crowd, line, cover, music) fresh?

**What the product does today without the model:** nothing targeted. Every listed bar is treated
the same until Scouts happen to report on it, so popular bars can sit with stale info while
Scouts report on quiet ones.

**Decision rule:** predict the bar's review count. If the prediction is at or above the threshold
of 233 reviews (the training 75th percentile) [`threshold.reviews`], pay a Scout bounty for that
bar. Otherwise no bounty.

```
predicted = expm1(model.predict(features(bar)))
bounty    = predicted >= 233
```

**Baseline:** predict the training mean, 193.4 reviews [`baseline.prediction`], for every bar.
That is below 233, so the baseline gives no bar a bounty. That matches what the product does today.

## 2. Model

- **Family:** multiple linear regression, statsmodels OLS, on log(1 + reviews). Logs are used
  because review counts are very skewed (median 109, max 4,876). Predictions are converted back
  to reviews.
- **Features:** 25 yes/no traits: price, noise level, music, dancing, happy hour, late closing,
  groups, outdoor seating, TVs, food, bar type. See `DATA.md`.
- **Key training fit:** adjusted R² = 0.344 on the log scale [`v1.adj_r2_train_log`].
- **Biggest effects** [`v1.coefficients`], holding the other traits fixed:

| Trait | Effect on expected reviews |
|---|---|
| Serves food | +114% |
| Outdoor seating | +47% |
| Music venue | +45% |
| Good for groups | +41% |
| Cocktail bar | +24% |
| Happy hour | +24% |
| Closes late on weekends | −19% |
| Sports bar | −31% |
| Has TVs | −37% |
| Quiet | −47% |
| Cheap ($) | −52% |

DJ, dancing and "loud" have small effects that are not statistically different from zero.

## 3. Fit on the test rows (1,622 bars)

| Metric | Baseline (mean) | v1 | Key |
|---|---|---|---|
| R², log scale | −0.26 | 0.33 | `baseline.test_log_r2`, `v1.r2_test_log` |
| R², reviews | 0.00 | 0.04 | `*.test.r2` |
| MAE, reviews | 172 | 136 | `*.test.mae` |
| Median absolute error, reviews | 130 | 57 | `*.test.median_abs_error` |
| RMSE, reviews | 301 | 295 | `*.test.rmse` |

Training R² (log) 0.349 vs test 0.332 [`v1.r2_train_log`, `v1.r2_test_log`]: almost no drop, so
the model is not overfitting.

**Read this before presenting.** The model is strong at the typical bar: the median miss falls
from 130 to 57 reviews. It cannot predict the few huge outliers (bars with thousands of reviews),
so R² measured in raw reviews stays near 0.04 and RMSE barely moves. On the log scale, where it was
fitted, it explains a third of the variation. The baseline's log-scale R² is negative because the
mean (193) is far above the typical bar (median 109).

## 4. Decision for a new input

`example_new_bar.json`: a new Lower East Side cocktail bar and lounge, $$, loud, DJ, dancing,
happy hour, open to 2 am Fri/Sat, good for groups, no food, no outdoor seating, no TVs.

Result [`decision_example.output`]: predicted **82 reviews**, below 233, so **no bounty**. The model
expects it to be a niche spot: lounges, late closing and no food all lower expected popularity.

## 5. Value worksheet

All dollar figures are **ASSUMPTIONS** made up for the assignment [`value.assumptions`].

| Assumption | Value | Where a real number could come from |
|---|---|---|
| Bars listed in a launch neighbourhood | 200 | Count of bars in the new area |
| Bounty cost per bar per month | $150 | About 30 Scout check-ins × $5 in points |
| Value of fresh info at a popular bar per month | $400 | Group requests for that bar × conversion × revenue per visit |
| Value of fresh info at any other bar per month | $50 | Same, for low-demand bars |

Net value per bar per month by outcome:

| Outcome | Meaning | Value |
|---|---|---|
| Bounty on a popular bar (TP) | $400 − $150 | +$250 |
| Bounty on a non-popular bar (FP) | $50 − $150 | −$100 |
| No bounty (FN or TN) | no cost, no gain | $0 |

Test-bar counts ("popular" = actual reviews ≥ 233) [`*.bounty_counts`]:

| | TP | FP | FN | TN |
|---|---|---|---|---|
| Baseline | 0 | 0 | 415 | 1,207 |
| v1 | 86 | 68 | 329 | 1,139 |

Hand calculation for v1: (86 × 250 + 68 × −100) / 1,622 = $9.06 per bar per month, × 200 bars =
**$1,813 per month** [`value.v1.usd_per_month`], with about 19 bounties a month.
Baseline: no bounties, **$0 per month**. Difference: **+$1,813 per month**
[`value.v1_minus_baseline_usd_per_month`].

## 6. Slide checklist

| Slide | Item | Where it comes from |
|---|---|---|
| Title | Team name, members | Nightlife Autopilot, Group 7: Meira Berlow, Abhiviraj, Ryan Mehta, Dawid Litwiniec |
| Title | Repo link and script path | https://github.com/bigballyguy214/class_project/blob/main/group_model/bar_popularity_regression.py |
| 1 | Data source, rows, period, split | `DATA.md` table; `split.n_train`, `split.n_test` |
| 1 | Outcome, features, family | §2 above |
| 1 | Key training fit statistic | `v1.adj_r2_train_log` = 0.344 |
| 2 | Baseline vs v1 chart | `outputs/baseline_vs_v1_error.png` |
| 2 | Headline: what the model changes and what it's worth | §3 and §5 |
| 2 | Decision rule and result for the new input | §1 and §4 |
| 2 | Value, baseline vs v1 | §5 |

## 7. How each number is calculated

- **70/30 split:** shuffle the 5,406 bars with seed 42; fit on 3,784, score on 1,622.
- **Baseline:** the average review count of the training bars (193.4), predicted for every test bar.
- **Adjusted R²:** share of variation in log(1 + reviews) the model explains on the training bars,
  penalised for the number of features.
- **R²:** 1 − (sum of squared prediction errors) / (sum of squared distances from the test mean).
  0 means no better than guessing the mean; negative means worse.
- **MAE:** average size of the miss, in reviews. **Median absolute error:** the typical miss, half
  of bars are closer. **RMSE:** square root of the average squared miss; big outliers dominate it.
- **% effect of a trait:** e^coefficient − 1, because the model is on the log scale.
- **Threshold:** the review count 75% of training bars are below (233).
- **TP/FP/FN/TN:** compare the bounty decision with whether the bar's actual reviews are ≥ 233.
- **Value per month:** (TP × $250 + FP × −$100) / test bars × 200 bars.
