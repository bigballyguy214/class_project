"""Group model assignment, v1: multiple linear regression for how popular a bar is.

Question the model serves (Nightlife Autopilot ops lead, opening a new neighbourhood): before we
have any Scout reports, which bars will groups actually ask about, so we pay Scout bounties to keep
those bars' live info (crowd, line, cover, music) fresh?

Outcome: popularity = the bar's Yelp review count (a proxy for how many people go and talk about
it). v1 is fitted on log(1 + reviews) because the count is heavily skewed; predictions and errors
are converted back to reviews.

Features: only traits the app shows groups or can learn on day one: price level, crowd-reported
noise level, music (DJ, live, karaoke, jukebox), good for dancing, happy hour, closes late on
Fri/Sat, good for groups, outdoor seating, TVs, serves food, and bar type (cocktail, dive, sports,
wine, pub, lounge, dance club, beer bar, music venue). City is NOT a feature, so the model can be
applied to a new city such as New York.

Decision rule: bounty if predicted reviews >= THRESHOLD (training 75th percentile), else none.
Baseline (no model): predict the training mean for every bar, so every bar gets the same decision.

Data: REAL, the Yelp Open Dataset business file (academic/non-commercial use only; Yelp's data
agreement forbids redistributing it, so it is NOT committed). The script downloads it to
group_model/data/ (gitignored) on first run. Rows: businesses whose categories include "Bars"
and "Nightlife", that are open (is_open = 1),
and whose price level and noise level are recorded. Yelp's dataset covers 11 North American metros,
not New York.

Steps
  1. load (download if missing) and filter the bars; build the features
  2. random 70/30 train/test split with --seed
  3. baseline = training mean review count for every test bar
  4. v1 = statsmodels OLS on log1p(reviews); print summary and coefficients
  5. baseline vs v1 on the test rows: R^2 (log scale and reviews), RMSE, MAE, median abs error
  6. decide(bar) on a realistic new bar (example_new_bar.json or --new-input)
  7. dollars per month for a launch neighbourhood, baseline vs v1
  8. write outputs/metrics.json, outputs/report.md and four PNG charts

Run (from the repo root; installs: pip install -r group_model/requirements.txt):
  python group_model/bar_popularity_regression.py
  python group_model/bar_popularity_regression.py --seed 7 --new-input my_bar.json
"""
import argparse
import ast
import json
import platform
import urllib.request
from pathlib import Path

import matplotlib
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import statsmodels  # noqa: E402
import statsmodels.api as sm  # noqa: E402
import sklearn  # noqa: E402
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

import charts  # noqa: E402  (one function per chart)

HERE = Path(__file__).resolve().parent
YELP_DIR = HERE / "data"  # gitignored: Yelp data may not be redistributed
YELP_FILE = YELP_DIR / "yelp_academic_dataset_business.json"
# Mirror of the official Yelp Open Dataset file (https://www.yelp.com/dataset), same content.
YELP_URL = ("https://huggingface.co/datasets/Shengxiang-Lin/Yelp-JSON/resolve/main/"
            "yelp_academic_dataset_business.json")
OUT = HERE / "outputs"
EXAMPLE = HERE / "example_new_bar.json"
TARGET = "review_count"

# ---------------------------------------------------------------------------------------------
# Business assumptions. Every figure is MADE UP for the assignment (labelled ASSUMPTION).
# ---------------------------------------------------------------------------------------------
ASSUMPTIONS = {
    "bars_in_launch_neighbourhood": 200,          # ASSUMPTION: bars we list in a new area
    "bounty_cost_usd_per_bar_month": 150.0,       # ASSUMPTION: ~30 Scout check-ins x $5 in points
    "value_fresh_info_popular_bar_usd_month": 400.0,   # ASSUMPTION: many group requests convert
    "value_fresh_info_other_bar_usd_month": 50.0,      # ASSUMPTION: few requests, little gain
}
# ASSUMPTION: "popular" = at or above the TRAINING 75th percentile of review_count.
THRESHOLD_QUANTILE = 0.75

BAR_TYPES = {  # feature name -> Yelp category
    "cocktail_bar": "Cocktail Bars", "dive_bar": "Dive Bars", "sports_bar": "Sports Bars",
    "wine_bar": "Wine Bars", "pub": "Pubs", "lounge": "Lounges", "dance_club": "Dance Clubs",
    "beer_bar": "Beer Bar", "music_venue": "Music Venues",
}
TERMS = (["price_cheap", "price_high",
          "noise_quiet", "noise_loud", "noise_very_loud",
          "music_dj", "music_live", "music_karaoke", "music_jukebox",
          "good_for_dancing", "happy_hour", "closes_late_weekend", "good_for_groups",
          "outdoor_seating", "has_tv", "serves_food"] + list(BAR_TYPES))
# Reference levels (absorbed in the intercept): price "$$", noise "average".


# ---------------------------------------------------------------------------------------------
# 1. Data
# ---------------------------------------------------------------------------------------------
def ensure_yelp_file():
    """Download the Yelp business file once (about 119 MB) into a gitignored folder."""
    if not YELP_FILE.exists():
        YELP_DIR.mkdir(parents=True, exist_ok=True)
        print(f"downloading {YELP_URL}\n  -> {YELP_FILE} (about 119 MB, once)")
        tmp = YELP_FILE.with_suffix(".part")
        urllib.request.urlretrieve(YELP_URL, tmp)
        tmp.rename(YELP_FILE)
    return YELP_FILE


def _attr(attrs, key):
    """Yelp stores attributes as strings like "True", "u'loud'" or "{'dj': True}"; parse them."""
    v = (attrs or {}).get(key)
    if v is None or v == "None":
        return None
    try:
        return ast.literal_eval(v) if isinstance(v, str) else v
    except (ValueError, SyntaxError):
        return v


def _closes_late(hours):
    """1 if the bar closes at or after 1 am on Friday or Saturday (close time past midnight)."""
    for day in ("Friday", "Saturday"):
        span = (hours or {}).get(day)
        if not span or "-" not in span:
            continue
        try:
            o, c = [int(t.split(":")[0]) + int(t.split(":")[1]) / 60 for t in span.split("-")]
        except ValueError:
            continue
        if c < o and c >= 1:  # e.g. 18:0-2:0 closes at 2 am
            return 1
    return 0


def features_from_raw(rec):
    """Turn one raw Yelp-style record (dict) into the model's 0/1 features."""
    a = rec.get("attributes") or {}
    cats = {c.strip() for c in (rec.get("categories") or "").split(",")}
    price = _attr(a, "RestaurantsPriceRange2")
    noise = _attr(a, "NoiseLevel")
    music = _attr(a, "Music") or {}
    if not isinstance(music, dict):
        music = {}
    f = {
        "price_cheap": int(price == 1),
        "price_high": int(price in (3, 4)),
        "price_unknown": int(price is None),
        "noise_quiet": int(noise == "quiet"),
        "noise_loud": int(noise == "loud"),
        "noise_very_loud": int(noise == "very_loud"),
        "noise_unknown": int(noise is None),
        "music_dj": int(music.get("dj") is True),
        "music_live": int(music.get("live") is True),
        "music_karaoke": int(music.get("karaoke") is True),
        "music_jukebox": int(music.get("jukebox") is True),
        "good_for_dancing": int(_attr(a, "GoodForDancing") is True),
        "happy_hour": int(_attr(a, "HappyHour") is True),
        "closes_late_weekend": _closes_late(rec.get("hours")),
        "good_for_groups": int(_attr(a, "RestaurantsGoodForGroups") is True),
        "outdoor_seating": int(_attr(a, "OutdoorSeating") is True),
        "has_tv": int(_attr(a, "HasTV") is True),
        "serves_food": int("Restaurants" in cats),
    }
    for name, cat in BAR_TYPES.items():
        f[name] = int(cat in cats)
    return f


def load_bars(path=None):
    """One row per open bar: id, name, city, review_count, stars and the model features."""
    path = Path(path) if path else ensure_yelp_file()
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            cats = {c.strip() for c in (r.get("categories") or "").split(",")}
            if "Bars" in cats and "Nightlife" in cats and r.get("is_open") == 1:
                f = features_from_raw(r)
                # Keep bars whose price and noise level are recorded. Missing attributes are not
                # random: busy bars get more fields filled in by reviewers, so "unknown" would leak
                # popularity into the features.
                if f.pop("price_unknown") or f.pop("noise_unknown"):
                    continue
                rows.append({"business_id": r["business_id"], "name": r["name"], "city": r["city"],
                             "state": r["state"], "stars": r["stars"], TARGET: r[TARGET],
                             **f})
    return pd.DataFrame(rows)


def split(df, seed):
    """Random 70/30 split, fixed seed (sklearn rounds the test size up)."""
    return train_test_split(df, test_size=0.30, random_state=seed)


def design_matrix(df):
    return sm.add_constant(df[TERMS].astype(float), has_constant="add")


# ---------------------------------------------------------------------------------------------
# Decision
# ---------------------------------------------------------------------------------------------
def decide(bar, model, threshold):
    """Take one new bar as a raw Yelp-style dict (attributes, categories, hours) and decide.

    Returns the predicted review count and whether to pay a Scout bounty for it.
    """
    f = features_from_raw(bar)
    if f["price_unknown"] or f["noise_unknown"]:
        raise ValueError("price level and noise level are required for a prediction")
    X = design_matrix(pd.DataFrame([f]))
    pred = float(np.expm1(model.predict(X)[0]))
    return {"predicted_reviews": round(max(pred, 0.0), 1),
            "scout_bounty": bool(pred >= threshold),
            "threshold_reviews": threshold}


def value_per_month(tp, fp, fn, tn):
    """Dollars per month for the launch neighbourhood, from test-set rates (see ASSUMPTIONS)."""
    a = ASSUMPTIONS
    n = tp + fp + fn + tn
    per_bar = {  # net value per bar per month by outcome
        "tp": a["value_fresh_info_popular_bar_usd_month"] - a["bounty_cost_usd_per_bar_month"],
        "fp": a["value_fresh_info_other_bar_usd_month"] - a["bounty_cost_usd_per_bar_month"],
        "fn": 0.0, "tn": 0.0,   # no bounty: no cost, no fresh-info gain
    }
    counts = {"tp": tp, "fp": fp, "fn": fn, "tn": tn}
    per_bar_avg = sum(per_bar[k] * counts[k] for k in counts) / n
    return {"usd_per_bar_by_outcome": per_bar, "usd_per_bar": round(per_bar_avg, 3),
            "usd_per_month": round(per_bar_avg * a["bars_in_launch_neighbourhood"], 2),
            "bounties_per_month": round((tp + fp) / n * a["bars_in_launch_neighbourhood"], 1)}


def confusion(pred_flag, true_flag):
    pred_flag, true_flag = np.asarray(pred_flag, bool), np.asarray(true_flag, bool)
    return {"tp": int((pred_flag & true_flag).sum()), "fp": int((pred_flag & ~true_flag).sum()),
            "fn": int((~pred_flag & true_flag).sum()), "tn": int((~pred_flag & ~true_flag).sum())}


def errors(y_true, y_pred):
    return {"r2": round(float(r2_score(y_true, y_pred)), 4),
            "rmse": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 2),
            "mae": round(float(mean_absolute_error(y_true, y_pred)), 2),
            "median_abs_error": round(float(median_absolute_error(y_true, y_pred)), 2)}


# ---------------------------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--new-input", type=Path, default=EXAMPLE, help="JSON file with one new bar")
    ap.add_argument("--yelp-path", type=Path, default=None, help="use a local business.json")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    # 1-2. data and split
    bars = load_bars(args.yelp_path)
    train, test = split(bars, args.seed)
    print(f"[1] {len(bars):,} open bars from the Yelp Open Dataset, {bars['city'].nunique()} cities")
    print(f"[2] split: {len(train):,} train / {len(test):,} test (seed {args.seed})")
    print(f"    reviews: train mean {train[TARGET].mean():.1f}, median {train[TARGET].median():.0f}; "
          f"test mean {test[TARGET].mean():.1f}, median {test[TARGET].median():.0f}")

    # 3. baseline: training mean for everyone
    base_pred = float(train[TARGET].mean())
    threshold = float(train[TARGET].quantile(THRESHOLD_QUANTILE))

    # 4. v1: OLS on log1p(reviews)
    y_tr = np.log1p(train[TARGET])
    model = sm.OLS(y_tr, design_matrix(train)).fit()
    summary = model.summary().as_text()
    (OUT / "ols_summary.txt").write_text(summary)
    print("[4] v1 = OLS on log(1 + reviews)\n" + summary)
    coefs = (pd.DataFrame({"coef": model.params, "p": model.pvalues}).drop("const")
             .assign(pct=lambda d: (np.exp(d["coef"]) - 1) * 100).sort_values("coef"))
    print("    effect of each trait on expected reviews, holding the others fixed:")
    for term, r in coefs.sort_values("coef", ascending=False).iterrows():
        print(f"      {term:22s} {r['pct']:+6.0f}%   p={r['p']:.3g}")

    # 5. evaluation
    pred_tr = np.expm1(model.predict(design_matrix(train)))
    pred_te = np.expm1(model.predict(design_matrix(test)))
    y_te = test[TARGET].to_numpy()
    base_te = np.full(len(test), base_pred)
    ev = {
        "baseline": errors(y_te, base_te),
        "v1": errors(y_te, pred_te),
        "baseline_log_r2": round(float(r2_score(np.log1p(y_te), np.log1p(base_te))), 4),
        "v1_log_r2_test": round(float(r2_score(np.log1p(y_te), model.predict(design_matrix(test)))), 4),
        "v1_log_r2_train": round(float(model.rsquared), 4),
        "v1_adj_r2_train_log": round(float(model.rsquared_adj), 4),
        "v1_train_errors": errors(train[TARGET].to_numpy(), pred_tr),
    }
    print("[5] test rows, baseline vs v1")
    print(f"    {'metric':28s}{'baseline':>12s}{'v1':>12s}")
    print(f"    {'R^2 (log scale)':28s}{ev['baseline_log_r2']:12.3f}{ev['v1_log_r2_test']:12.3f}")
    for k, lab in [("r2", "R^2 (reviews)"), ("rmse", "RMSE (reviews)"), ("mae", "MAE (reviews)"),
                   ("median_abs_error", "median abs error (reviews)")]:
        print(f"    {lab:28s}{ev['baseline'][k]:12.2f}{ev['v1'][k]:12.2f}")
    print(f"    v1 adjusted R^2, training fit (log scale): {ev['v1_adj_r2_train_log']:.3f}")
    print(f"    v1 R^2 log scale, train {ev['v1_log_r2_train']:.3f} vs test {ev['v1_log_r2_test']:.3f}")

    # 6. decision for a new bar
    new_bar = {k: v for k, v in json.loads(args.new_input.read_text()).items() if not k.startswith("_")}
    decision = decide(new_bar, model, threshold)
    print(f"[6] threshold = {threshold:.0f} reviews (training 75th percentile)")
    print(f"    new bar '{new_bar.get('name')}': {decision}")

    # 7. value: bounty decisions on test bars, scaled to a launch neighbourhood
    truly_popular = y_te >= threshold
    cm_v1 = confusion(pred_te >= threshold, truly_popular)
    base_flag = base_pred >= threshold          # same decision for every bar
    cm_base = confusion(np.full(len(test), base_flag), truly_popular)
    val_base, val_v1 = value_per_month(**cm_base), value_per_month(**cm_v1)
    print("[7] bounty decisions on the test bars (popular = actual reviews >= threshold)")
    print(f"    baseline predicts {base_pred:.1f} for every bar -> bounty for "
          f"{'every' if base_flag else 'no'} bar; counts {cm_base}")
    print(f"    v1 counts {cm_v1}")
    for k, v in ASSUMPTIONS.items():
        print(f"      ASSUMPTION {k:40s} {v}")
    print(f"    baseline ${val_base['usd_per_month']:,.0f}/month  v1 ${val_v1['usd_per_month']:,.0f}/month  "
          f"difference ${val_v1['usd_per_month'] - val_base['usd_per_month']:+,.0f}/month")

    # 8. outputs
    metrics = {
        "seed": args.seed,
        "data": {"source": "Yelp Open Dataset, business file (academic use; not redistributed)",
                 "rows": len(bars), "cities": sorted(bars["city"].value_counts().head(11).index),
                 "filter": "categories include Bars and Nightlife; is_open = 1; price level and noise level recorded"},
        "split": {"n_train": len(train), "n_test": len(test),
                  "train_mean_reviews": round(float(train[TARGET].mean()), 2),
                  "train_median_reviews": float(train[TARGET].median()),
                  "test_mean_reviews": round(float(test[TARGET].mean()), 2)},
        "baseline": {"rule": "predict the training mean for every bar", "prediction": round(base_pred, 2),
                     "test": ev["baseline"], "test_log_r2": ev["baseline_log_r2"],
                     "bounty_for_every_bar": bool(base_flag), "bounty_counts": cm_base},
        "v1": {"model": "statsmodels OLS on log1p(review_count)", "terms": TERMS,
               "adj_r2_train_log": ev["v1_adj_r2_train_log"], "r2_train_log": ev["v1_log_r2_train"],
               "r2_test_log": ev["v1_log_r2_test"], "test": ev["v1"], "train": ev["v1_train_errors"],
               "intercept": round(float(model.params["const"]), 4),
               "coefficients": [{"term": t, "coef": round(float(r["coef"]), 4),
                                 "pct_change_reviews": round(float(r["pct"]), 1), "p": float(r["p"])}
                                for t, r in coefs.sort_values("coef", ascending=False).iterrows()],
               "bounty_counts": cm_v1},
        "threshold": {"reviews": threshold, "rule": "training 75th percentile of review_count",
                      "test_share_popular": round(float(truly_popular.mean()), 4)},
        "decision_example": {"input_file": args.new_input.name, "input": new_bar, "output": decision},
        "value": {"assumptions": ASSUMPTIONS, "baseline": val_base, "v1": val_v1,
                  "v1_minus_baseline_usd_per_month":
                      round(val_v1["usd_per_month"] - val_base["usd_per_month"], 2)},
        "versions": {"python": platform.python_version(), "pandas": pd.__version__,
                     "numpy": np.__version__, "statsmodels": statsmodels.__version__,
                     "scikit-learn": sklearn.__version__, "matplotlib": matplotlib.__version__},
    }
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))
    write_report(metrics)
    charts.make_all(y_te, pred_te, base_pred, threshold, ev, coefs, OUT)  # code in charts.py
    print(f"[8] wrote metrics.json, report.md, ols_summary.txt and 4 PNG charts to {OUT}")
    return metrics


def write_report(m):
    b, v, val = m["baseline"], m["v1"], m["value"]
    lines = [
        "# Bar popularity regression: results (generated by bar_popularity_regression.py)", "",
        f"Seed {m['seed']}. Data: Yelp Open Dataset, {m['data']['rows']:,} open bars "
        f"({m['data']['filter']}). Outcome: review_count. v1 = OLS on log(1 + reviews).", "",
        "## Split", "", "| part | rows | mean reviews |", "|---|---|---|",
        f"| train | {m['split']['n_train']:,} | {m['split']['train_mean_reviews']} |",
        f"| test | {m['split']['n_test']:,} | {m['split']['test_mean_reviews']} |", "",
        "## Baseline vs v1 on the test rows", "",
        "| metric | baseline (training mean) | v1 |", "|---|---|---|",
        f"| R² (log scale) | {b['test_log_r2']:.3f} | {v['r2_test_log']:.3f} |",
        f"| R² (reviews) | {b['test']['r2']:.3f} | {v['test']['r2']:.3f} |",
        f"| RMSE (reviews) | {b['test']['rmse']} | {v['test']['rmse']} |",
        f"| MAE (reviews) | {b['test']['mae']} | {v['test']['mae']} |",
        f"| median abs error (reviews) | {b['test']['median_abs_error']} | {v['test']['median_abs_error']} |",
        "", f"Adjusted R², training fit (log scale): {v['adj_r2_train_log']:.3f}. "
        f"R² log scale train {v['r2_train_log']:.3f} vs test {v['r2_test_log']:.3f}.", "",
        "## Coefficients (% change in expected reviews)", "", "| term | % change | p-value |", "|---|---|---|",
        *[f"| {c['term']} | {c['pct_change_reviews']:+.0f}% | {c['p']:.3g} |" for c in v["coefficients"]], "",
        f"## Bounty decisions on the test bars (threshold {m['threshold']['reviews']:.0f} reviews)", "",
        "| cell | meaning | baseline | v1 |", "|---|---|---|---|",
        *[f"| {k.upper()} | {lab} | {b['bounty_counts'][k]} | {v['bounty_counts'][k]} |"
          for k, lab in [("tp", "popular, bounty"), ("fp", "not popular, bounty"),
                         ("fn", "popular, no bounty"), ("tn", "not popular, no bounty")]], "",
        "## Decision for the example new bar", "", f"Input: `{m['decision_example']['input_file']}`. "
        f"Output: `{json.dumps(m['decision_example']['output'])}`", "",
        "## Value per month for a launch neighbourhood (all inputs are ASSUMPTIONS)", "",
        *[f"- ASSUMPTION {k}: {x}" for k, x in val["assumptions"].items()], "",
        "| rule | bounties / month | USD per bar | USD per month |", "|---|---|---|---|",
        f"| baseline | {val['baseline']['bounties_per_month']} | {val['baseline']['usd_per_bar']} | "
        f"{val['baseline']['usd_per_month']:,.2f} |",
        f"| v1 | {val['v1']['bounties_per_month']} | {val['v1']['usd_per_bar']} | {val['v1']['usd_per_month']:,.2f} |",
        f"| v1 - baseline | | | {val['v1_minus_baseline_usd_per_month']:+,.2f} |", "",
        "## Versions", "", ", ".join(f"{k} {x}" for k, x in m["versions"].items()),
    ]
    (OUT / "report.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
