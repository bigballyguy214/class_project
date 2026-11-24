"""Code for every chart in outputs/. One function per PNG.

Called by bar_popularity_regression.py after the model is fitted, so the charts always match the
numbers in outputs/metrics.json. Plain style: black and grey on white, labelled axes.

| PNG                              | function                    |
|----------------------------------|-----------------------------|
| outputs/baseline_vs_v1_error.png | chart_baseline_vs_v1_error  |
| outputs/predicted_vs_actual.png  | chart_predicted_vs_actual   |
| outputs/coefficients.png         | chart_coefficients          |
| outputs/residuals.png            | chart_residuals             |
"""
import matplotlib
matplotlib.use("Agg")  # write PNGs without a display
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
GREY, DARK = "#bdbdbd", "#404040"


def chart_baseline_vs_v1_error(ev, out_dir):
    """Baseline vs v1 on the TEST bars: R² (log scale), MAE and median absolute error (reviews).

    `ev` holds the test metrics computed in bar_popularity_regression.main() (step 5).
    """
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8))
    panels = [
        ("test R² (log scale)", ev["baseline_log_r2"], ev["v1_log_r2_test"], "{:.2f}"),
        ("test MAE (reviews)", ev["baseline"]["mae"], ev["v1"]["mae"], "{:.0f}"),
        ("test median abs error (reviews)", ev["baseline"]["median_abs_error"],
         ev["v1"]["median_abs_error"], "{:.0f}"),
    ]
    for ax, (label, base_val, v1_val, fmt) in zip(axes, panels):
        bars = ax.bar(["baseline", "v1"], [base_val, v1_val], color=[GREY, DARK], edgecolor="black")
        for rect, val in zip(bars, [base_val, v1_val]):  # print each value on its bar
            ax.annotate(fmt.format(val), (rect.get_x() + rect.get_width() / 2, val), ha="center",
                        va="bottom", xytext=(0, 3), textcoords="offset points")
        ax.set_ylabel(label)
        ax.axhline(0, color="black", lw=0.8)
    fig.tight_layout()
    fig.savefig(out_dir / "baseline_vs_v1_error.png", dpi=160)
    plt.close(fig)


def chart_predicted_vs_actual(y_test, pred_test, base_pred, threshold, out_dir):
    """Each TEST bar: v1's predicted reviews (x) vs actual reviews (y), both on log scales.

    The solid line is a perfect prediction, the dashed line is the baseline (training mean, the
    same guess for every bar) and the dotted line is the bounty threshold.
    """
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.scatter(pred_test, y_test, s=6, alpha=0.3, color="black", label="v1, one test bar")
    lim = [1, max(y_test.max(), pred_test.max()) * 1.1]
    ax.plot(lim, lim, color="black", lw=1, label="perfect prediction")
    ax.axvline(base_pred, color="black", ls="--", lw=1, label=f"baseline: training mean {base_pred:.0f}")
    ax.axvline(threshold, color="grey", ls=":", lw=1.5, label=f"bounty threshold {threshold:.0f}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("predicted reviews (log scale)")
    ax.set_ylabel("actual reviews, test bars (log scale)")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "predicted_vs_actual.png", dpi=160)
    plt.close(fig)


def chart_coefficients(coefs, out_dir):
    """% change in expected reviews for each trait, from the v1 coefficients (e^coef - 1).

    `coefs` is a DataFrame indexed by term with columns `pct` and `p`. Dark bars are
    statistically significant (p < 0.05), grey bars are not.
    """
    fig, ax = plt.subplots(figsize=(7, 7))
    colors = [DARK if p < 0.05 else GREY for p in coefs["p"]]
    ax.barh(coefs.index, coefs["pct"], color=colors, edgecolor="black")
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("% change in expected reviews (dark = p < 0.05)")
    fig.tight_layout()
    fig.savefig(out_dir / "coefficients.png", dpi=160)
    plt.close(fig)


def chart_residuals(y_test, pred_test, out_dir):
    """Residuals on the TEST bars, on the log scale the model was fitted on.

    Residual = log(1 + actual) - log(1 + predicted). A cloud centred on 0 with no strong pattern
    means the linear model is a reasonable fit.
    """
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.scatter(np.log1p(pred_test), np.log1p(y_test) - np.log1p(pred_test), s=6, alpha=0.3, color="black")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("predicted log(1 + reviews)")
    ax.set_ylabel("residual, test bars (log scale)")
    fig.tight_layout()
    fig.savefig(out_dir / "residuals.png", dpi=160)
    plt.close(fig)


def make_all(y_test, pred_test, base_pred, threshold, ev, coefs, out_dir):
    """Write all four charts."""
    chart_baseline_vs_v1_error(ev, out_dir)
    chart_predicted_vs_actual(y_test, pred_test, base_pred, threshold, out_dir)
    chart_coefficients(coefs, out_dir)
    chart_residuals(y_test, pred_test, out_dir)
