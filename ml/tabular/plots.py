"""Static evaluation charts (PNG) for the model cards and README.

Palette: validated categorical slots 1-2 (blue, orange) on a light surface,
recessive hairline grid, text in ink tokens, thin 2px lines.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834"]          # 1: internal test, 2: temporal test
SEQ_BLUE = ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"],
    "font.size": 10,
    "text.color": INK,
    "axes.labelcolor": INK_2,
    "axes.edgecolor": AXIS,
    "axes.titlecolor": INK,
    "axes.titlesize": 12,
    "axes.titleweight": "semibold",
    "axes.titlelocation": "left",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "legend.frameon": False,
    "lines.linewidth": 2,
})


def _save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def roc_plot(curves: dict, title: str, path):
    """curves: name -> (y_true, y_prob, auroc)."""
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    ax.plot([0, 1], [0, 1], color=AXIS, linewidth=1, linestyle="--")
    for color, (name, (y, p, auc)) in zip(SERIES, curves.items()):
        fpr, tpr, _ = roc_curve(y, p)
        ax.plot(fpr, tpr, color=color, label=f"{name}  (AUROC {auc:.3f})")
    ax.set(xlim=(0, 1), ylim=(0, 1.01), xlabel="False positive rate",
           ylabel="True positive rate (sensitivity)")
    ax.set_title(title)
    ax.legend(loc="lower right")
    _save(fig, path)


def calibration_plot(curves: dict, title: str, path, bins: int = 10):
    """curves: name -> (y_true, y_prob). Quantile bins so each point has data."""
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    ax.plot([0, 1], [0, 1], color=AXIS, linewidth=1, linestyle="--", label="Perfect calibration")
    top = 0.0
    for color, (name, (y, p)) in zip(SERIES, curves.items()):
        edges = np.unique(np.quantile(p, np.linspace(0, 1, bins + 1)))
        idx = np.clip(np.digitize(p, edges[1:-1]), 0, len(edges) - 2)
        mp = np.array([p[idx == b].mean() for b in range(len(edges) - 1)])
        fo = np.array([y[idx == b].mean() for b in range(len(edges) - 1)])
        ax.plot(mp, fo, color=color, marker="o", markersize=5,
                markeredgecolor=SURFACE, markeredgewidth=1.5, label=name)
        top = max(top, mp.max(), fo.max())
    lim = min(1.0, top * 1.1 + 0.02)
    ax.set(xlim=(0, lim), ylim=(0, lim), xlabel="Predicted risk", ylabel="Observed rate")
    ax.set_title(title)
    ax.legend(loc="upper left")
    _save(fig, path)


def importance_plot(names: list[str], values: np.ndarray, title: str, path, unit: str):
    order = np.argsort(values)
    fig, ax = plt.subplots(figsize=(6, 0.36 * len(names) + 1.0))
    ax.barh(np.array(names)[order], values[order], color=SERIES[0], height=0.6)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel(f"Mean |SHAP value| (impact on predicted {unit})")
    ax.set_title(title)
    for i, v in enumerate(values[order]):
        if i >= len(order) - 3:  # direct-label only the top three
            ax.text(v, i, f"  {v:.2f}", va="center", color=INK_2, fontsize=9)
    _save(fig, path)


def fairness_plot(rows: list[dict], title: str, path):
    """rows: {group, subgroup, auroc, ci_low, ci_high, n}. One dot + CI per subgroup,
    with a gap and a header between groups (Sex / Age / Ethnicity)."""
    ys, pos, headers, prev = [], 0.0, [], None
    for r in rows:
        if r["group"] != prev:
            if prev is not None:
                pos += 0.6
            headers.append((r["group"], pos))
            pos += 0.8
            prev = r["group"]
        ys.append(pos)
        pos += 1.0
    ys = -np.array(ys)

    fig, ax = plt.subplots(figsize=(6.4, 0.34 * pos + 1.3))
    lo = [r["auroc"] - r["ci_low"] for r in rows]
    hi = [r["ci_high"] - r["auroc"] for r in rows]
    overall = rows[0]["overall"]
    ax.axvline(overall, color=MUTED, linewidth=1, linestyle="--")
    ax.errorbar([r["auroc"] for r in rows], ys, xerr=[lo, hi], fmt="o", color=SERIES[0],
                ecolor=AXIS, elinewidth=2, capsize=0, markersize=6,
                markeredgecolor=SURFACE, markeredgewidth=1.5)
    ax.set_yticks(ys, [f"{r['subgroup']}  (n={r['n']:,})" for r in rows])
    for name, y in headers:
        ax.text(0, -y, name.upper(), transform=ax.get_yaxis_transform(), ha="right",
                va="center", color=INK, fontsize=8.5, fontweight="semibold")
    ax.grid(axis="y", visible=False)
    ax.set_ylim(-pos + 0.4, 0.6)
    ax.set_xlabel(f"AUROC (95% CI)  ·  dashed line = overall AUROC {overall:.3f}")
    ax.set_title(title)
    _save(fig, path)


def model_comparison_plot(results: dict, title: str, path):
    """results: model name -> list of CV fold AUROCs."""
    names = list(results)
    means = [np.mean(results[n]) for n in names]
    sds = [np.std(results[n]) for n in names]
    fig, ax = plt.subplots(figsize=(5.2, 0.5 * len(names) + 1.2))
    y = np.arange(len(names))[::-1]
    ax.errorbar(means, y, xerr=sds, fmt="o", color=SERIES[0], ecolor=AXIS, elinewidth=2,
                markersize=7, markeredgecolor=SURFACE, markeredgewidth=1.5)
    for yi, m in zip(y, means):
        ax.text(m, yi + 0.18, f"{m:.3f}", ha="center", color=INK_2, fontsize=9)
    ax.set_yticks(y, names)
    ax.set_ylim(-0.6, len(names) - 0.3)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel(f"Cross-validated AUROC (mean ± SD, {len(results[names[0]])} folds)")
    ax.set_title(title)
    _save(fig, path)
