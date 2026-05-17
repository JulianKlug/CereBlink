"""Recreate the two forest plots from odds-ratio tables."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def forest_plot(data, out_path, xlim, xticks, figsize):
    """Render a forest plot.

    `data` is a list of (label, OR, CI_low, CI_high, p_value) tuples,
    ordered top-to-bottom as they should appear on the y-axis.
    """
    variables = [row[0] for row in data]
    ors = np.array([row[1] for row in data])
    lows = np.array([row[2] for row in data])
    highs = np.array([row[3] for row in data])
    pvals = np.array([row[4] for row in data])

    n = len(data)
    y = np.arange(n)[::-1]

    fig, ax = plt.subplots(figsize=figsize)

    ax.errorbar(
        ors,
        y,
        xerr=[ors - lows, highs - ors],
        fmt="none",
        ecolor="black",
        elinewidth=1.0,
        capsize=0,
        zorder=2,
    )

    for xi, yi, p in zip(ors, y, pvals):
        if p < 0.05:
            ax.plot(xi, yi, marker="o", markersize=11, color="black", zorder=3)
        else:
            ax.plot(
                xi,
                yi,
                marker="o",
                markersize=9,
                markerfacecolor="white",
                markeredgecolor="black",
                markeredgewidth=1.2,
                zorder=3,
            )

    ax.axvline(1.0, linestyle="--", color="gray", linewidth=1, zorder=1)

    ax.set_xscale("log")
    ax.set_xlim(xlim)
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{t:.2f}" for t in xticks])
    ax.minorticks_off()

    ax.set_yticks(y)
    ax.set_yticklabels(variables)
    ax.set_ylim(-0.7, n - 0.3)

    ax.set_xlabel("Odds Ratio (95% CI)")

    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0)

    # Right-side annotation columns.
    trans = ax.get_yaxis_transform()  # x in axes coords, y in data coords.
    or_col_x = 1.06
    p_col_x = 1.32

    ax.text(
        or_col_x,
        n - 0.15,
        "OR (95% CI)",
        transform=trans,
        fontweight="bold",
        va="bottom",
        ha="left",
    )
    ax.text(
        p_col_x,
        n - 0.15,
        "p-value",
        transform=trans,
        fontweight="bold",
        va="bottom",
        ha="left",
    )

    for yi, or_, lo, hi, p in zip(y, ors, lows, highs, pvals):
        ax.text(
            or_col_x,
            yi,
            f"{or_:.2f} ({lo:.2f}–{hi:.2f})",
            transform=trans,
            va="center",
            ha="left",
        )
        ax.text(
            p_col_x,
            yi,
            f"p={p:.3f}",
            transform=trans,
            va="center",
            ha="left",
        )

    fig.subplots_adjust(left=0.22, right=0.70, top=0.92, bottom=0.12)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


MAIN = [
    ("Age (per year)",   0.97, 0.94, 1.00, 0.026),
    ("Hypertension",     2.82, 1.20, 6.59, 0.017),
    ("Sex (male)",       1.35, 0.59, 3.12, 0.480),
    ("Smoking status",   1.16, 0.53, 2.55, 0.713),
    ("WFNS grade",       0.94, 0.46, 1.91, 0.855),
    ("Hunt & Hess grade",0.88, 0.41, 1.91, 0.755),
]


def rescale_age_per_decade(data, new_label="Age (per 10 years)"):
    """Return a copy of `data` with the Age row rescaled from per-year to per-decade.

    For a logistic-regression coefficient, OR_per_10y = OR_per_1y ** 10, and the
    CI bounds transform the same way. The p-value is unchanged.
    """
    out = []
    for row in data:
        label, or_, lo, hi, p = row
        if label.lower().startswith("age"):
            out.append((new_label, or_ ** 10, lo ** 10, hi ** 10, p))
        else:
            out.append(row)
    return out

FULL = [
    ("Age",           0.98, 0.96, 1.00, 0.021),
    ("Sex",           1.01, 0.59, 1.75, 0.961),
    ("Smoker",        0.91, 0.54, 1.54, 0.733),
    ("Alcohol abuse", 1.09, 0.38, 3.12, 0.871),
    ("WFNS",          0.97, 0.63, 1.51, 0.903),
    ("HH",            1.02, 0.67, 1.56, 0.909),
    ("GCS admission", 1.00, 0.89, 1.13, 0.994),
    ("HTN",           1.36, 0.73, 2.50, 0.331),
    ("DM",            1.04, 0.32, 3.42, 0.949),
    ("Statin",        1.59, 0.53, 4.79, 0.408),
    ("ASS",           2.72, 1.15, 6.42, 0.023),
    ("Clopidogrel",   1.04, 0.12, 9.31, 0.971),
    ("OAC",           2.14, 0.56, 8.18, 0.268),
]


def main():
    forest_plot(
        MAIN,
        "forest_plot_main.png",
        xlim=(0.25, 8.0),
        xticks=[0.25, 0.50, 1.00, 2.00, 4.00, 8.00],
        figsize=(10, 5.5),
    )
    forest_plot(
        FULL,
        "forest_plot_full.png",
        xlim=(0.10, 10.0),
        xticks=[0.10, 0.50, 1.00, 2.00, 4.00, 8.00],
        figsize=(10, 8.5),
    )

    # Per-decade Age variants. OR is raised to the 10th power so the effect is
    # easier to read on the log axis; non-Age rows are unchanged.
    forest_plot(
        rescale_age_per_decade(MAIN),
        "forest_plot_main_age_per_decade.png",
        xlim=(0.25, 8.0),
        xticks=[0.25, 0.50, 1.00, 2.00, 4.00, 8.00],
        figsize=(10, 5.5),
    )
    forest_plot(
        rescale_age_per_decade(FULL, new_label="Age (per 10 years)"),
        "forest_plot_full_age_per_decade.png",
        xlim=(0.10, 10.0),
        xticks=[0.10, 0.50, 1.00, 2.00, 4.00, 8.00],
        figsize=(10, 8.5),
    )


if __name__ == "__main__":
    main()
