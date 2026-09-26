"""Figure 2: DCI onset by functional outcome (A) and by DCI-related infarction (B), with unadjusted and adjusted p.

    A  mRS > 2 | mRS <= 2                      B  DCI-related infarct | no secondary infarct
       boxplots of days from haemorrhage to DCI, bracket annotated 'p=...; adj. p=...'
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .dci_timing_outcomes import GroupComparison, ModelSpec, Outcome, group_comparison

# Group codes of GroupComparison.group -> axis label, in plotting order
GROUP_LABELS = {
    Outcome.MRS: {0: 'mRS > 2', 1: 'mRS ≤ 2'},
    Outcome.INFARCTION: {1: 'DCI-related infarct', 0: 'No secondary infarct'},
}
PALETTES = {Outcome.MRS: 'magma', Outcome.INFARCTION: 'viridis'}
PANEL_LETTERS = ['A', 'B']

# Figure layout as in the submitted manuscript
FIGURE_SIZE = (10, 5)
BOX_ALPHA = 0.8
Y_LIMIT = (0, 25)
BRACKET_QUANTILE = 0.98  # bracket just above the upper whiskers, e.g. day 22
BRACKET_HEIGHT = 0.4
TEXT_OFFSET = 0.3
P_DISPLAY_FLOOR = 0.01
ROUNDING_AMBIGUOUS_P = 0.05  # 0.048 would print as 0.05; show three decimals instead


def _format_p(p: float) -> str:
    # e.g. 0.004 -> '<0.01', 0.139 -> '=0.14', 0.048 -> '=0.048'
    if p < P_DISPLAY_FLOOR:
        return f'<{P_DISPLAY_FLOOR}'
    if round(p, 2) == ROUNDING_AMBIGUOUS_P:
        return f'={p:.3f}'
    return f'={p:.2f}'


def _panel(ax, comparison: GroupComparison, letter: str) -> None:
    labels = GROUP_LABELS[comparison.outcome]
    included = comparison.group.notna()
    frame = pd.DataFrame({
        'group': comparison.group[included].map(labels),
        'days': comparison.dci_day[included],
    })

    order = list(labels.values())
    sns.boxplot(data=frame, x='group', y='days', order=order, hue='group', hue_order=order, legend=False,
                palette=PALETTES[comparison.outcome], showfliers=False, ax=ax)

    # Translucent boxes; boxprops together with hue crashes seaborn 0.13.2
    for box in ax.patches:
        box.set_alpha(BOX_ALPHA)
    ax.set_xlabel('')
    ax.set_ylabel('DCI onset (days)')
    ax.set_ylim(*Y_LIMIT)

    # Bracket between the two boxes with unadjusted and adjusted p
    y = comparison.dci_day.quantile(BRACKET_QUANTILE)
    ax.plot([0, 0, 1, 1], [y, y + BRACKET_HEIGHT, y + BRACKET_HEIGHT, y], lw=1.5, c='black')
    text = f'p{_format_p(comparison.p_unadjusted)}; adj. p{_format_p(comparison.p_adjusted)}'
    ax.text(0.5, y + BRACKET_HEIGHT + TEXT_OFFSET, text, ha='center', va='bottom', color='black')

    ax.text(0, 1.05, letter, transform=ax.transAxes, fontsize=12, va='top')


def plot_figure2(data: pd.DataFrame, spec: ModelSpec) -> plt.Figure:
    sns.set_theme(style='whitegrid')
    fig, axes = plt.subplots(1, len(Outcome), figsize=FIGURE_SIZE, sharey=True)
    for ax, outcome, letter in zip(axes, Outcome, PANEL_LETTERS):
        _panel(ax, group_comparison(data, outcome, spec), letter)
    fig.tight_layout()
    return fig
