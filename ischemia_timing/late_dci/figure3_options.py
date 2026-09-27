"""Figure 3 candidates: circumstances of DCI diagnosis. Five layouts of the same aggregate tables; one to be chosen.

    OPTION 1  A  assessability and pCT verification, stacked bars   B  trigger frequencies, dot + 95% CI
    OPTION 2  trigger source by clinical assessability, 100% stacked bars
    OPTION 3  alluvial: all DCI -> assessability -> pCT -> trigger source
    OPTION 4  UpSet: combinations of pCT triggers
    OPTION 5  waffle: one square per DCI patient, grouped by assessability, coloured by trigger source

Colour always encodes trigger source, e.g. blue = clinical only in every option.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch, Rectangle

from .dci_triggers import (ASSESSABILITY_LABELS, CLINICAL_DETERIORATION, CLINICAL_TRIGGERS, COMBINATION_SETS,
                           MONITORING_TRIGGERS, NO_PCT, NOT_RECORDED, PCT_LABELS, Population, TriggerSource)
from .cohort import CLINICAL_SIGNS, PCT_TRIGGERS

# Reference palette: categorical slots 1-3 for trigger source, neutral inks for everything else
TEXT_PRIMARY = '#0b0b0b'
TEXT_SECONDARY = '#52514e'
GRID_COLOR = '#e4e3df'
SURFACE = '#ffffff'
NODE_COLOR = '#52514e'
SOURCE_COLORS = {
    TriggerSource.CLINICAL_ONLY.value: '#2a78d6',
    TriggerSource.BOTH.value: '#1baf7a',
    TriggerSource.MONITORING_ONLY.value: '#eb6834',
    TriggerSource.NONE.value: '#9b9a94',
    NO_PCT: '#dddcd6',
}
SOURCE_ORDER = list(SOURCE_COLORS)
PCT_SOURCES = SOURCE_ORDER[:-1]

# Yes / no / not recorded shades for assessability and pCT bars (option 1A)
STATUS_COLORS = ['#52514e', '#9b9a94', '#dddcd6']
ASSESSABILITY_ORDER = list(ASSESSABILITY_LABELS.values()) + [NOT_RECORDED]
PCT_ORDER = list(PCT_LABELS.values()) + [NOT_RECORDED]

ITEM_LABELS = {
    'decreased_consciousness': 'Decreased consciousness',
    'focal_neuro_signs': 'Focal neurological signs',
    'unexplained_persisting_DoC_or_deficit': 'Unexplained persisting DoC / deficit',
    'delirium': 'Delirium',
    'raised_icp': 'Raised ICP',
    'suspect_TCD': 'Suspicious TCD',
    'decreased_ptio2': 'Decreased PtiO₂',
    'suspect_microdialysis': 'Suspicious microdialysis',
    'decreased_NIRS': 'Decreased NIRS',
    CLINICAL_DETERIORATION: 'Decreased consciousness / focal signs',
}
CLINICAL_COLOR = SOURCE_COLORS[TriggerSource.CLINICAL_ONLY.value]
MONITORING_COLOR = SOURCE_COLORS[TriggerSource.MONITORING_ONLY.value]

MARKER_SIZE = 8
GAP_LINE_WIDTH = 2  # surface gap between adjacent fills
MIN_LABEL_SHARE = 0.06  # stacked segment narrower than 6% -> no inline label
DARK_FILLS = {'#52514e', '#2a78d6'}  # white inline text on these

ALLUVIAL_NODE_WIDTH = 0.05
ALLUVIAL_NODE_GAP = 5  # patients' worth of vertical space between nodes
ALLUVIAL_ALPHA = 0.55
ALLUVIAL_CURVE_POINTS = 50

WAFFLE_COLUMNS = 10
WAFFLE_GROUP_GAP = 2  # squares between assessability groups


class Option(Enum):
    PATHWAY_AND_TRIGGERS = 1
    SOURCE_BY_ASSESSABILITY = 2
    ALLUVIAL = 3
    UPSET = 4
    WAFFLE = 5


@dataclass(frozen=True)
class FigureData:
    circumstances: pd.DataFrame  # dci_triggers.diagnosis_circumstances
    paths: pd.DataFrame          # dci_triggers.diagnostic_paths
    combinations: pd.DataFrame   # dci_triggers.trigger_combinations


def _clean(ax) -> None:
    for side in ['top', 'right']:
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=TEXT_SECONDARY)


def _count_label(n: int, total: int) -> str:
    # e.g. 44 of 98 -> '44 (45%)'
    return f'{n} ({100 * n / total:.0f}%)'


def _display(label: str) -> str:
    # First letter up unless an acronym leads, e.g. 'no pCT' -> 'No pCT', 'pCT' -> 'pCT'
    return label if label[1:2].isupper() else label[0].upper() + label[1:]


def _text_color(fill: str) -> str:
    return 'white' if fill in DARK_FILLS else TEXT_PRIMARY


def _source_legend(fig, sources: list[str], ncol: int) -> None:
    handles = [Patch(color=SOURCE_COLORS[source], label=_display(source)) for source in sources]
    fig.legend(handles=handles, loc='upper center', ncol=ncol, frameon=False, bbox_to_anchor=(0.5, 0.0),
               labelcolor=TEXT_PRIMARY)


def _stacked_row(ax, y: float, counts: pd.Series, colors: list[str], total: int, height: float = 0.6) -> None:
    """One 100% bar; inline 'label n (%)' where the segment is wide enough."""
    left = 0.0
    for (label, n), color in zip(counts.items(), colors):
        share = n / total
        ax.barh(y, share, left=left, height=height, color=color, edgecolor=SURFACE, linewidth=GAP_LINE_WIDTH)
        if share >= MIN_LABEL_SHARE:
            ax.text(left + share / 2, y, f'{n}\n({100 * share:.0f}%)', ha='center', va='center', fontsize=8,
                    color=_text_color(color))
        left += share


# Option 1 --------------------------------------------------------------------------------------------------------

def _pathway_bars(ax, paths: pd.DataFrame) -> None:
    total = int(paths['n'].sum())
    rows = [
        ('Clinical examination', paths.groupby('assessability')['n'].sum().reindex(ASSESSABILITY_ORDER, fill_value=0)),
        ('Perfusion CT', paths.groupby('pct')['n'].sum().reindex(PCT_ORDER, fill_value=0)),
    ]

    for y, (_, counts) in enumerate(rows):
        _stacked_row(ax, y, counts, STATUS_COLORS, total)

    # Shades shared by both rows, e.g. dark = assessable / pCT done
    names = [f'{_display(a)} / {b}' for a, b in zip(ASSESSABILITY_ORDER[:-1], PCT_ORDER[:-1])] + [_display(NOT_RECORDED)]
    handles = [Patch(color=color, label=name) for color, name in zip(STATUS_COLORS, names)]
    ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 0.0), ncol=len(handles), frameon=False,
              fontsize=8)

    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([name for name, _ in rows])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0, colors=TEXT_PRIMARY)
    ax.set_title(f'A  Verified DCI (n = {total})', loc='left', fontweight='bold')


def _trigger_dots(ax, circumstances: pd.DataFrame) -> None:
    signs = circumstances[circumstances['population'] == Population.CLINICALLY_ASSESSABLE.value]
    triggers = circumstances[circumstances['population'] == Population.PCT_VERIFIED.value]
    groups = [
        (f'Clinically assessable (n = {signs["n_population"].iloc[0]})', signs, CLINICAL_SIGNS),
        (f'pCT verified (n = {triggers["n_population"].iloc[0]}), clinical', triggers,
         [t for t in PCT_TRIGGERS if t in CLINICAL_TRIGGERS]),
        (f'pCT verified (n = {triggers["n_population"].iloc[0]}), monitoring', triggers, MONITORING_TRIGGERS),
    ]

    # Rows top to bottom: group header, then items sorted by frequency
    y, ticks, labels = 0, [], []
    for header, table, items in groups:
        ax.text(0, y, header, fontsize=8.5, fontweight='bold', color=TEXT_PRIMARY, va='center')
        y += 1
        rows = table.set_index('item').loc[items].sort_values('percent', ascending=False)
        for item, row in rows.iterrows():
            color = MONITORING_COLOR if item in MONITORING_TRIGGERS else CLINICAL_COLOR
            ax.plot([row['lower'], row['upper']], [y, y], color=color, linewidth=2, solid_capstyle='round')
            ax.plot(row['percent'], y, 'o', color=color, markersize=MARKER_SIZE, markeredgecolor=SURFACE,
                    markeredgewidth=GAP_LINE_WIDTH)
            ax.text(row['upper'] + 2, y, f'{row["n_yes"]}/{row["n_recorded"]}', va='center', fontsize=8,
                    color=TEXT_SECONDARY)
            ticks.append(y)
            labels.append(ITEM_LABELS[item])
            y += 1

    ax.set_yticks(ticks)
    ax.set_yticklabels(labels)
    ax.set_ylim(y - 0.4, -0.8)
    ax.set_xlim(0, 100)
    ax.set_xlabel('Patients (%), 95% CI')
    ax.grid(axis='x', color=GRID_COLOR, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis='y', length=0)
    _clean(ax)
    ax.spines['left'].set_visible(False)
    ax.set_title('B  Findings at DCI diagnosis', loc='left', fontweight='bold')


def _option_pathway_and_triggers(data: FigureData):
    fig, axes = plt.subplots(2, 1, figsize=(8, 7.5), gridspec_kw={'height_ratios': [1, 3.2]})
    _pathway_bars(axes[0], data.paths)
    _trigger_dots(axes[1], data.circumstances)
    fig.tight_layout()
    return fig


# Option 2 --------------------------------------------------------------------------------------------------------

def _option_source_by_assessability(data: FigureData):
    verified = data.paths[data.paths['pct'] == PCT_LABELS[1]]
    rows = [(f'All pCT verified\n(n = {verified["n"].sum()})', verified)]
    rows += [(f'{_display(group)}\n(n = {verified.loc[verified["assessability"] == group, "n"].sum()})',
              verified[verified['assessability'] == group]) for group in ASSESSABILITY_ORDER]

    fig, ax = plt.subplots(figsize=(8, 3.8))
    for y, (_, group) in enumerate(rows):
        counts = group.groupby('source')['n'].sum().reindex(PCT_SOURCES, fill_value=0)
        _stacked_row(ax, y, counts, [SOURCE_COLORS[s] for s in PCT_SOURCES], int(counts.sum()))

    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([label for label, _ in rows])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xticks(np.linspace(0, 1, 5))
    ax.set_xticklabels([f'{p:.0f}%' for p in np.linspace(0, 100, 5)])
    ax.set_xlabel('pCT-verified DCI patients')
    ax.tick_params(axis='y', length=0)
    _clean(ax)
    ax.spines['left'].set_visible(False)
    _source_legend(fig, PCT_SOURCES, ncol=len(PCT_SOURCES))
    ax.set_title('Trigger of the diagnostic pCT, by clinical assessability', loc='left', fontweight='bold')
    fig.tight_layout()
    return fig


# Option 3 --------------------------------------------------------------------------------------------------------

ALL_DCI = 'All DCI'
STAGES = [('all', [ALL_DCI]), ('assessability', ASSESSABILITY_ORDER), ('pct', PCT_ORDER), ('source', SOURCE_ORDER)]


def _node_spans(totals: pd.Series, order: list[str], height: float) -> dict[str, tuple[float, float]]:
    # Nodes stacked top-down, vertically centred, e.g. {'assessable': (0, 78), 'not assessable': (83, 105), ...}
    present = [node for node in order if totals.get(node, 0) > 0]
    used = totals[present].sum() + ALLUVIAL_NODE_GAP * (len(present) - 1)
    top = (height - used) / 2
    spans = {}
    for node in present:
        spans[node] = (top, top + totals[node])
        top += totals[node] + ALLUVIAL_NODE_GAP
    return spans


def _ribbon(ax, x0: float, x1: float, left: tuple[float, float], right: tuple[float, float], color: str) -> None:
    # Smoothstep between node edges, e.g. top edge from y=10 at x0 to y=40 at x1
    t = np.linspace(0, 1, ALLUVIAL_CURVE_POINTS)
    ease = 3 * t ** 2 - 2 * t ** 3
    x = x0 + (x1 - x0) * t
    top = left[0] + (right[0] - left[0]) * ease
    bottom = left[1] + (right[1] - left[1]) * ease
    ax.fill_between(x, top, bottom, color=color, alpha=ALLUVIAL_ALPHA, linewidth=0)


def _option_alluvial(data: FigureData):
    paths = data.paths.assign(all=ALL_DCI)
    total = int(paths['n'].sum())
    height = total + ALLUVIAL_NODE_GAP * (len(SOURCE_ORDER) - 1)
    source_rank = {source: rank for rank, source in enumerate(SOURCE_ORDER)}

    spans = [_node_spans(paths.groupby(column)['n'].sum(), order, height) for column, order in STAGES]
    rank = [{node: i for i, node in enumerate(order)} for _, order in STAGES]

    fig, ax = plt.subplots(figsize=(10, 6))

    # Ribbons per (left node, right node, source); stacked inside nodes by partner node, then source
    for stage in range(len(STAGES) - 1):
        left_column, right_column = STAGES[stage][0], STAGES[stage + 1][0]
        # Last stage is the source itself -> group once, e.g. ['pct', 'source']
        keys = list(dict.fromkeys([left_column, right_column, 'source']))
        flows = paths.groupby(keys)['n'].sum().reset_index()
        x0, x1 = stage + ALLUVIAL_NODE_WIDTH / 2, stage + 1 - ALLUVIAL_NODE_WIDTH / 2

        left_offset = {node: span[0] for node, span in spans[stage].items()}
        by_left = flows.assign(key=flows[right_column].map(rank[stage + 1]), s=flows['source'].map(source_rank))
        left_edges = {}
        for row in by_left.sort_values(['key', 's']).itertuples():
            start = left_offset[row[1]]
            left_edges[(row[1], row[2], row.source)] = (start, start + row.n)
            left_offset[row[1]] += row.n

        right_offset = {node: span[0] for node, span in spans[stage + 1].items()}
        by_right = flows.assign(key=flows[left_column].map(rank[stage]), s=flows['source'].map(source_rank))
        for row in by_right.sort_values(['key', 's']).itertuples():
            start = right_offset[row[2]]
            right_offset[row[2]] += row.n
            _ribbon(ax, x0, x1, left_edges[(row[1], row[2], row.source)], (start, start + row.n),
                    SOURCE_COLORS[row.source])

    # Nodes with direct labels, e.g. 'Assessable 78'
    for stage, stage_spans in enumerate(spans):
        for node, (top, bottom) in stage_spans.items():
            color = SOURCE_COLORS.get(node, NODE_COLOR) if stage == len(STAGES) - 1 else NODE_COLOR
            ax.add_patch(Rectangle((stage - ALLUVIAL_NODE_WIDTH / 2, top), ALLUVIAL_NODE_WIDTH, bottom - top,
                                   color=color, linewidth=0))
            last = stage == len(STAGES) - 1
            x = stage + (1 if last else -1) * (ALLUVIAL_NODE_WIDTH / 2 + 0.03)
            ax.text(x, (top + bottom) / 2, f'{_display(node)}\n{_count_label(int(bottom - top), total)}',
                    ha='left' if last else 'right', va='center', fontsize=8, color=TEXT_PRIMARY)

    stage_titles = ['', 'Clinical examination', 'Perfusion CT', 'Trigger source']
    for stage, title in enumerate(stage_titles):
        ax.text(stage, -4, title, ha='center', va='bottom', fontsize=9, fontweight='bold', color=TEXT_SECONDARY)

    ax.set_xlim(-0.6, len(STAGES) - 0.3)
    ax.set_ylim(height + 2, -8)
    ax.axis('off')
    fig.tight_layout()
    return fig


# Option 4 --------------------------------------------------------------------------------------------------------

def _combination_color(row: pd.Series) -> str:
    clinical = any(row[s] for s in COMBINATION_SETS if s not in MONITORING_TRIGGERS)
    monitoring = any(row[s] for s in MONITORING_TRIGGERS)
    if clinical and monitoring:
        return SOURCE_COLORS[TriggerSource.BOTH.value]
    if clinical:
        return SOURCE_COLORS[TriggerSource.CLINICAL_ONLY.value]
    if monitoring:
        return SOURCE_COLORS[TriggerSource.MONITORING_ONLY.value]
    return SOURCE_COLORS[TriggerSource.NONE.value]


def _option_upset(data: FigureData):
    combinations = data.combinations.sort_values('n', ascending=False).reset_index(drop=True)
    sizes = pd.Series({s: int(combinations.loc[combinations[s], 'n'].sum()) for s in COMBINATION_SETS})
    sets = list(sizes.sort_values(ascending=False).index)
    x = np.arange(len(combinations))
    colors = [_combination_color(row) for _, row in combinations.iterrows()]

    fig = plt.figure(figsize=(12, 6))
    grid = fig.add_gridspec(2, 2, width_ratios=[1.2, 4], height_ratios=[2, 1.6], wspace=0.55, hspace=0.05)
    bars = fig.add_subplot(grid[0, 1])
    matrix = fig.add_subplot(grid[1, 1], sharex=bars)
    totals = fig.add_subplot(grid[1, 0], sharey=matrix)

    # Intersection sizes
    bars.bar(x, combinations['n'], color=colors, width=0.7)
    for xi, n in zip(x, combinations['n']):
        bars.text(xi, n + 0.5, str(n), ha='center', va='bottom', fontsize=7.5, color=TEXT_SECONDARY)
    bars.set_ylabel('Patients with exactly\nthis combination')
    bars.grid(axis='y', color=GRID_COLOR, linewidth=0.8)
    bars.set_axisbelow(True)
    _clean(bars)
    bars.tick_params(axis='x', length=0, labelbottom=False)

    # Membership matrix, members joined by a line
    for xi, (_, row) in zip(x, combinations.iterrows()):
        members = [y for y, s in enumerate(sets) if row[s]]
        matrix.scatter([xi] * len(sets), range(len(sets)), s=28, color=GRID_COLOR, zorder=1)
        if members:
            matrix.plot([xi, xi], [min(members), max(members)], color=TEXT_PRIMARY, linewidth=1.5, zorder=2)
            matrix.scatter([xi] * len(members), members, s=36, color=TEXT_PRIMARY, zorder=3)
    matrix.set_xticks([])
    for spine in matrix.spines.values():
        spine.set_visible(False)
    matrix.tick_params(length=0, labelleft=False)

    # Set sizes, e.g. suspicious TCD in 27 patients
    set_colors = [MONITORING_COLOR if s in MONITORING_TRIGGERS else CLINICAL_COLOR for s in sets]
    totals.barh(range(len(sets)), [sizes[s] for s in sets], color=set_colors, height=0.6)
    totals.set_yticks(range(len(sets)))
    totals.set_yticklabels([ITEM_LABELS[s] for s in sets])
    totals.invert_yaxis()
    totals.invert_xaxis()
    totals.set_xlabel('Patients')
    totals.spines['left'].set_visible(False)
    totals.spines['top'].set_visible(False)
    totals.tick_params(colors=TEXT_SECONDARY, axis='x')
    totals.tick_params(axis='y', length=0)
    totals.yaxis.tick_right()
    totals.tick_params(axis='y', labelsize=8.5, colors=TEXT_PRIMARY)

    _source_legend(fig, PCT_SOURCES, ncol=len(PCT_SOURCES))
    fig.suptitle(f'Triggers of the diagnostic pCT (n = {combinations["n"].sum()})', x=0.02, y=0.96, ha='left',
                 fontweight='bold', fontsize=11)
    fig.subplots_adjust(left=0.03, right=0.99, top=0.9, bottom=0.12)
    return fig


# Option 5 --------------------------------------------------------------------------------------------------------

def _option_waffle(data: FigureData):
    fig, ax = plt.subplots(figsize=(10, 4.6))
    column_start = 0
    total = int(data.paths['n'].sum())

    for group in ASSESSABILITY_ORDER:
        rows = data.paths[data.paths['assessability'] == group]
        counts = rows.groupby('source')['n'].sum().reindex(SOURCE_ORDER, fill_value=0)
        fills = [SOURCE_COLORS[source] for source, n in counts.items() for _ in range(n)]

        # Fill bottom-up, row by row, e.g. square 12 -> column 1, row 1
        columns = min(WAFFLE_COLUMNS, max(1, int(np.ceil(np.sqrt(len(fills))))))
        for index, color in enumerate(fills):
            column, row = index % columns, index // columns
            ax.add_patch(Rectangle((column_start + column, row), 1, 1, facecolor=color, edgecolor=SURFACE,
                                   linewidth=GAP_LINE_WIDTH))

        ax.text(column_start + columns / 2, -0.6, f'{_display(group)}\n{_count_label(len(fills), total)}',
                ha='center', va='top', fontsize=9, color=TEXT_PRIMARY)
        column_start += columns + WAFFLE_GROUP_GAP

    counts = data.paths.groupby('source')['n'].sum().reindex(SOURCE_ORDER, fill_value=0)
    handles = [Patch(color=SOURCE_COLORS[s], label=f'{_display(s)} ({counts[s]})') for s in SOURCE_ORDER]
    fig.legend(handles=handles, loc='upper center', ncol=len(SOURCE_ORDER), frameon=False, bbox_to_anchor=(0.5, 0.08))

    ax.set_xlim(-0.5, column_start - WAFFLE_GROUP_GAP + 0.5)
    ax.set_ylim(-3.2, total / WAFFLE_COLUMNS + 0.5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(f'Verified DCI (n = {total}), one square per patient, by clinical assessability', loc='left',
                 fontweight='bold')
    fig.tight_layout()
    return fig


PLOTTERS = {
    Option.PATHWAY_AND_TRIGGERS: _option_pathway_and_triggers,
    Option.SOURCE_BY_ASSESSABILITY: _option_source_by_assessability,
    Option.ALLUVIAL: _option_alluvial,
    Option.UPSET: _option_upset,
    Option.WAFFLE: _option_waffle,
}


def plot_option(option: Option, data: FigureData):
    """Matplotlib figure for one candidate layout."""
    return PLOTTERS[option](data)
