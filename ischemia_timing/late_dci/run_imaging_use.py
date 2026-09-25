"""Imaging use over calendar time: pCTs per patient and perfusion metric availability.

From code/:
    python -m ischemia_timing.late_dci.run_imaging_use [--data_dir ...] [--secrets ...] [--output_dir ...]

Writes aggregate tables (CSV), one supplementary figure (PNG) and results.md; no patient-level data.
"""
from __future__ import annotations

import argparse
import os
import warnings

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import imaging_use
from .cohort import PERFUSION_METRICS, AnalysisSet, build_patients, select
from .data_sources import DEFAULT_DATA_DIR, DEFAULT_SECRETS_PATH, load_sources

DEFAULT_OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'results', 'imaging_over_time'))
FLOAT_FORMAT = '.3f'
FIGURE_DPI = 300
FIGURE_NAME = 'imaging_over_time.png'

METRIC_LABELS = {
    'TTP_increase': 'TTP increased',
    'TTD_increase': 'TTD increased',
    'Tmax_increase': 'Tmax increased',
    'MTT_increased': 'MTT increased',
    'CBV_reduced': 'CBV reduced',
    'CBF_reduced': 'CBF reduced',
}

# Categorical slots 1-2 of the reference palette; ink and grid stay neutral
GROUP_COLORS = {'no DCI': '#2a78d6', 'DCI': '#eb6834'}
TEXT_SECONDARY = '#52514e'
GRID_COLOR = '#e4e3df'
NO_DATA_COLOR = '#f1f0ec'

# Dodge the two groups so error bars do not overlap, e.g. 2015 -> 2014.85 / 2015.15
GROUP_OFFSET = 0.15
MIN_PATIENTS_FOR_CI = 3
HEATMAP_TEXT_THRESHOLD = 60  # percent; darker cells get white text


def _plot_pct_per_year(ax, by_year: pd.DataFrame) -> None:
    for index, (group, color) in enumerate(GROUP_COLORS.items()):
        rows = by_year[by_year['group'] == group]
        x = rows['year'] + (index - 0.5) * 2 * GROUP_OFFSET
        # CI hidden for years with few patients, e.g. 2 DCI patients in 2011
        shown = rows['n'] >= MIN_PATIENTS_FOR_CI
        error = np.vstack([rows['mean_pct'] - rows['lower'], rows['upper'] - rows['mean_pct']]) * shown.to_numpy()
        ax.errorbar(x, rows['mean_pct'], yerr=error, color=color, linewidth=2, marker='o', markersize=5,
                    capsize=0, elinewidth=1, label=group)

    years = sorted(by_year['year'].unique())
    ax.set_xticks(years)
    ax.set_xticklabels(years, rotation=90)
    ax.set_ylim(bottom=0)
    ax.set_ylabel(f'Perfusion CTs per patient, mean (95% CI if n ≥ {MIN_PATIENTS_FOR_CI})')
    ax.set_xlabel('Year of haemorrhage')
    ax.grid(axis='y', color=GRID_COLOR, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ['top', 'right']:
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, loc='upper left')
    ax.set_title('A  Perfusion CTs per patient', loc='left', fontweight='bold')


def _plot_metric_availability(ax, fig, availability: pd.DataFrame, years: list[int]) -> None:
    percent = availability.pivot(index='metric', columns='year', values='percent').reindex(index=PERFUSION_METRICS, columns=years)
    counts = availability.pivot(index='metric', columns='year', values='n_available').reindex(index=PERFUSION_METRICS, columns=years)
    n_dci = availability.drop_duplicates('year').set_index('year')['n_dci'].reindex(years)

    cmap = plt.get_cmap('Blues').copy()
    cmap.set_bad(NO_DATA_COLOR)
    image = ax.imshow(np.ma.masked_invalid(percent.to_numpy(dtype=float)), cmap=cmap, vmin=0, vmax=100, aspect='auto')

    # Cell text: patients with the metric reported, e.g. '11' of n = 11 DCI patients that year
    for row, metric in enumerate(PERFUSION_METRICS):
        for column, year in enumerate(years):
            if np.isnan(percent.loc[metric, year]):
                continue
            color = 'white' if percent.loc[metric, year] >= HEATMAP_TEXT_THRESHOLD else TEXT_SECONDARY
            ax.text(column, row, int(counts.loc[metric, year]), ha='center', va='center', fontsize=7, color=color)

    ax.set_yticks(range(len(PERFUSION_METRICS)))
    ax.set_yticklabels([METRIC_LABELS[metric] for metric in PERFUSION_METRICS])
    ax.set_xticks(range(len(years)))
    ax.set_xticklabels([f'{year}\nn={0 if np.isnan(n) else int(n)}' for year, n in n_dci.items()], rotation=90, fontsize=8)
    ax.set_xlabel('Year of haemorrhage (n = patients with DCI)')
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title('B  Perfusion metrics reported at DCI diagnosis', loc='left', fontweight='bold')

    colorbar = fig.colorbar(image, ax=ax, fraction=0.04, pad=0.02)
    colorbar.set_label('% of patients with DCI')
    colorbar.outline.set_visible(False)


def _plot(by_year: pd.DataFrame, availability: pd.DataFrame, path: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={'width_ratios': [1, 1.25]})
    years = list(range(int(by_year['year'].min()), int(by_year['year'].max()) + 1))

    _plot_pct_per_year(axes[0], by_year[by_year['group'] != 'all'])
    _plot_metric_availability(axes[1], fig, availability, years)

    fig.tight_layout()
    fig.savefig(path, dpi=FIGURE_DPI)
    plt.close(fig)


def _markdown(title: str, table: pd.DataFrame, note: str = '') -> str:
    section = f'## {title}\n\n' + (f'{note}\n\n' if note else '')
    return section + table.to_markdown(index=False, floatfmt=FLOAT_FORMAT) + '\n'


def run(data_dir: str, secrets_path: str, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    patients = select(build_patients(load_sources(data_dir, secrets_path)), AnalysisSet.FULL_COHORT).patients

    tables = [
        ('pct_by_year', 'pCTs per patient by year', imaging_use.pct_by_year(patients),
         'Full cohort with known pCT count; mean with t-based 95% CI.'),
        ('pct_trend', 'Calendar-year trend in pCT use', imaging_use.pct_trend(patients),
         'Number of pCTs: negative binomial, IRR per year; adjusted model adds DCI status with hospital days '
         '(length of stay + 1) as exposure. '
         '>= 1 pCT: logistic, OR per year.'),
        ('perfusion_availability_by_year', 'Perfusion metric availability by year', imaging_use.perfusion_availability_by_year(patients),
         'Verified DCI patients; available = value recorded (not blank, not "na").'),
        ('perfusion_availability_trend', 'Calendar-year trend in perfusion metric availability',
         imaging_use.perfusion_availability_trend(patients), 'Verified DCI patients; logistic, OR per year.'),
    ]
    dci_by_year, dci_trend = imaging_use.dci_trend(patients)
    tables += [
        ('dci_by_year', 'Verified DCI by year', dci_by_year, 'Full cohort.'),
        ('dci_trend', 'Calendar-year trend in verified DCI', dci_trend, 'Full cohort; logistic, OR per year.'),
    ]

    sections = []
    for name, title, table, note in tables:
        table.to_csv(os.path.join(output_dir, f'{name}.csv'), index=False)
        sections.append(_markdown(title, table, note))

    by_year, availability = tables[0][2], tables[2][2]
    _plot(by_year, availability, os.path.join(output_dir, FIGURE_NAME))

    with open(os.path.join(output_dir, 'results.md'), 'w') as file:
        file.write('# Imaging use over calendar time\n\n' + '\n'.join(sections))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data_dir', default=DEFAULT_DATA_DIR)
    parser.add_argument('--secrets', default=DEFAULT_SECRETS_PATH)
    parser.add_argument('--output_dir', default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    warnings.filterwarnings('ignore', category=FutureWarning)
    run(args.data_dir, args.secrets, args.output_dir)
    print(f'Results written to {args.output_dir}')


if __name__ == '__main__':
    main()
