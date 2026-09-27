"""Create Figure 1 (onset of vasospasm, DCI and DCI-related infarction over ICU CT acquisitions).

From code/:
    python -m ischemia_timing.late_dci.run_figure1 [--all_years] [--data_dir ...] [--secrets ...] [--output_dir ...]

Writes figure1.png, summary.csv, summary.md and legend.md; no patient-level data.
"""
from __future__ import annotations

import argparse
import os
import warnings

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .cohort import FIRST_INCLUDED_YEAR
from .data_sources import DEFAULT_DATA_DIR, DEFAULT_SECRETS_PATH, load_ct_acquisitions, load_sources
from .figure1 import build_figure1_data, ct_linkage_counts, plot_figure1, summarise
from .table1 import YearFilter, select_registry

DEFAULT_OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'results', 'figure1'))
FIGURE_NAME = 'figure1.png'
FIGURE_DPI = 900
FLOAT_FORMAT = '.1f'

# Manuscript legend; {population} depends on the year filter
LEGEND = (
    'Figure 1. Distribution of onset of vasospasm, delayed cerebral ischaemia (DCI) and DCI-related infarction '
    'in {population}. The distribution of computed tomography (CT) scans acquired during the ICU stay is plotted '
    'in the background (grey; right axis). The x-axis shows days from onset of subarachnoid haemorrhage (admission '
    'date if onset was unknown). Kernel density estimates are overlaid on every distribution.'
)


def run(data_dir: str, secrets_path: str, output_dir: str, year_filter: YearFilter) -> None:
    os.makedirs(output_dir, exist_ok=True)
    sources, ct_acquisitions = load_sources(data_dir, secrets_path), load_ct_acquisitions(data_dir)
    events, cts = build_figure1_data(sources, ct_acquisitions, year_filter, FIRST_INCLUDED_YEAR)
    linkage = ct_linkage_counts(select_registry(sources, year_filter, FIRST_INCLUDED_YEAR), ct_acquisitions)
    linkage.to_csv(os.path.join(output_dir, 'ct_linkage.csv'), index=False)

    fig = plot_figure1(events, cts)
    fig.savefig(os.path.join(output_dir, FIGURE_NAME), dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close(fig)

    summary = summarise(events, cts)
    summary.to_csv(os.path.join(output_dir, 'summary.csv'), index=False)

    period = f'admission from {FIRST_INCLUDED_YEAR}' if year_filter == YearFilter.STUDY_PERIOD else 'all admission years'
    with open(os.path.join(output_dir, 'summary.md'), 'w') as file:
        file.write(f'# Figure 1 distributions\n\nKSSG registry, {period}. Days from ictus.\n\n')
        file.write(summary.to_markdown(index=False, floatfmt=FLOAT_FORMAT) + '\n')

    population = (f'patients with aneurysmal subarachnoid haemorrhage admitted from {FIRST_INCLUDED_YEAR} onward'
                  if year_filter == YearFilter.STUDY_PERIOD else 'all patients with aneurysmal subarachnoid haemorrhage')
    with open(os.path.join(output_dir, 'legend.md'), 'w') as file:
        file.write(LEGEND.format(population=population) + '\n')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--all_years', action='store_true', help=f'do not restrict to admission from {FIRST_INCLUDED_YEAR}')
    parser.add_argument('--data_dir', default=DEFAULT_DATA_DIR)
    parser.add_argument('--secrets', default=DEFAULT_SECRETS_PATH)
    parser.add_argument('--output_dir', default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    warnings.filterwarnings('ignore', category=UserWarning)
    year_filter = YearFilter.ALL_YEARS if args.all_years else YearFilter.STUDY_PERIOD
    run(args.data_dir, args.secrets, args.output_dir, year_filter)
    print(f'Results written to {args.output_dir}')


if __name__ == '__main__':
    main()
