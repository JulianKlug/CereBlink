"""Create Figure 2 (DCI onset by functional outcome and by DCI-related infarction).

From code/:
    python -m ischemia_timing.late_dci.run_figure2 [--all_years] [--submitted] [--data_dir ...] [--secrets ...] [--output_dir ...]

Writes figure2.png and legend.md; no patient-level data.
"""
from __future__ import annotations

import argparse
import os
import warnings

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .cohort import FIRST_INCLUDED_YEAR
from .data_sources import DEFAULT_DATA_DIR, DEFAULT_SECRETS_PATH, load_sources
from .dci_timing_outcomes import ModelSpec, build_outcome_data
from .figure2 import plot_figure2
from .table1 import YearFilter

DEFAULT_OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'results', 'figure2'))
FIGURE_NAME = 'figure2.png'
FIGURE_DPI = 300

# Manuscript legend (ordinal specification); {population} depends on the year filter
LEGEND = (
    'Figure 2. Onset of delayed cerebral ischaemia (DCI) by outcome in {population} who developed DCI. '
    '(A) Functional outcome at follow-up, favourable (modified Rankin Scale [mRS] ≤ 2) versus unfavourable '
    '(mRS > 2); patients without a follow-up mRS are not shown. (B) DCI-related infarction on follow-up imaging. '
    'Boxes show the median and interquartile range; whiskers extend to 1.5 times the interquartile range; '
    'outliers are not shown. p, Mann–Whitney U test; adj. p, DCI onset (per day from haemorrhage) in a '
    'multivariable ordinal logistic regression of mRS 0–6 (A) or a logistic regression of DCI-related infarction '
    '(B), adjusted for age, sex, WFNS grade and modified Fisher grade. WFNS, World Federation of Neurological '
    'Surgeons.'
)


def run(data_dir: str, secrets_path: str, output_dir: str, year_filter: YearFilter, spec: ModelSpec) -> None:
    os.makedirs(output_dir, exist_ok=True)
    data = build_outcome_data(load_sources(data_dir, secrets_path), year_filter, FIRST_INCLUDED_YEAR)

    fig = plot_figure2(data, spec)
    fig.savefig(os.path.join(output_dir, FIGURE_NAME), dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close(fig)

    if spec != ModelSpec.ORDINAL_LOGIT:
        return
    population = (f'patients with aneurysmal subarachnoid haemorrhage admitted from {FIRST_INCLUDED_YEAR} onward'
                  if year_filter == YearFilter.STUDY_PERIOD else 'patients with aneurysmal subarachnoid haemorrhage')
    with open(os.path.join(output_dir, 'legend.md'), 'w') as file:
        file.write(LEGEND.format(population=population) + '\n')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--all_years', action='store_true', help=f'do not restrict to admission from {FIRST_INCLUDED_YEAR}')
    parser.add_argument('--submitted', action='store_true', help='submitted probit models (reproduction check)')
    parser.add_argument('--data_dir', default=DEFAULT_DATA_DIR)
    parser.add_argument('--secrets', default=DEFAULT_SECRETS_PATH)
    parser.add_argument('--output_dir', default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    warnings.filterwarnings('ignore', category=UserWarning)
    warnings.filterwarnings('ignore', category=FutureWarning)
    year_filter = YearFilter.ALL_YEARS if args.all_years else YearFilter.STUDY_PERIOD
    spec = ModelSpec.SUBMITTED if args.submitted else ModelSpec.ORDINAL_LOGIT
    run(args.data_dir, args.secrets, args.output_dir, year_filter, spec)
    print(f'Results written to {args.output_dir}')


if __name__ == '__main__':
    main()
