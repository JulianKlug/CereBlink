"""DCI timing vs functional outcome and DCI-related infarction: primary analysis and sensitivity analyses.

From code/:
    python -m ischemia_timing.late_dci.run_dci_timing_outcomes [--all_years] [--data_dir ...] [--secrets ...] [--output_dir ...]

Writes aggregate tables (CSV) and results.md; no patient-level data.
"""
from __future__ import annotations

import argparse
import os
import warnings

import pandas as pd

from . import dci_timing_outcomes as outcomes
from .cohort import FIRST_INCLUDED_YEAR
from .data_sources import DEFAULT_DATA_DIR, DEFAULT_SECRETS_PATH, load_sources
from .table1 import YearFilter

DEFAULT_OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'results', 'dci_timing_outcomes'))
FLOAT_FORMAT = '.3f'

NOT_ANALYSED = (
    'Editor critical 1 (clinical vs surrogate DCI ascertainment) is not analysed: the ascertainment '
    'columns of the DCI timings file are still being extracted.'
)


def _markdown(title: str, table: pd.DataFrame, note: str) -> str:
    return f'## {title}\n\n{note}\n\n' + table.to_markdown(index=False, floatfmt=FLOAT_FORMAT) + '\n'


def run(data_dir: str, secrets_path: str, output_dir: str, year_filter: YearFilter) -> None:
    os.makedirs(output_dir, exist_ok=True)
    data = outcomes.build_outcome_data(load_sources(data_dir, secrets_path), year_filter, FIRST_INCLUDED_YEAR)
    unadjusted, adjusted = outcomes.primary(data)

    tables = [
        ('unadjusted', 'Unadjusted: DCI day by outcome group', unadjusted,
         'Median (IQR) days from haemorrhage to DCI; Mann-Whitney U.'),
        ('adjusted', 'Adjusted: OR per day of DCI onset', adjusted,
         'Adjusted for age, sex, WFNS and modified Fisher. OR > 1: later DCI, higher mRS / more infarction. '
         '"submitted (probit)" reproduces the manuscript (probit on mRS > 2 with missing mRS counted as > 2; '
         'exp(coefficient) is not an odds ratio). "logistic": ordinal on mRS 0-6, missing excluded; binary '
         'for infarction. "logistic, mRS > 2": binary logistic on mRS > 2, missing excluded.'),
        ('sensitivity', 'Sensitivity analyses (ordinal / binary logistic)', outcomes.sensitivity(data),
         'Same adjustment as the primary model unless stated.'),
        ('follow_up_interval', 'Follow-up interval and mRS source', outcomes.follow_up_interval(data),
         'Editor major 4. Interval from admission to the follow-up visit used; deaths have no visit.'),
        ('follow_up_missingness', 'Missing follow-up mRS vs DCI timing', outcomes.follow_up_missingness(data),
         'Editor major 4. Logistic regression of missing mRS on DCI day; adjusted model adds age, sex, WFNS, Fisher.'),
        ('severity', 'DCI timing vs admission severity', outcomes.severity_association(data),
         'Reviewer 2. Spearman rho; linear regression of DCI day on age, sex, WFNS and Fisher (days per unit).'),
    ]

    sections = []
    for name, title, table, note in tables:
        table.to_csv(os.path.join(output_dir, f'{name}.csv'), index=False)
        sections.append(_markdown(title, table, note))

    period = f'admission from {FIRST_INCLUDED_YEAR}' if year_filter == YearFilter.STUDY_PERIOD else 'all admission years'
    with open(os.path.join(output_dir, 'results.md'), 'w') as file:
        file.write(f'# DCI timing and outcomes\n\nKSSG registry, {period}; {len(data)} DCI patients with onset date.\n\n')
        file.write('\n'.join(sections) + f'\n## Not analysed\n\n{NOT_ANALYSED}\n')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--all_years', action='store_true', help=f'do not restrict to admission from {FIRST_INCLUDED_YEAR}')
    parser.add_argument('--data_dir', default=DEFAULT_DATA_DIR)
    parser.add_argument('--secrets', default=DEFAULT_SECRETS_PATH)
    parser.add_argument('--output_dir', default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    warnings.filterwarnings('ignore', category=UserWarning)
    warnings.filterwarnings('ignore', category=FutureWarning)
    year_filter = YearFilter.ALL_YEARS if args.all_years else YearFilter.STUDY_PERIOD
    run(args.data_dir, args.secrets, args.output_dir, year_filter)
    print(f'Results written to {args.output_dir}')


if __name__ == '__main__':
    main()
