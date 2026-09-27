"""Create Table 1 (registry population overall and by DCI).

From code/:
    python -m ischemia_timing.late_dci.run_table1 [--all_years] [--data_dir ...] [--secrets ...] [--output_dir ...]

Writes table1.csv, table1.md, flow.csv and legend.md; no patient-level data.
"""
from __future__ import annotations

import argparse
import os
import warnings

from .cohort import FIRST_INCLUDED_YEAR
from .data_sources import DEFAULT_DATA_DIR, DEFAULT_SECRETS_PATH, load_sources
from .table1 import YearFilter, build_table1, population_flow

DEFAULT_OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'results', 'table1'))

# Manuscript legend; {population} depends on the year filter
LEGEND = (
    'Table 1. Baseline characteristics and outcomes of {population}, overall and by occurrence of delayed '
    'cerebral ischaemia (DCI). Values are n (%) or median (interquartile range). Percentages are calculated '
    'among patients with a known value, and the number of patients with a missing value is given in brackets; '
    'missing values were not imputed. Aneurysm location percentages refer to patients with a known location; '
    'patients with an unknown location are counted separately. Coiling includes stent-assisted procedures and '
    'was considered unknown when neither coiling nor stenting was documented as performed and at least one of '
    'the two was not recorded. Length of stay was counted from admission to ICU or hospital discharge. A missing '
    '1-year modified Rankin Scale score was replaced by the 2-year, then the 5-year score, and the score was set '
    'to 6 for patients who died. DCI status is the verified status of the DCI timings file; patients without a '
    'verified status are included in the overall population only. Outcome data were linked to the registry by study ID, or by name and date of '
    'birth when the ID was missing; one patient with conflicting duplicate outcome records was counted as '
    'missing. GCS, Glasgow Coma Scale; ICU, intensive care unit; WFNS, World Federation of Neurological Surgeons.'
)


def run(data_dir: str, secrets_path: str, output_dir: str, year_filter: YearFilter) -> None:
    os.makedirs(output_dir, exist_ok=True)
    sources = load_sources(data_dir, secrets_path)
    table = build_table1(sources, year_filter, FIRST_INCLUDED_YEAR)
    table.to_csv(os.path.join(output_dir, 'table1.csv'), index=False)
    flow = population_flow(sources.registry, sources.dci_timings, year_filter, FIRST_INCLUDED_YEAR)
    flow.to_csv(os.path.join(output_dir, 'flow.csv'), index=False)

    period = f'admission from {FIRST_INCLUDED_YEAR}' if year_filter == YearFilter.STUDY_PERIOD else 'all admission years'
    with open(os.path.join(output_dir, 'table1.md'), 'w') as file:
        file.write(f'# Table 1\n\nKSSG registry, {period}. n (%) of known values or median (Q1-Q3); [m missing] where values are missing.\n\n')
        file.write(table.rename(columns=lambda c: c.replace('\n', ' ')).to_markdown(index=False) + '\n')
        file.write('\n## Population flow\n\n' + flow.to_markdown(index=False) + '\n')

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
