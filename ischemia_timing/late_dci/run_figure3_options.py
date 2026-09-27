"""Create the five Figure 3 candidates (circumstances of DCI diagnosis).

From code/:
    python -m ischemia_timing.late_dci.run_figure3_options [--data_dir ...] [--secrets ...] [--output_dir ...]

Writes option_1.png ... option_5.png and options.md; no patient-level data.
"""
from __future__ import annotations

import argparse
import os
import warnings

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from . import dci_triggers
from .cohort import AnalysisSet, build_patients, select
from .data_sources import DEFAULT_DATA_DIR, DEFAULT_SECRETS_PATH, load_sources
from .figure3_options import FigureData, Option, plot_option

DEFAULT_OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'results', 'figure3_options'))
FIGURE_DPI = 300

DESCRIPTIONS = {
    Option.PATHWAY_AND_TRIGGERS: 'A: assessability and pCT verification (stacked bars). B: frequency of each finding, dot + Wilson 95% CI.',
    Option.SOURCE_BY_ASSESSABILITY: 'Trigger source of the diagnostic pCT (clinical / monitoring / both / none), 100% bars by assessability.',
    Option.ALLUVIAL: 'Alluvial: all DCI -> clinical examination -> pCT -> trigger source; ribbons coloured by trigger source.',
    Option.UPSET: 'UpSet: exact combinations of pCT triggers; bars coloured by trigger source, set sizes left.',
    Option.WAFFLE: 'Waffle: one square per DCI patient, grouped by assessability, coloured by trigger source.',
}
NOTE = ('Clinical triggers: decreased consciousness, focal signs, unexplained persisting DoC / deficit, delirium. '
        'Monitoring triggers: ICP, TCD, PtiO2, microdialysis, NIRS; blank = not monitored = no, except TCD (every patient; '
        'blank = missing, not a trigger).')


def run(data_dir: str, secrets_path: str, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    patients = select(build_patients(load_sources(data_dir, secrets_path)), AnalysisSet.FULL_COHORT).patients
    data = FigureData(
        circumstances=dci_triggers.diagnosis_circumstances(patients),
        paths=dci_triggers.diagnostic_paths(patients),
        combinations=dci_triggers.trigger_combinations(patients),
    )

    lines = []
    for option in Option:
        fig = plot_option(option, data)
        fig.savefig(os.path.join(output_dir, f'option_{option.value}.png'), dpi=FIGURE_DPI, bbox_inches='tight')
        plt.close(fig)
        lines.append(f'{option.value}. {DESCRIPTIONS[option]}')

    with open(os.path.join(output_dir, 'options.md'), 'w') as file:
        file.write('# Figure 3 candidates\n\n' + '\n'.join(lines) + f'\n\n{NOTE}\n')


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
