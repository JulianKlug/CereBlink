"""Compile the provisional supplementary material into two PDFs: every candidate item, and the core selection.

From code/ (after run_analysis, run_dci_timing_outcomes and run_imaging_use):
    python -m ischemia_timing.late_dci.run_supplement [--results_dir ...] [--output_dir ...]

Writes supplement_<selection>.md and .pdf; reads aggregate outputs only. Requires pandoc and xelatex.
"""
from __future__ import annotations

import argparse
import datetime
import os
import subprocess

from .supplement import Selection, build_markdown

DEFAULT_RESULTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'results'))
DEFAULT_OUTPUT_DIR = os.path.join(DEFAULT_RESULTS_DIR, 'supplement')
MANUSCRIPT_TITLE = 'Timing of Delayed Cerebral Ischemia and Functional Outcome After Aneurysmal Subarachnoid Hemorrhage'

SUBTITLES = {
    Selection.ALL: 'PROVISIONAL: all candidate items ([core] = planned for the final supplement)',
    Selection.CORE: 'PROVISIONAL: core items planned for the final supplement',
}

# Watermark and footer so no page can be mistaken for the final version
LATEX_HEADER = r"""
\usepackage{draftwatermark}
\SetWatermarkText{PROVISIONAL}
\SetWatermarkScale{0.7}
\SetWatermarkLightness{0.9}
\usepackage{fancyhdr}
\pagestyle{fancy}
\fancyhf{}
\fancyfoot[L]{\small Provisional, compiled %s}
\fancyfoot[R]{\small \thepage}
\renewcommand{\headrulewidth}{0pt}
\usepackage{float}
\usepackage{etoolbox}
\floatplacement{figure}{H}
\AtBeginEnvironment{longtable}{\footnotesize}
"""


def _front_matter(selection: Selection, today: str) -> str:
    return (f'---\ntitle: "Supplemental material: {MANUSCRIPT_TITLE}"\nsubtitle: "{SUBTITLES[selection]}"\n'
            f'date: "{today}"\n---\n\n')


def run(results_dir: str, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    today = datetime.date.today().isoformat()
    header_path = os.path.join(output_dir, 'header.tex')
    with open(header_path, 'w') as file:
        file.write(LATEX_HEADER % today)

    for selection in Selection:
        stem = os.path.join(output_dir, f'supplement_{selection.value}')
        with open(f'{stem}.md', 'w') as file:
            file.write(_front_matter(selection, today) + build_markdown(results_dir, selection))

        subprocess.run(['pandoc', f'{stem}.md', '-o', f'{stem}.pdf', '--pdf-engine=xelatex',
                        '-H', header_path, '-V', 'geometry:margin=2cm', '-V', 'mainfont=DejaVu Serif',
                        '-V', 'fontsize=10pt'], check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--results_dir', default=DEFAULT_RESULTS_DIR)
    parser.add_argument('--output_dir', default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    run(args.results_dir, args.output_dir)
    print(f'Supplement written to {args.output_dir}')


if __name__ == '__main__':
    main()
