"""Figure 1 summary counts.

Run from code/: python -m pytest ischemia_timing/late_dci/test_figure1.py
"""
import numpy as np
import pandas as pd

from ischemia_timing.late_dci import figure1


def test_summary_counts_onset_before_day_5_and_after_day_21():
    events = pd.DataFrame({figure1.DCI: [2.0, 4.9, 5.0, 10.0, 21.0, 21.5, np.nan]})
    cts = pd.Series([1.0, 2.0], name=figure1.CT)

    summary = figure1.summarise(events, cts).set_index('distribution')

    # Day 5 itself is not 'before day 5'; day 21 itself is not 'after day 21'
    assert summary.loc[figure1.DCI, 'n_before_day_5'] == 2
    assert summary.loc[figure1.DCI, 'n_after_day_21'] == 1
