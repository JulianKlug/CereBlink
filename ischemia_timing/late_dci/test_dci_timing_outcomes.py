"""DCI timing vs outcome models.

Run from code/: python -m pytest ischemia_timing/late_dci/test_dci_timing_outcomes.py
"""
import numpy as np
import pandas as pd

from ischemia_timing.late_dci import dci_timing_outcomes as outcomes

RNG = np.random.default_rng(0)
NAN = np.nan


def _data(n: int = 60) -> pd.DataFrame:
    dci_day = RNG.uniform(3, 15, n)
    return pd.DataFrame({
        outcomes.DCI_DAY: dci_day,
        'mrs': np.where(RNG.uniform(size=n) < 0.2, np.nan, RNG.integers(0, 7, n)),
        'age': RNG.uniform(30, 80, n),
        'male': RNG.integers(0, 2, n).astype(float),
        'wfns': RNG.integers(1, 6, n).astype(float),
        'fisher': RNG.integers(1, 5, n).astype(float),
    })


def test_crude_missingness_model_keeps_patients_with_incomplete_covariates():
    # 5 patients lack age: excluded from the adjusted model only
    data = _data()
    data.loc[:4, 'age'] = np.nan

    table = outcomes.follow_up_missingness(data).set_index('model')

    assert table.loc['crude', 'n'] == 60
    assert table.loc['adjusted', 'n'] == 55


def test_survival_restriction_keeps_patients_alive_with_dci_by_the_day():
    # alive, DCI day 10 -> kept; died day 12 after DCI day 5 -> excluded at day 14; DCI day 16 -> excluded;
    # died day 20, DCI day 14 at 18:00 (calendar day 14) -> kept
    data = pd.DataFrame({outcomes.DCI_DAY: [10.0, 5.0, 16.0, 14.75], 'death_day': [NAN, 12.0, NAN, 20.0]})

    assert outcomes.alive_with_dci_by(14)(data).tolist() == [True, False, False, True]


def test_proportional_odds_check_has_one_row_per_cumulative_threshold():
    table = outcomes.proportional_odds_check(_data(200))

    assert table['threshold'].tolist() == ['ordinal (all)', 'mRS > 0', 'mRS > 1', 'mRS > 2', 'mRS > 3', 'mRS > 4', 'mRS > 5']
    assert table['OR_per_day'].notna().all()


def test_linearity_check_compares_linear_and_quadratic_models():
    row = outcomes.linearity_check(_data(200)).iloc[0]

    assert row['df'] == 1
    assert 0 <= row['p_lr'] <= 1
    assert row['aic_quadratic'] > 0 and row['aic_linear'] > 0
