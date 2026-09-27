"""Checks of DCI diagnosis circumstance counts on a hand-built table.

Run from code/: python -m pytest ischemia_timing/late_dci/test_dci_triggers.py
"""
import numpy as np
import pandas as pd

from ischemia_timing.late_dci import dci_triggers
from ischemia_timing.late_dci.cohort import PCT_TRIGGERS

NAN = np.nan


def _patients() -> pd.DataFrame:
    # 4 DCI patients + 1 without DCI (ignored); patient 3 assessable but not pCT verified
    patients = pd.DataFrame({
        'dci': [True, True, True, True, False],
        'clinically_assessable': [1, 1, 0, NAN, 1],
        'CTP_verification': [1, 0, 1, 1, 1],
        'decreased_consciousness': [1, 0, 1, 1, 1],
        'focal_neuro_signs': [NAN, 1, 0, 0, 1],
    })
    for trigger in PCT_TRIGGERS:
        patients[trigger] = NAN
    patients['suspect_TCD'] = [1, 1, 0, NAN, 1]
    patients['DCI_recurrence'] = 0
    patients['year'] = 2016.0
    return patients


def _row(table: pd.DataFrame, population: dci_triggers.Population, item: str) -> pd.Series:
    return table[(table['population'] == population.value) & (table['item'] == item)].iloc[0]


def test_counts_restricted_to_population():
    table = dci_triggers.diagnosis_circumstances(_patients())

    assessable = _row(table, dci_triggers.Population.ALL_DCI, 'clinically_assessable')
    assert (assessable['n_population'], assessable['n_yes'], assessable['n_recorded'], assessable['n_missing']) == (4, 2, 3, 1)

    # Only patients 1 and 2 are assessable
    consciousness = _row(table, dci_triggers.Population.CLINICALLY_ASSESSABLE, 'decreased_consciousness')
    assert (consciousness['n_population'], consciousness['n_yes'], consciousness['percent']) == (2, 1, 50.0)

    focal = _row(table, dci_triggers.Population.CLINICALLY_ASSESSABLE, 'focal_neuro_signs')
    assert (focal['n_yes'], focal['n_recorded'], focal['n_missing']) == (1, 1, 1)

    # pCT verified: patients 1, 3, 4; patient 2's TCD finding excluded; TCD universal, so patient 4's blank = missing
    tcd = _row(table, dci_triggers.Population.PCT_VERIFIED, 'suspect_TCD')
    assert (tcd['n_population'], tcd['n_yes'], tcd['n_recorded'], tcd['n_missing'], tcd['n_blank_as_no']) == (3, 1, 2, 1, 0)


def test_blank_monitoring_counts_as_no_trigger():
    table = dci_triggers.diagnosis_circumstances(_patients())

    ptio2 = _row(table, dci_triggers.Population.PCT_VERIFIED, 'decreased_ptio2')
    assert (ptio2['n_yes'], ptio2['n_recorded'], ptio2['n_missing'], ptio2['n_blank_as_no']) == (0, 3, 0, 3)
    assert ptio2['percent'] == 0.0

    # Clinical trigger: blank stays missing
    delirium = _row(table, dci_triggers.Population.PCT_VERIFIED, 'delirium')
    assert (delirium['n_recorded'], delirium['n_missing'], delirium['n_blank_as_no']) == (0, 3, 0)
    assert np.isnan(delirium['percent'])


def test_diagnostic_paths_classify_trigger_source():
    # Patient 1: clinical sign + TCD -> both; 3, 4: clinical sign only; 2: no pCT
    paths = dci_triggers.diagnostic_paths(_patients())
    counts = {(row.assessability, row.pct, row.source): row.n for row in paths.itertuples()}

    assert counts == {
        ('assessable', 'pCT', dci_triggers.TriggerSource.BOTH.value): 1,
        ('assessable', 'no pCT', dci_triggers.NO_PCT): 1,
        ('not assessable', 'pCT', dci_triggers.TriggerSource.CLINICAL_ONLY.value): 1,
        ('not recorded', 'pCT', dci_triggers.TriggerSource.CLINICAL_ONLY.value): 1,
    }


def test_trigger_combinations_count_pct_verified_only():
    combinations = dci_triggers.trigger_combinations(_patients())

    assert combinations['n'].sum() == 3
    both = combinations[combinations['clinical deterioration'] & combinations['suspect_TCD']]
    assert both['n'].tolist() == [1]


def test_device_use_counts_filled_cells_as_in_use():
    use = dci_triggers.device_use(_patients()).set_index('device')

    # TCD in every patient by protocol: in use 4/4, positive 2 of 3 recorded
    tcd = use.loc['suspect_TCD']
    assert (tcd['n_in_use'], tcd['n_recorded'], tcd['n_positive']) == (4, 3, 2)
    assert use.loc['decreased_ptio2', 'n_in_use'] == 0


def test_hodges_lehmann_shift():
    # y = x + 2 -> median difference x - y = -2
    x = np.arange(10.0)
    estimate, lower, upper = dci_triggers.hodges_lehmann(x, x + 2)
    assert estimate == -2
    assert lower < -2 < upper


def test_late_tail_uses_day_14():
    patients = _patients().assign(t_dci=[3.0, 15.0, 14.0, 20.0, NAN], DCI_recurrence=[0, 1, 0, NAN, 0])
    distribution = dci_triggers.late_tail(patients).distribution.set_index('group')

    assert distribution.loc['all DCI', 'n_late'] == 2


def test_hodges_lehmann_ci_matches_exact_moses_interval():
    # R wilcox.test(x, y, conf.int = TRUE, exact = TRUE): lower = 30th smallest difference, upper = 91st
    x = np.array([1.1, 2.3, 3.7, 4.2, 5.9, 6.4, 7.8, 8.5, 9.6, 10.3, 11.9, 12.2])
    y = np.array([0.4, 1.8, 2.9, 3.1, 4.6, 5.2, 6.7, 7.3, 8.9, 9.1])
    estimate, lower, upper = dci_triggers.hodges_lehmann(x, y)
    assert (estimate, round(lower, 6), round(upper, 6)) == (1.9, -1.1, 5.4)
