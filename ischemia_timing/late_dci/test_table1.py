"""DCI image dates of the registry come from the verified DCI timings file.

Run from code/: python -m pytest ischemia_timing/late_dci/test_table1.py
"""
import numpy as np
import pandas as pd

from ischemia_timing.late_dci import table1

NAN = np.nan


def test_timings_dci_dates_replace_registry_dates():
    # Registry rows: matched by ID, matched by name + birth date, verified no DCI, absent from timings
    registry = pd.DataFrame({
        table1.ID: ['A', NAN, 'C', 'D'],
        'Name': ['a', 'b', 'c', 'd'],
        'Date_birth': pd.to_datetime(['1960-01-01', '1970-01-01', '1980-01-01', '1990-01-01']),
        'Date_DCI_ischemia_first_image': ['01.01.2015'] * 4,
    })
    timings = pd.DataFrame({
        table1.ID: ['A', NAN, 'C'],
        'Name': ['a', 'b', 'c'],
        'Date_birth': pd.to_datetime(['1960-01-01', '1970-01-01', '1980-01-01']),
        'DCI_YN_verified': [1, 1, 0],
        'Date_DCI_ischemia_first_image': ['05.01.2015', '07.01.2015', '09.01.2015'],
        'Time_DCI_ischemia_first_image': ['10:00', NAN, NAN],
        'Date_DCI_infarct_first_image': [NAN, '10.01.2015', NAN],
        'Time_DCI_infarct_first_image': [NAN] * 3,
    })

    dates = table1.timings_dci_dates(timings, registry)

    assert dates['Date_DCI_ischemia_first_image'].tolist()[:2] == ['05.01.2015', '07.01.2015']
    assert dates['Date_DCI_ischemia_first_image'].iloc[2:].isna().all()
    assert dates['Time_DCI_ischemia_first_image'].iloc[0] == '10:00'
    assert dates['Date_DCI_infarct_first_image'].iloc[1] == '10.01.2015'


def test_timings_dci_status_is_verified_status():
    # Verified DCI, verified no DCI, absent from timings -> unknown
    registry = pd.DataFrame({table1.ID: ['A', 'B', 'C'], 'Name': ['a', 'b', 'c'],
                             'Date_birth': pd.to_datetime(['1960-01-01'] * 3), 'DCI_ischemia': [0, 1, 1]})
    timings = pd.DataFrame({table1.ID: ['A', 'B'], 'Name': ['a', 'b'],
                            'Date_birth': pd.to_datetime(['1960-01-01'] * 2), 'DCI_YN_verified': [1, 0]})

    status = table1.timings_dci_status(timings, registry)

    assert status.iloc[:2].tolist() == [1, 0]
    assert np.isnan(status.iloc[2])


def test_population_flow_counts_each_step():
    # Rows: 2009 admission, unknown admission, 2015 verified DCI, 2016 verified no DCI, 2017 absent from timings
    registry = pd.DataFrame({
        table1.ID: ['A', 'B', 'C', 'D', 'E'], 'Name': list('abcde'),
        'Date_birth': pd.to_datetime(['1960-01-01'] * 5),
        'Date_admission': pd.to_datetime(['2009-05-01', None, '2015-05-01', '2016-05-01', '2017-05-01']),
    })
    timings = pd.DataFrame({table1.ID: ['A', 'C', 'D'], 'Name': list('acd'),
                            'Date_birth': pd.to_datetime(['1960-01-01'] * 3), 'DCI_YN_verified': [1, 1, 0]})

    flow = table1.population_flow(registry, timings, table1.YearFilter.STUDY_PERIOD, 2011).set_index('step')['n']

    assert flow.tolist() == [5, 4, 3, 2, 1, 1]


def test_negative_registry_age_recomputed_from_birth_date():
    # Age -45 is a sign error; recomputed from birth date to ictus (admission if ictus unknown)
    registry = pd.DataFrame({
        'Age': [50.0, -45.0, -45.0],
        'Date_birth': pd.to_datetime(['1965-01-01', '1970-01-01', '1970-01-01']),
        'Date_Ictus': pd.to_datetime(['2015-01-01', '2015-01-01', None]),
        'Date_admission': pd.to_datetime(['2015-01-01', '2015-01-01', '2015-07-01']),
    })

    age = table1._age(registry)

    assert age.round(1).tolist() == [50.0, 45.0, 45.5]
