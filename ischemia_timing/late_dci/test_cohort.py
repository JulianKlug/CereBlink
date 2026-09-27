"""Landmark classification on calendar days.

Run from code/: python -m pytest ischemia_timing/late_dci/test_cohort.py
"""
import numpy as np
import pandas as pd

from ischemia_timing.late_dci import cohort
from ischemia_timing.late_dci.cohort import Event

NAN = np.nan


def _patients(t_dci):
    n = len(t_dci)
    return pd.DataFrame({
        'dci': [not np.isnan(t) for t in t_dci], 't_dci': t_dci,
        'death': [0.0] * n, 't_death': [NAN] * n, 't_discharge': [30.0] * n, 't_icu_discharge': [20.0] * n,
    })


def test_landmark_uses_calendar_day_of_dci():
    # Calendar day 7 with or without a recorded time -> early; day 8 -> late; day 21 at 18:00 -> within the cap
    patients = _patients([7.0, 7.58, 8.25, 21.75, 22.1])

    data = cohort.build_landmark_dataset(patients).dataset

    assert data.index.tolist() == [2, 3, 4]
    assert data['event'].tolist() == [Event.DCI, Event.DCI, Event.CENSORED]
    assert data['time'].tolist() == [1.0, 14.0, 14.0]


def test_piecewise_split_uses_calendar_day_of_dci():
    # DCI on calendar day 7 at 14:00 belongs to the early period
    rows = cohort.build_piecewise_dataset(_patients([7.58]))

    assert rows[['start', 'stop', 'event', 'late_period']].values.tolist() == [[0.0, 7.0, 1, 0]]


def test_split_landmark_dataset_at_follow_up_day():
    # From the landmark: DCI at time 3 -> one early row; DCI at time 10 -> early (0, 7] + late (7, 10]
    dataset = pd.DataFrame({'time': [3.0, 10.0], 'event': [Event.DCI, Event.DCI], 'aspirin': [1.0, 0.0]})

    rows = cohort.split_landmark_dataset(dataset, split_time=7.0)

    assert rows[['patient', 'start', 'stop', 'event', 'late_period']].values.tolist() == [
        [0, 0.0, 3.0, 1, 0], [1, 0.0, 7.0, 0, 0], [1, 7.0, 10.0, 1, 1]]
