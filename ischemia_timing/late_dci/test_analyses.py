"""Late-DCI analyses.

Run from code/: python -m pytest ischemia_timing/late_dci/test_analyses.py
"""
import pandas as pd

from ischemia_timing.late_dci import analyses
from ischemia_timing.late_dci.cohort import Event


def test_discharge_alive_before_the_cap_is_a_competing_event():
    # DCI; discharged alive at time 5; administratively censored at the cap (time 14); death
    dataset = pd.DataFrame({'time': [2.0, 5.0, 14.0, 6.0],
                            'event': [Event.DCI, Event.CENSORED, Event.CENSORED, Event.DEATH]})

    curve = analyses.cumulative_incidence_discharge_competing(dataset)

    # Final CIFs: DCI 1/4, death or discharge 2/4
    assert curve['cif_event'].iloc[-1] == 0.25
    assert curve['cif_competing'].iloc[-1] == 0.5
