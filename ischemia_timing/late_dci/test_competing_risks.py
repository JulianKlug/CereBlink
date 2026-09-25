"""Checks of the competing-risk estimators on simulated data.

Run from code/: python -m pytest ischemia_timing/late_dci/test_competing_risks.py
"""
import numpy as np
import pandas as pd
from lifelines import AalenJohansenFitter, CoxPHFitter

from ischemia_timing.late_dci import competing_risks as cr

SEED = 7
N_SUBJECTS = 2000
TRUE_SUBDISTRIBUTION_LOG_HR = 0.5
CAUSE_ONE_PLATEAU = 0.4  # Fine & Gray (1999) design: P(cause 1 | x = 0)
COMPETING_RATE = 0.3
MAX_CENSORING_TIME = 4.0
TOLERANCE_AJ = 1e-8
TOLERANCE_LOG_HR = 0.15


def _simulate_fine_gray(rng: np.random.Generator) -> pd.DataFrame:
    """Cause 1 follows a proportional subdistribution hazard model in x (Fine & Gray 1999)."""
    x = rng.binomial(1, 0.5, N_SUBJECTS).astype(float)
    hazard_ratio = np.exp(TRUE_SUBDISTRIBUTION_LOG_HR * x)
    p_cause_one = 1 - (1 - CAUSE_ONE_PLATEAU) ** hazard_ratio

    cause_one = rng.uniform(size=N_SUBJECTS) < p_cause_one

    # Invert F1(t | x) = 1 - (1 - p (1 - e^-t))^hr, conditional on cause 1
    u = rng.uniform(size=N_SUBJECTS) * p_cause_one
    t_cause_one = -np.log(1 - (1 - (1 - u) ** (1 / hazard_ratio)) / CAUSE_ONE_PLATEAU)
    t_competing = rng.exponential(1 / COMPETING_RATE, N_SUBJECTS)
    latent = np.where(cause_one, t_cause_one, t_competing)

    censoring = rng.uniform(0, MAX_CENSORING_TIME, N_SUBJECTS)
    time = np.minimum(latent, censoring)
    event = np.where(latent <= censoring, np.where(cause_one, cr.EVENT, cr.COMPETING), cr.CENSORED)
    return pd.DataFrame({'time': time, 'event': event, 'x': x})


def test_aalen_johansen_matches_lifelines():
    data = _simulate_fine_gray(np.random.default_rng(SEED))
    ours = cr.aalen_johansen(data['time'], data['event'])

    reference = AalenJohansenFitter(calculate_variance=False)
    reference.fit(data['time'], data['event'], event_of_interest=cr.EVENT)
    reference_cif = reference.cumulative_density_.iloc[:, 0]

    probe_times = [0.5, 1.0, 2.0, 3.0]
    ours_at = [cr.cif_at(ours, t) for t in probe_times]
    reference_at = [reference_cif[reference_cif.index <= t].iloc[-1] for t in probe_times]
    assert np.allclose(ours_at, reference_at, atol=TOLERANCE_AJ)


def test_fine_gray_recovers_true_coefficient():
    data = _simulate_fine_gray(np.random.default_rng(SEED))
    result = cr.fine_gray(data, 'time', 'event', ['x'])
    assert abs(result.params['x'] - TRUE_SUBDISTRIBUTION_LOG_HR) < TOLERANCE_LOG_HR


def test_cause_specific_risk_without_covariates_matches_aalen_johansen():
    data = _simulate_fine_gray(np.random.default_rng(SEED)).assign(x=0.0, noise=lambda d: np.random.default_rng(SEED).normal(size=len(d)) * 1e-6)
    horizon = 2.0

    models = []
    for cause in (cr.EVENT, cr.COMPETING):
        fitter = CoxPHFitter()
        fitter.fit(data.assign(is_event=(data['event'] == cause).astype(int))[['time', 'is_event', 'noise']],
                   duration_col='time', event_col='is_event')
        models.append(fitter)

    risk = cr.cause_specific_risk(models[0], models[1], data[['noise']], horizon)
    observed = cr.cif_at(cr.aalen_johansen(data['time'], data['event']), horizon)
    assert abs(risk.mean() - observed) < 0.01


def test_brier_score_of_perfect_prediction_is_zero():
    time = np.array([1.0, 2.0, 3.0, 5.0])
    event = np.array([cr.EVENT, cr.COMPETING, cr.EVENT, cr.CENSORED])
    horizon = 4.0
    truth = np.array([1.0, 0.0, 1.0, 0.0])
    assert cr.brier_score(time, event, truth, horizon) == 0.0


def test_c_index_orders_pairs():
    time = np.array([1.0, 2.0, 3.0])
    event = np.array([cr.EVENT, cr.EVENT, cr.CENSORED])
    assert cr.c_index(time, event, np.array([0.9, 0.5, 0.1]), horizon=5.0) == 1.0
    assert cr.c_index(time, event, np.array([0.1, 0.5, 0.9]), horizon=5.0) == 0.0
