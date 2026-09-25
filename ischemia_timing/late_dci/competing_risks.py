"""Competing-risk estimators.

Event coding: 0 = censored, 1 = event of interest, 2 = competing event.
Times are on any scale; `horizon` is on the same scale.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter

CENSORED = 0
EVENT = 1
COMPETING = 2

RISK_CLIP = 1e-10


class StepFunction:
    """Right-continuous step function, e.g. a Kaplan-Meier curve."""

    def __init__(self, x: np.ndarray, y: np.ndarray, initial: float = 1.0):
        self.x = np.asarray(x, dtype=float)
        self.y = np.asarray(y, dtype=float)
        self.initial = initial

    def at(self, t) -> np.ndarray:
        index = np.searchsorted(self.x, np.asarray(t, dtype=float), side='right') - 1
        return np.where(index < 0, self.initial, self.y[np.maximum(index, 0)])

    def before(self, t) -> np.ndarray:
        """Left limit f(t-)."""
        index = np.searchsorted(self.x, np.asarray(t, dtype=float), side='left') - 1
        return np.where(index < 0, self.initial, self.y[np.maximum(index, 0)])


def aalen_johansen(time, event) -> pd.DataFrame:
    """Cumulative incidence of both event types, one row per distinct time."""
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=int)

    survival, cif_event, cif_competing = 1.0, 0.0, 0.0
    rows = [(0.0, len(time), 0.0, 0.0, 1.0)]
    for t in np.unique(time):
        at_risk = np.sum(time >= t)
        n_event = np.sum((time == t) & (event == EVENT))
        n_competing = np.sum((time == t) & (event == COMPETING))

        cif_event += survival * n_event / at_risk
        cif_competing += survival * n_competing / at_risk
        survival *= 1 - (n_event + n_competing) / at_risk
        rows.append((t, at_risk, cif_event, cif_competing, survival))

    return pd.DataFrame(rows, columns=['time', 'at_risk', 'cif_event', 'cif_competing', 'event_free'])


def cif_at(curve: pd.DataFrame, t: float, column: str = 'cif_event') -> float:
    return float(StepFunction(curve['time'], curve[column], initial=0.0).at(t))


def censoring_survival(time, event) -> StepFunction:
    """Reverse Kaplan-Meier G(t); at tied times events precede censoring."""
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=int)

    censor_times = np.unique(time[event == CENSORED])
    values, g = [], 1.0
    for t in censor_times:
        at_risk = np.sum(time >= t) - np.sum((time == t) & (event != CENSORED))
        n_censored = np.sum((time == t) & (event == CENSORED))
        g *= 1 - n_censored / at_risk if at_risk > 0 else 1.0
        values.append(g)

    return StepFunction(censor_times, np.array(values))


@dataclass(frozen=True)
class FineGrayResult:
    summary: pd.DataFrame
    params: pd.Series


def _fine_gray_rows(time: np.ndarray, event: np.ndarray) -> pd.DataFrame:
    """Counting-process rows with IPCW weights (Geskus 2011).

    Competing-event subjects stay in the risk set after their event with weight
    G(t-)/G(T_i-), piecewise constant between censoring times. Example: competing
    death at 3, censoring at 5 -> rows (0,3] w=1, (3,5] w=G(3)/G(3-), (5,max] w=G(5)/G(3-).
    """
    censoring = censoring_survival(time, event)
    censor_times = censoring.x
    last_event_time = time[event == EVENT].max()

    rows = []
    for subject, (t, e) in enumerate(zip(time, event)):
        rows.append((subject, 0.0, t, int(e == EVENT), 1.0))
        if e != COMPETING or t >= last_event_time:
            continue

        knots = censor_times[(censor_times > t) & (censor_times < last_event_time)]
        bounds = np.concatenate([[t], knots, [last_event_time]])
        g_at_event = float(censoring.before(t))
        for start, stop in zip(bounds[:-1], bounds[1:]):
            weight = float(censoring.at(start)) / g_at_event
            if stop > start and weight > 0:
                rows.append((subject, start, stop, 0, weight))

    return pd.DataFrame(rows, columns=['subject', 'start', 'stop', 'fg_event', 'weight'])


def fine_gray(data: pd.DataFrame, duration_col: str, event_col: str, covariates: list[str]) -> FineGrayResult:
    """Fine-Gray subdistribution hazard model for event type 1, robust SEs."""
    time = data[duration_col].to_numpy(dtype=float)
    event = data[event_col].to_numpy(dtype=int)

    rows = _fine_gray_rows(time, event)
    rows = rows.join(data[covariates].reset_index(drop=True), on='subject')

    # Counting-process rows as delayed entry; sandwich SEs clustered by subject
    fitter = CoxPHFitter()
    fitter.fit(rows, duration_col='stop', event_col='fg_event', entry_col='start',
               weights_col='weight', cluster_col='subject', robust=True)
    return FineGrayResult(summary=fitter.summary, params=fitter.params_)


def _cumulative_hazard(fitter: CoxPHFitter, X: pd.DataFrame, times: np.ndarray) -> np.ndarray:
    # lifelines' baseline is at the training covariate means, matching predict_partial_hazard
    baseline = fitter.baseline_cumulative_hazard_.iloc[:, 0]
    baseline_at = StepFunction(baseline.index.to_numpy(), baseline.to_numpy(), initial=0.0).at(times)
    partial = fitter.predict_partial_hazard(X).to_numpy()
    return partial[:, None] * baseline_at[None, :]


def cause_specific_risk(event_model: CoxPHFitter, competing_model: CoxPHFitter, X: pd.DataFrame, horizon: float) -> np.ndarray:
    """Absolute risk F1(horizon | x) = sum_u S(u- | x) dA1(u | x) from two cause-specific Cox models."""
    grid = np.union1d(event_model.baseline_cumulative_hazard_.index, competing_model.baseline_cumulative_hazard_.index)
    grid = grid[grid <= horizon]

    event_hazard = _cumulative_hazard(event_model, X, grid)
    total_hazard = event_hazard + _cumulative_hazard(competing_model, X, grid)

    event_increment = np.diff(event_hazard, axis=1, prepend=0.0)
    survival_before = np.exp(-np.hstack([np.zeros((len(X), 1)), total_hazard[:, :-1]]))
    return (survival_before * event_increment).sum(axis=1)


def brier_score(time, event, risk, horizon: float) -> float:
    """IPCW Brier score for event type 1 at `horizon` (Graf 1999, competing-risk version)."""
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=int)
    risk = np.asarray(risk, dtype=float)
    censoring = censoring_survival(time, event)

    had_event = (time <= horizon) & (event != CENSORED)
    event_free = ~had_event & (time >= horizon)

    weight = np.zeros_like(time)
    weight[had_event] = 1 / censoring.before(time[had_event])
    weight[event_free] = 1 / censoring.before(horizon)

    outcome = (had_event & (event == EVENT)).astype(float)
    return float(np.mean(weight * (outcome - risk) ** 2))


def c_index(time, event, risk, horizon: float) -> float:
    """Wolbers (2009) concordance for event type 1, truncated at `horizon`.

    Pair (i, j) is comparable when i has the event by the horizon and j either
    stays event-free longer or had the competing event first.
    """
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=int)
    risk = np.asarray(risk, dtype=float)

    concordant, comparable = 0.0, 0
    for i in np.flatnonzero((event == EVENT) & (time <= horizon)):
        others = (time > time[i]) | ((event == COMPETING) & (time <= time[i]))
        comparable += int(others.sum())
        concordant += np.sum(risk[i] > risk[others]) + 0.5 * np.sum(risk[i] == risk[others])

    return concordant / comparable if comparable else np.nan


def _truncate(time: np.ndarray, event: np.ndarray, horizon: float) -> tuple[np.ndarray, np.ndarray]:
    return np.minimum(time, horizon), np.where(time <= horizon, event, CENSORED)


def calibration_slope(time, event, risk, horizon: float) -> float:
    """Fine-Gray coefficient of cloglog(predicted risk); 1 = ideal (Austin 2022)."""
    time, event = _truncate(np.asarray(time, dtype=float), np.asarray(event, dtype=int), horizon)
    risk = np.clip(np.asarray(risk, dtype=float), RISK_CLIP, 1 - RISK_CLIP)

    data = pd.DataFrame({'time': time, 'event': event, 'cloglog_risk': np.log(-np.log(1 - risk))})
    return float(fine_gray(data, 'time', 'event', ['cloglog_risk']).params['cloglog_risk'])


def calibration_by_group(time, event, risk, horizon: float, n_groups: int) -> pd.DataFrame:
    """Mean predicted vs Aalen-Johansen observed risk at `horizon`, by predicted-risk quantile group."""
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=int)
    risk = np.asarray(risk, dtype=float)
    group = pd.qcut(risk, n_groups, labels=False, duplicates='drop')

    rows = []
    for g in np.unique(group):
        member = group == g
        observed = cif_at(aalen_johansen(time[member], event[member]), horizon)
        rows.append((int(g) + 1, int(member.sum()), float(risk[member].mean()), observed))

    return pd.DataFrame(rows, columns=['group', 'n', 'predicted', 'observed'])
