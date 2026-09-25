"""Raw data access: file locations, decryption and password layout."""
from __future__ import annotations

import os
from dataclasses import dataclass

import pandas as pd

from utils.utils import load_encrypted_xlsx

DEFAULT_DATA_DIR = '/mnt/data1/klug/datasets/kssg/SAH'
DEFAULT_SECRETS_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '.secrets'))

REGISTRY_FILE = 'post_hoc_modified_aSAH_DATA_2009_2023_24122023.xlsx'
OUTCOMES_FILE = 'outcomes_aSAH_DATA_2009_2024_18122024.xlsx'
DCI_TIMINGS_FILE = 'dci_timings_19092026_joint.xlsx'

# .secrets holds one password per line, e.g. line 1 -> outcomes file, line 2 -> registry
OUTCOMES_PASSWORD_LINE = 0
REGISTRY_PASSWORD_LINE = 1


@dataclass(frozen=True)
class RawSources:
    registry: pd.DataFrame
    outcomes: pd.DataFrame
    dci_timings: pd.DataFrame


def _read_passwords(secrets_path: str) -> list[str]:
    with open(secrets_path) as file:
        return [line.strip() for line in file if line.strip()]


def load_sources(data_dir: str = DEFAULT_DATA_DIR, secrets_path: str = DEFAULT_SECRETS_PATH) -> RawSources:
    passwords = _read_passwords(secrets_path)

    registry = load_encrypted_xlsx(os.path.join(data_dir, REGISTRY_FILE), password=passwords[REGISTRY_PASSWORD_LINE])
    outcomes = load_encrypted_xlsx(os.path.join(data_dir, OUTCOMES_FILE), password=passwords[OUTCOMES_PASSWORD_LINE])
    dci_timings = pd.read_excel(os.path.join(data_dir, DCI_TIMINGS_FILE))

    return RawSources(registry=registry, outcomes=outcomes, dci_timings=dci_timings)
