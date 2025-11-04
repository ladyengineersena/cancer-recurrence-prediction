from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .utils import list_feature_columns


@dataclass
class DataConfig:
    data_dir: str
    clinical_file: str = "clinical.csv"
    lifestyle_file: str = "lifestyle.csv"
    imaging_file: str = "imaging_embeddings.csv"
    outcomes_file: str = "outcomes.csv"
    imaging_prefix: str = "img_feat_"
    test_size: float = 0.2
    val_size: float = 0.2
    random_state: int = 42


@dataclass
class Dataset:
    x_train: np.ndarray
    x_val: np.ndarray
    x_test: np.ndarray
    durations_train: np.ndarray
    events_train: np.ndarray
    durations_val: np.ndarray
    events_val: np.ndarray
    durations_test: np.ndarray
    events_test: np.ndarray
    feature_names: List[str]
    scaler: StandardScaler


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def _inner_join_on_patient_id(frames: list[pd.DataFrame]) -> pd.DataFrame:
    base = frames[0]
    for f in frames[1:]:
        base = base.merge(f, on="patient_id", how="inner")
    return base


def load_and_prepare(config: DataConfig) -> Dataset:
    from sklearn.preprocessing import LabelEncoder
    
    data_dir = Path(config.data_dir)
    clinical = _read_csv(data_dir / config.clinical_file)
    lifestyle = _read_csv(data_dir / config.lifestyle_file)
    imaging = _read_csv(data_dir / config.imaging_file)
    outcomes = _read_csv(data_dir / config.outcomes_file)

    df = _inner_join_on_patient_id([clinical, lifestyle, imaging, outcomes])

    durations = df["time_to_event"].to_numpy(dtype=float)
    events = df["event"].to_numpy(dtype=int)

    exclude = {"patient_id", "time_to_event", "event"}
    candidate_features = [c for c in df.columns if c not in exclude]
    imaging_cols = list_feature_columns(config.imaging_prefix, candidate_features)
    tabular_cols = [c for c in candidate_features if c not in imaging_cols]
    
    # Encode categorical variables
    df_encoded = df.copy()
    label_encoders = {}
    for col in tabular_cols:
        if df[col].dtype == 'object' or df[col].dtype.name == 'category':
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le
    
    feature_names = tabular_cols + imaging_cols
    x = df_encoded[feature_names].to_numpy(dtype=float)

    # Stratify only if sufficient samples per class
    stratify_events = events if len(set(events)) > 1 and min(events.sum(), len(events) - events.sum()) >= 2 else None
    x_train, x_test, d_train, d_test, e_train, e_test = train_test_split(
        x, durations, events, test_size=config.test_size, random_state=config.random_state, stratify=stratify_events
    )
    stratify_train = e_train if len(set(e_train)) > 1 and min(e_train.sum(), len(e_train) - e_train.sum()) >= 2 else None
    x_train, x_val, d_train, d_val, e_train, e_val = train_test_split(
        x_train, d_train, e_train, test_size=config.val_size, random_state=config.random_state, stratify=stratify_train
    )

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_val = scaler.transform(x_val)
    x_test = scaler.transform(x_test)

    return Dataset(
        x_train=x_train,
        x_val=x_val,
        x_test=x_test,
        durations_train=d_train,
        events_train=e_train,
        durations_val=d_val,
        events_val=e_val,
        durations_test=d_test,
        events_test=e_test,
        feature_names=feature_names,
        scaler=scaler,
    )

