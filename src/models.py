from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import joblib
import numpy as np
from lifelines import CoxPHFitter
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv

# Optional DeepSurv
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


ModelType = Literal["cox", "rsf", "deepsurv"]


@dataclass
class TrainArtifacts:
    model: object
    model_type: ModelType


def _to_structured_surv(durations: np.ndarray, events: np.ndarray):
    return Surv.from_arrays(event=events.astype(bool), time=durations.astype(float))


def train_cox(x_train: np.ndarray, durations: np.ndarray, events: np.ndarray) -> CoxPHFitter:
    import pandas as pd

    df = pd.DataFrame(x_train, columns=[f"f_{i}" for i in range(x_train.shape[1])])
    df["duration"] = durations
    df["event"] = events
    cph = CoxPHFitter()
    cph.fit(df, duration_col="duration", event_col="event")
    return cph


def predict_risk_cox(model: CoxPHFitter, x: np.ndarray) -> np.ndarray:
    import pandas as pd

    df = pd.DataFrame(x, columns=[f"f_{i}" for i in range(x.shape[1])])
    # Partial hazard as risk score
    return model.predict_partial_hazard(df).to_numpy().ravel()


def train_rsf(x_train: np.ndarray, durations: np.ndarray, events: np.ndarray) -> RandomSurvivalForest:
    y = _to_structured_surv(durations, events)
    rsf = RandomSurvivalForest(n_estimators=200, min_samples_split=10, min_samples_leaf=5, n_jobs=-1, random_state=42)
    rsf.fit(x_train, y)
    return rsf


def predict_risk_rsf(model: RandomSurvivalForest, x: np.ndarray) -> np.ndarray:
    # Use negative expected survival time as risk surrogate
    surv_funcs = model.predict_survival_function(x)
    # Integrate survival to approximate expected time
    expected = np.array([np.trapz(fn.y, fn.x) for fn in surv_funcs])
    return -expected


class DeepSurvNet(nn.Module):
    def __init__(self, in_features: int, hidden: int = 128, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _cox_ph_loss(log_h: torch.Tensor, durations: torch.Tensor, events: torch.Tensor) -> torch.Tensor:
    # Negative partial log-likelihood
    # Sort by time descending
    order = torch.argsort(durations, descending=True)
    log_h = log_h[order]
    events = events[order]
    risk = torch.logcumsumexp(log_h, dim=0)
    pll = (log_h - risk) * events
    return -pll.sum() / events.sum().clamp_min(1.0)


def train_deepsurv(
    x_train: np.ndarray,
    durations: np.ndarray,
    events: np.ndarray,
    max_epochs: int = 30,
    batch_size: int = 64,
    lr: float = 1e-3,
) -> DeepSurvNet:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DeepSurvNet(in_features=x_train.shape[1]).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    ds = TensorDataset(
        torch.as_tensor(x_train, dtype=torch.float32),
        torch.as_tensor(durations, dtype=torch.float32),
        torch.as_tensor(events, dtype=torch.float32),
    )
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

    model.train()
    for _ in range(max_epochs):
        for xb, tb, eb in dl:
            xb, tb, eb = xb.to(device), tb.to(device), eb.to(device)
            optim.zero_grad()
            log_h = model(xb).squeeze(-1)
            loss = _cox_ph_loss(log_h, tb, eb)
            loss.backward()
            optim.step()
    return model.cpu()


def predict_risk_deepsurv(model: DeepSurvNet, x: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        scores = model(torch.as_tensor(x, dtype=torch.float32)).squeeze(-1).numpy()
    return scores


def save_model(artifacts: TrainArtifacts, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifacts, path)


def load_model(path: str | Path) -> TrainArtifacts:
    return joblib.load(path)

