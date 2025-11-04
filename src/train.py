import argparse
from pathlib import Path

import numpy as np

from .data_pipeline import DataConfig, load_and_prepare
from .models import (
    TrainArtifacts,
    ModelType,
    train_cox,
    train_rsf,
    train_deepsurv,
    predict_risk_cox,
    predict_risk_rsf,
    predict_risk_deepsurv,
    save_model,
)
from .utils import ensure_dir


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train survival models for cancer recurrence prediction")
    p.add_argument("--data_dir", required=True, help="Directory with CSV files")
    p.add_argument("--model_type", choices=["cox", "rsf", "deepsurv"], default="cox")
    p.add_argument("--output_dir", required=True, help="Directory to save model and artifacts")
    p.add_argument("--max_epochs", type=int, default=30, help="DeepSurv: max epochs")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_dir)

    ds = load_and_prepare(DataConfig(data_dir=args.data_dir))

    if args.model_type == "cox":
        model = train_cox(ds.x_train, ds.durations_train, ds.events_train)
        risk_val = predict_risk_cox(model, ds.x_val)
    elif args.model_type == "rsf":
        model = train_rsf(ds.x_train, ds.durations_train, ds.events_train)
        risk_val = predict_risk_rsf(model, ds.x_val)
    else:
        model = train_deepsurv(ds.x_train, ds.durations_train, ds.events_train, max_epochs=args.max_epochs)
        risk_val = predict_risk_deepsurv(model, ds.x_val)

    artifacts = TrainArtifacts(model=model, model_type=args.model_type)
    save_model(artifacts, Path(args.output_dir) / "model.joblib")

    # Save a simple risk threshold suggestion (median on validation)
    threshold = float(np.median(risk_val))
    from .utils import save_json

    save_json({"risk_threshold": threshold}, Path(args.output_dir) / "config.json")


if __name__ == "__main__":
    main()

