import argparse
from pathlib import Path

import numpy as np
from lifelines.utils import concordance_index

from .data_pipeline import DataConfig, load_and_prepare
from .models import load_model, predict_risk_cox, predict_risk_rsf, predict_risk_deepsurv
from .utils import save_json

# Optional sksurv for advanced metrics
try:
    from sksurv.metrics import brier_score, cumulative_dynamic_auc
    from sksurv.util import Surv
    SKSURV_AVAILABLE = True
except ImportError:
    SKSURV_AVAILABLE = False
    brier_score = None
    cumulative_dynamic_auc = None
    Surv = None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate survival models for cancer recurrence prediction")
    p.add_argument("--data_dir", required=True)
    p.add_argument("--model_path", required=True)
    p.add_argument("--model_type", choices=["cox", "rsf", "deepsurv"], required=True)
    p.add_argument("--report_path", required=True)
    p.add_argument("--eval_times", nargs="*", type=float, default=[180, 365, 730], help="days")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ds = load_and_prepare(DataConfig(data_dir=args.data_dir))
    artifacts = load_model(args.model_path)

    if args.model_type == "cox":
        risk_test = predict_risk_cox(artifacts.model, ds.x_test)
    elif args.model_type == "rsf":
        risk_test = predict_risk_rsf(artifacts.model, ds.x_test)
    else:
        risk_test = predict_risk_deepsurv(artifacts.model, ds.x_test)

    # Global concordance index (always available)
    cindex = float(concordance_index(ds.durations_test, -risk_test, ds.events_test))

    metrics = {
        "c_index": cindex,
        "times": args.eval_times,
    }

    # Time-dependent AUC and Brier score (optional, requires scikit-survival)
    if SKSURV_AVAILABLE:
        y_train = Surv.from_arrays(event=ds.events_train.astype(bool), time=ds.durations_train.astype(float))
        y_test = Surv.from_arrays(event=ds.events_test.astype(bool), time=ds.durations_test.astype(float))

        times = np.array(args.eval_times)
        try:
            aucs, _, _ = cumulative_dynamic_auc(y_train, y_test, -risk_test, times)
            _, brier = brier_score(y_train, y_test, np.tile(-risk_test, (times.size, 1)).T, times)
            metrics["time_dependent_auc"] = [float(x) for x in aucs]
            metrics["brier_score"] = [float(x) for x in brier]
        except Exception as e:
            print(f"Warning: Could not compute time-dependent metrics: {e}")
            metrics["time_dependent_auc"] = None
            metrics["brier_score"] = None
    else:
        metrics["time_dependent_auc"] = None
        metrics["brier_score"] = None
        metrics["note"] = "Install scikit-survival for time-dependent AUC and Brier score"

    # Load risk threshold if available to compute warning rate
    cfg_path = Path(args.model_path).with_name("config.json")
    if cfg_path.exists():
        from .utils import load_json

        risk_threshold = float(load_json(cfg_path).get("risk_threshold", float(np.median(risk_test))))
        warn_rate = float((risk_test >= risk_threshold).mean())
        metrics["risk_threshold"] = risk_threshold
        metrics["warning_rate"] = warn_rate

    save_json(metrics, args.report_path)
    print(f"Evaluation complete. Metrics saved to {args.report_path}")
    print(f"C-index: {cindex:.3f}")


if __name__ == "__main__":
    main()
