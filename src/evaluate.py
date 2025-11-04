import argparse
from pathlib import Path

import numpy as np
from lifelines.utils import concordance_index
from sksurv.metrics import brier_score, cumulative_dynamic_auc

from .data_pipeline import DataConfig, load_and_prepare
from .models import load_model, predict_risk_cox, predict_risk_rsf, predict_risk_deepsurv
from .utils import save_json


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

    # Global concordance index
    cindex = float(concordance_index(ds.durations_test, -risk_test, ds.events_test))

    # Time-dependent AUC and Brier score using training as reference
    from sksurv.util import Surv

    y_train = Surv.from_arrays(event=ds.events_train.astype(bool), time=ds.durations_train.astype(float))
    y_test = Surv.from_arrays(event=ds.events_test.astype(bool), time=ds.durations_test.astype(float))

    times = np.array(args.eval_times)
    aucs, _, _ = cumulative_dynamic_auc(y_train, y_test, -risk_test, times)
    _, brier = brier_score(y_train, y_test, np.tile(-risk_test, (times.size, 1)).T, times)

    metrics = {
        "c_index": cindex,
        "times": args.eval_times,
        "time_dependent_auc": [float(x) for x in aucs],
        "brier_score": [float(x) for x in brier],
    }

    # Load risk threshold if available to compute warning rate
    cfg_path = Path(args.model_path).with_name("config.json")
    if cfg_path.exists():
        from .utils import load_json

        risk_threshold = float(load_json(cfg_path).get("risk_threshold", float(np.median(risk_test))))
        warn_rate = float((risk_test >= risk_threshold).mean())
        metrics["risk_threshold"] = risk_threshold
        metrics["warning_rate"] = warn_rate

    save_json(metrics, args.report_path)


if __name__ == "__main__":
    main()

