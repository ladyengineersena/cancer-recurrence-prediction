Cancer Recurrence Prediction (Survival Modeling)

This project predicts early recurrence risk after cancer treatment using multimodal follow-up data and survival analysis. It provides a clinician-assistive warning signal and does not prescribe treatments.

## Repository Structure

```
cancer-recurrence-prediction/
├── data/                        # (empty by default); see Data section
│   └── sample/                  # synthetic example dataset (safe)
├── notebooks/
│   ├── 01_exploratory.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_training_baseline.ipynb
│   ├── 04_survival_model.ipynb
│   └── 05_explainability.ipynb
├── src/
│   ├── data_pipeline.py
│   ├── models.py
│   ├── train.py
│   ├── evaluate.py
│   └── utils.py
├── docker/
│   └── Dockerfile
├── requirements.txt
├── LICENSE
└── CITATION.cff
```

## Data

Expected CSV files under a data directory, joined by `patient_id`:
- `clinical.csv` — clinical variables
- `lifestyle.csv` — lifestyle/monitoring variables
- `imaging_embeddings.csv` — precomputed imaging feature vectors (`img_feat_0..img_feat_N`)
- `outcomes.csv` — survival targets: `time_to_event` (days) and `event` (1=recurrence, 0=censored)

A safe, synthetic dataset exists in `data/sample/`.

## Quickstart (Local)

```bash
python -m venv .venv
. .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Train (Cox) - requires lifelines
python -m src.train --data_dir data/sample --model_type cox --output_dir outputs/cox_sample

# Evaluate
python -m src.evaluate --data_dir data/sample --model_type cox \
  --model_path outputs/cox_sample/model.joblib --report_path outputs/cox_sample/metrics.json
```

**Note:** Some models require optional dependencies:
- **Cox model**: Requires `lifelines` (included in requirements.txt)
- **RSF model**: Requires `scikit-survival` (optional, install separately)
- **DeepSurv model**: Requires `torch` (included in requirements.txt)
- **Advanced metrics**: Requires `scikit-survival` for time-dependent AUC and Brier score

## Docker

```bash
docker build -t crp:latest -f docker/Dockerfile .
docker run --rm -v $PWD/data:/app/data -v $PWD/outputs:/app/outputs crp:latest \
  python -m src.train --data_dir data/sample --model_type cox --output_dir outputs/cox_sample
```

## Ethics & Compliance

- Research and decision support only; not for autonomous diagnosis/prescription.
- Ensure compliance with privacy and governance (HIPAA/GDPR) and IRB.

## Citation

See `CITATION.cff`.

## License

See `LICENSE`.

