#!/usr/bin/env python3
import json
import os

notebooks_data = {
    '02_preprocessing.ipynb': {
        'cells': [{
            'cell_type': 'markdown',
            'metadata': {},
            'source': [
                '# 02 - Preprocessing\n',
                '\n',
                '- Join modalities\n',
                '- Train/val/test split\n',
                '- Standardize features\n',
                '\n',
                '```python\n',
                'from src.data_pipeline import DataConfig, load_and_prepare\n',
                '\n',
                'cfg = DataConfig(data_dir="../data/sample")\n',
                'ds = load_and_prepare(cfg)\n',
                'len(ds.feature_names), ds.x_train.shape\n',
                '```'
            ]
        }]
    },
    '03_training_baseline.ipynb': {
        'cells': [{
            'cell_type': 'markdown',
            'metadata': {},
            'source': [
                '# 03 - Training Baseline\n',
                '\n',
                'Train Cox and RSF baselines and compare validation c-index.\n',
                '\n',
                '```python\n',
                'from pathlib import Path\n',
                'import numpy as np\n',
                'from src.data_pipeline import DataConfig, load_and_prepare\n',
                'from src.models import train_cox, train_rsf, predict_risk_cox, predict_risk_rsf\n',
                'from lifelines.utils import concordance_index\n',
                '\n',
                'cfg = DataConfig(data_dir="../data/sample")\n',
                'ds = load_and_prepare(cfg)\n',
                '\n',
                'cox = train_cox(ds.x_train, ds.durations_train, ds.events_train)\n',
                'rsf = train_rsf(ds.x_train, ds.durations_train, ds.events_train)\n',
                '\n',
                'risk_val_cox = predict_risk_cox(cox, ds.x_val)\n',
                'risk_val_rsf = predict_risk_rsf(rsf, ds.x_val)\n',
                '\n',
                'c_cox = concordance_index(ds.durations_val, -risk_val_cox, ds.events_val)\n',
                'c_rsf = concordance_index(ds.durations_val, -risk_val_rsf, ds.events_val)\n',
                '\n',
                'c_cox, c_rsf\n',
                '```'
            ]
        }]
    },
    '04_survival_model.ipynb': {
        'cells': [{
            'cell_type': 'markdown',
            'metadata': {},
            'source': [
                '# 04 - Survival Model (DeepSurv)\n',
                '\n',
                'Train a DeepSurv model and compare against Cox/RSF.\n',
                '\n',
                '```python\n',
                'from src.data_pipeline import DataConfig, load_and_prepare\n',
                'from src.models import train_deepsurv, predict_risk_deepsurv\n',
                'from lifelines.utils import concordance_index\n',
                '\n',
                'cfg = DataConfig(data_dir="../data/sample")\n',
                'ds = load_and_prepare(cfg)\n',
                '\n',
                'model = train_deepsurv(ds.x_train, ds.durations_train, ds.events_train, max_epochs=10)\n',
                'risk_val = predict_risk_deepsurv(model, ds.x_val)\n',
                'concordance_index(ds.durations_val, -risk_val, ds.events_val)\n',
                '```'
            ]
        }]
    },
    '05_explainability.ipynb': {
        'cells': [{
            'cell_type': 'markdown',
            'metadata': {},
            'source': [
                '# 05 - Explainability\n',
                '\n',
                '- Coefficients for Cox\n',
                '- Feature importances for RSF\n',
                '- Gradient-based saliency for DeepSurv (optional)\n',
                '\n',
                '```python\n',
                'import numpy as np\n',
                'import pandas as pd\n',
                'from src.data_pipeline import DataConfig, load_and_prepare\n',
                'from src.models import train_cox, train_rsf\n',
                '\n',
                'cfg = DataConfig(data_dir="../data/sample")\n',
                'ds = load_and_prepare(cfg)\n',
                '\n',
                'cox = train_cox(ds.x_train, ds.durations_train, ds.events_train)\n',
                'coefs = pd.Series(cox.params_.values, index=[f"f_{i}" for i in range(ds.x_train.shape[1])]).sort_values(key=abs, ascending=False)\n',
                'coefs.head(10)\n',
                '```'
            ]
        }]
    }
}

metadata = {
    'kernelspec': {
        'display_name': 'Python 3',
        'language': 'python',
        'name': 'python3'
    },
    'language_info': {
        'name': 'python'
    }
}

os.makedirs('notebooks', exist_ok=True)

for filename, notebook in notebooks_data.items():
    full_notebook = {
        'cells': notebook['cells'],
        'metadata': metadata,
        'nbformat': 4,
        'nbformat_minor': 2
    }
    with open(f'notebooks/{filename}', 'w', encoding='utf-8') as f:
        json.dump(full_notebook, f, indent=1, ensure_ascii=False)
    print(f'Created {filename}')

