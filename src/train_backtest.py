"""Rolling backtest and model training script.
- Splits data into rolling windows
- Trains a model per-window and registers the latest model in ./registry/v{n}
- Writes evaluation metrics to artifacts/metrics.json
"""
import argparse, json, os
import pandas as pd
import numpy as np
from pathlib import Path
from src.features import make_features
from src.model import SimpleRegressor
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from datetime import datetime

def rolling_backtest(df, windows=3, window_size_days=180, step_days=90, out='artifacts'):
    from pathlib import Path as _Path
    df['date'] = pd.to_datetime(df['date'])
    last_date = df['date'].max()
    out = _Path(out); out.mkdir(parents=True, exist_ok=True)
    registry = _Path('registry'); registry.mkdir(exist_ok=True)
    metrics = []
    model_version = 0
    for w in range(windows):
        train_end = last_date - pd.Timedelta(days=w*step_days + 0)
        train_start = train_end - pd.Timedelta(days=window_size_days)
        val_start = train_end + pd.Timedelta(days=1)
        val_end = val_start + pd.Timedelta(days=step_days-1)
        train_df = df[(df['date']>=train_start) & (df['date']<=train_end)]
        val_df = df[(df['date']>=val_start) & (df['date']<=val_end)]
        if train_df.empty or val_df.empty:
            continue
        train_feat = make_features(train_df)
        val_feat = make_features(pd.concat([train_df.tail(100), val_df])) # ensure lags present
        features = [c for c in train_feat.columns if c not in ['date','store_id','sku','sales']]
        X_train = train_feat[features]; y_train = train_feat['sales']
        X_val = val_feat[features]; y_val = val_feat['sales']
        model = SimpleRegressor()
        model.fit(X_train.values, y_train.values)
        preds = model.predict(X_val.values)
        mape = float(mean_absolute_percentage_error(y_val, preds))
        mse = mean_squared_error(y_val, preds)
        rmse = float(np.sqrt(mse))

        # use timezone-aware now for timestamp
        try:
            from datetime import datetime, timezone
            ts = datetime.now(timezone.utc).isoformat()
        except Exception:
            ts = datetime.utcnow().isoformat() + 'Z'

        model_version += 1
        vpath = registry / f'v{model_version:03d}'
        meta = {'version': model_version, 'trained_at': ts, 'mape': mape, 'rmse': rmse, 'features': features}
        model.save(str(vpath))
        model.save_metadata(str(vpath), meta)
        metrics.append({'version': model_version, 'train_start': str(train_start.date()), 'train_end': str(train_end.date()), 'val_start': str(val_start.date()), 'val_end': str(val_end.date()), 'mape': mape, 'rmse': rmse})
    # write metrics.json using pathlib joining (works on Windows)
    with open(out / 'metrics.json','w') as f:
        json.dump(metrics, f, indent=2)
    print('Wrote metrics ->', out / 'metrics.json')
    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='data/')
    parser.add_argument('--out', default='artifacts/')
    args = parser.parse_args()
    df = pd.read_csv(Path(args.data_dir)/'multi_store_sales.csv')
    rolling_backtest(df, out=args.out)
