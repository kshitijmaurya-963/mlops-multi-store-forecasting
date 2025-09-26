"""Simple monitoring job:
- Loads latest model metadata and artifacts/metrics.json
- Computes a lightweight drift statistic on price/promo distributions vs training snapshot using KS test
- Writes artifacts/monitoring.json with metrics and drift flags
"""
import argparse, json, os
import pandas as pd
from pathlib import Path
from scipy.stats import ks_2samp
import numpy as np

def simple_monitor(data_csv, registry_dir='registry', out='artifacts/monitoring.json'):
    df = pd.read_csv(data_csv, parse_dates=['date'])
    # load latest model metadata
    reg = Path(registry_dir)
    versions = sorted([p for p in reg.iterdir() if p.is_dir() and p.name.startswith('v')])
    latest = versions[-1] if versions else None
    meta = {}
    if latest:
        try:
            with open(latest/'metadata.json') as f:
                meta = json.load(f)
        except:
            meta = {}
    # compute distribution comparisons vs earliest training window (approx)
    snapshot = df[df['date'] < (df['date'].min() + pd.Timedelta(days=365))]
    today = df[df['date'] >= (df['date'].max() - pd.Timedelta(days=30))]
    stats = {}
    for col in ['price','promo']:
        try:
            stat = float(ks_2samp(snapshot[col].values, today[col].values).statistic)
            pval = float(ks_2samp(snapshot[col].values, today[col].values).pvalue)
        except Exception as e:
            stat, pval = None, None
        stats[col] = {'ks_stat': stat, 'pvalue': pval, 'drift': (pval is not None and pval < 0.01)}
    outd = {'model_meta': meta, 'drift_stats': stats}
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w') as f:
        json.dump(outd, f, indent=2)
    print('Wrote monitoring ->', out)
    return outd

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='data/')
    parser.add_argument('--out', default='artifacts/monitoring.json')
    args = parser.parse_args()
    simple_monitor(Path(args.data_dir)/'multi_store_sales.csv', out=args.out)
