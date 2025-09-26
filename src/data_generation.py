"""Generate synthetic multi-store multi-sku time-series data.
Produces a CSV with columns: date, store_id, sku, sales, promo, price, holiday
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def generate_series(start_date, periods, seed=0, base=50, trend=0.01, noise=5.0, weekly_seasonality=True):
    rng = np.random.RandomState(seed)
    t = np.arange(periods)
    series = base * (1 + trend * t) + rng.normal(0, noise, size=periods)
    if weekly_seasonality:
        series += 5 * np.sin(2 * np.pi * t / 7)
    series = np.maximum(0, series)
    return series

def main(out_dir, n_stores=20, n_skus=8, days=365*2):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    start = datetime(2020,1,1)
    for s in range(n_stores):
        for k in range(n_skus):
            seed = s*100 + k
            sales = generate_series(start, days, seed=seed, base=30+seed%10, trend=0.0008*(seed%5))
            price = 50 + (seed%10) + 2*np.sin(np.linspace(0,3.14,days)) + np.random.RandomState(seed).normal(0,1,size=days)
            promo = (np.random.RandomState(seed+1).rand(days) < 0.05).astype(int)*1
            # simple holiday flag: weekends and some fixed dates
            dates = [start + timedelta(days=i) for i in range(days)]
            holiday = [1 if (d.weekday() in (5,6) or (d.month==1 and d.day==26) or (d.month==8 and d.day==15)) else 0 for d in dates]
            for i, d in enumerate(dates):
                rows.append({
                    'date': d.strftime('%Y-%m-%d'),
                    'store_id': f'store_{s:03d}',
                    'sku': f'sku_{k:03d}',
                    'sales': float(sales[i]) + promo[i]*5 - 0.5*(price[i]-50),
                    'promo': int(promo[i]),
                    'price': float(price[i]),
                    'holiday': int(holiday[i])
                })
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / 'multi_store_sales.csv', index=False)
    print('Wrote', out_dir / 'multi_store_sales.csv')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', default='data/')
    parser.add_argument('--n_stores', type=int, default=20)
    parser.add_argument('--n_skus', type=int, default=8)
    parser.add_argument('--days', type=int, default=365*2)
    args = parser.parse_args()
    main(args.out_dir, n_stores=args.n_stores, n_skus=args.n_skus, days=args.days)
