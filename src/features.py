"""Feature engineering for multi-series forecasting.
Includes lag and rolling features and simple encoding for store/sku.
"""
import pandas as pd
import numpy as np

def make_features(df, lags=(7,14,28), rolling_windows=(7,28)):
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['store_id','sku','date'])
    # create lag features per group
    for lag in lags:
        df[f'lag_{lag}'] = df.groupby(['store_id','sku'])['sales'].shift(lag)

    # create rolling means per group in a robust way using transform
    for w in rolling_windows:
        df[f'roll_mean_{w}'] = df.groupby(['store_id','sku'])['sales'].transform(lambda s: s.shift(1).rolling(w).mean())

    # simple target encoding: mean sales per sku and per store
    sku_mean = df.groupby('sku')['sales'].transform('mean')
    store_mean = df.groupby('store_id')['sales'].transform('mean')
    df['sku_mean'] = sku_mean
    df['store_mean'] = store_mean

    # drop rows where lag/rolling produced NaNs
    df = df.dropna().reset_index(drop=True)
    return df
