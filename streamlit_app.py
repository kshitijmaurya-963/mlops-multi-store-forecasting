import streamlit as st
import pandas as pd
import json
from pathlib import Path
import joblib
import subprocess
import sys
import time
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path('.')
DATA_CSV = ROOT / 'data' / 'multi_store_sales.csv'
ARTIFACTS = ROOT / 'artifacts'
REGISTRY = ROOT / 'registry'

st.set_page_config(page_title='MLOps Forecasting Dashboard', layout='wide')

st.title('MLOps Time-Series Platform — Dashboard')
st.markdown('Interact with local models, metrics, and monitoring for the multi-store forecasting scaffold.')

# --- helper functions

def load_metrics():
    p = ARTIFACTS / 'metrics.json'
    if not p.exists():
        return pd.DataFrame()
    return pd.read_json(p)


def load_monitoring():
    p = ARTIFACTS / 'monitoring.json'
    if not p.exists():
        return {}
    return json.loads(p.read_text())


def list_models():
    if not REGISTRY.exists():
        return []
    versions = sorted([p for p in REGISTRY.iterdir() if p.is_dir() and p.name.startswith('v')])
    return versions


def load_model_meta(version_path: Path):
    meta_file = version_path / 'metadata.json'
    if not meta_file.exists():
        return {}
    return json.loads(meta_file.read_text())


def predict_with_model(model_path: Path, feature_vector: list):
    m = joblib.load(model_path / 'model.joblib')
    X = np.array([feature_vector])
    pred = m.predict(X)[0]
    return float(pred)


def run_train_and_backtest():
    # runs the existing training script in a subprocess and streams output
    cmd = [sys.executable, '-m', 'src.train_backtest', '--data_dir', 'data/', '--out', 'artifacts/']
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    lines = []
    for line in proc.stdout:
        lines.append(line)
        st.session_state.train_logs = ''.join(lines)
        # yield to Streamlit
        time.sleep(0.01)
    proc.wait()
    return proc.returncode

# --- layout

sidebar_choice = st.sidebar.radio('Choose view', ['Overview', 'Models', 'Predict', 'Retrain', 'Data'])

if sidebar_choice == 'Overview':
    st.header('Overview')
    st.subheader('Backtest metrics')
    metrics = load_metrics()
    if metrics.empty:
        st.info('No metrics found. Run training/backtest to generate metrics.')
    else:
        st.dataframe(metrics)
        st.line_chart(metrics.set_index('version')[['mape','rmse']])

    st.subheader('Monitoring (drift checks)')
    mon = load_monitoring()
    if not mon:
        st.info('No monitoring output found. Run monitoring job to create artifacts/monitoring.json')
    else:
        st.json(mon)

    st.subheader('Quick dataset sample')
    if DATA_CSV.exists():
        df = pd.read_csv(DATA_CSV)
        st.dataframe(df.head(200))
    else:
        st.warning('Sample data not found at data/multi_store_sales.csv')

elif sidebar_choice == 'Models':
    st.header('Model Registry')
    versions = list_models()
    if not versions:
        st.info('No models in registry. Run training to produce registry/v001 etc.')
    else:
        cols = st.columns([1,2])
        with cols[0]:
            sel = st.selectbox('Model version', options=[p.name for p in versions])
        selected_path = REGISTRY / sel
        meta = load_model_meta(selected_path)
        st.subheader('Metadata')
        st.json(meta)

        st.subheader('Quick sanity test')
        if 'features' in meta:
            st.write('Model expects features (in order):')
            st.write(meta['features'])
            # form to input features
            with st.form('predict_form'):
                inputs = {}
                for f in meta['features']:
                    inputs[f] = st.text_input(f, value='0')
                submitted = st.form_submit_button('Run prediction')
                if submitted:
                    feat_vec = [float(inputs[f]) for f in meta['features']]
                    pred = predict_with_model(selected_path, feat_vec)
                    st.success(f'Prediction: {pred:.3f}')
        else:
            st.warning('No features metadata available: model may have been saved without a feature list')

elif sidebar_choice == 'Predict':
    st.header('Ad-hoc prediction')
    st.write('Choose model and manually enter features. If metadata has `features` saved it will be used to order inputs.')
    versions = list_models()
    model_choice = st.selectbox('Model', options=[p.name for p in versions] if versions else [])
    if model_choice:
        model_path = REGISTRY / model_choice
        meta = load_model_meta(model_path)
        feat_list = meta.get('features', None)
        if feat_list is None:
            st.info('No features list in metadata — provide a comma-separated list of names to create an input form (e.g. lag_7,lag_14,roll_mean_7)')
            names = st.text_input('Feature names (comma separated)')
            if names:
                feat_list = [n.strip() for n in names.split(',')]
        if feat_list:
            with st.form('manual_predict'):
                vals = {}
                for f in feat_list:
                    vals[f] = st.number_input(f, value=0.0, format='%.4f')
                run = st.form_submit_button('Predict')
                if run:
                    vec = [float(vals[f]) for f in feat_list]
                    pred = predict_with_model(model_path, vec)
                    st.success(f'Prediction: {pred:.3f}')

elif sidebar_choice == 'Retrain':
    st.header('Trigger retrain / rolling backtest')
    st.write('This will run the training/backtest script (may take some time locally). Logs will be displayed.')
    if 'train_logs' not in st.session_state:
        st.session_state.train_logs = ''

    if st.button('Run training/backtest'):
        st.session_state.train_logs = ''
        ret = run_train_and_backtest()
        if ret == 0:
            st.success('Training/backtest completed successfully')
        else:
            st.error(f'Training/backtest failed (exit code {ret})')

    st.subheader('Logs')
    st.code(st.session_state.train_logs[-10000:])

elif sidebar_choice == 'Data':
    st.header('Dataset & small utility')
    if DATA_CSV.exists():
        df = pd.read_csv(DATA_CSV, parse_dates=['date'])
        st.write(f'Dataset rows: {len(df)}')
        st.dataframe(df.head(200))
        if st.button('Show store/sku sample'):    
            sample = df.groupby(['store_id','sku']).size().reset_index().rename(columns={0:'count'})
            st.dataframe(sample.head(200))
    else:
        st.warning('data/multi_store_sales.csv not found')

# Footer
st.markdown('---')
st.caption('This dashboard runs locally and interacts with files under the project directory (registry/, artifacts/, data/).')
st.caption('Be careful when triggering retrain — it will run a local training job as a subprocess.')
