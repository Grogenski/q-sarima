# =========================================================
# Q-SARIMA Experiments Source Code
# Author: Lucas Grogenski Meloca
# PhD Thesis - PPCC/UEM
# =========================================================

import os
import json
import time
import gc
import numpy as np
import pandas as pd
from tqdm import tqdm
import pmdarima as pm
from joblib import Parallel, delayed
import warnings
from scipy.optimize import minimize

# Quantum Imports
import pennylane as qml
from pennylane import numpy as pnp

warnings.filterwarnings('ignore')

# =========================================================
# 1) Parameters and Paths Definition
# =========================================================

# Mapping of cities and their respective climates
CITY_CLIMATE_MAP = {
    'AM_Campos_de_Julio_MT.json': 'Am', 'AM_Dourados_MS.json': 'Am',
    'AM_Maracaju_MS.json': 'Am', 'AM_Sapezal_MT.json': 'Am',
    'AM_Sidrolandia_MS.json': 'Am', 'AW_Campo_Novo_do_Parecis_MT.json': 'Aw',
    'AW_Diamantino_MT.json': 'Aw', 'AW_Jatai_GO.json': 'Aw',
    'AW_Lucas_do_Rio_Verde_MT.json': 'Aw', 'AW_Nova_Mutum_MT.json': 'Aw',
    'AW_Nova_Ubirata_MT.json': 'Aw', 'AW_Primavera_do_Leste_MT.json': 'Aw',
    'AW_Rio_Verde_GO.json': 'Aw', 'AW_Sao_Desiderio_BA.json': 'Aw',
    'AW_Sorriso_MT.json': 'Aw'
}

# Target Variable and Climate for the current execution
# (These parameters vary according to the specific experiment)
VARIABLES_TO_PREDICT = ['ALLSKY_SFC_SW_DWN']
TARGET_CLIMATE = 'Aw'
ITERATIONS = [(16, 2)]  # Example configuration (Input, Horizon)
SEASONAL_PERIOD = 36

# Quantum Circuit and Optimizer Parameters
MAX_QUBITS = 4
QUANTUM_OPTIMIZER_MAXITER = 50

# Hybrid Walk-Forward Validation Parameters
MIN_TRAIN_YEARS = 2
MAX_WINDOW_YEARS = 5

# Directory Paths
BASE_PATH = 'results/'
DATA_PATH = 'dataset/'
os.makedirs(BASE_PATH, exist_ok=True)
os.makedirs(DATA_PATH, exist_ok=True)

# Output Files Definition
RAW_PRED_CSV = os.path.join(BASE_PATH,
                            f'{ITERATIONS[0][0]}_{ITERATIONS[0][1]}_{TARGET_CLIMATE}_{VARIABLES_TO_PREDICT[0]}_quantum_sarima_raw_predictions.csv')
METRICS_CSV = os.path.join(BASE_PATH,
                           f'{ITERATIONS[0][0]}_{ITERATIONS[0][1]}_{TARGET_CLIMATE}_{VARIABLES_TO_PREDICT[0]}_quantum_sarima_final_metrics.csv')

# Initialize files if they do not exist
if not os.path.exists(RAW_PRED_CSV):
    pd.DataFrame(columns=['climate', 'variable', 'test_year', 'input_dec', 'decile', 'observed', 'predicted', 'qubits',
                          'order']).to_csv(RAW_PRED_CSV, index=False)
if not os.path.exists(METRICS_CSV):
    pd.DataFrame(columns=['climate', 'variable', 'input_dec', 'RMSE', 'R2_adj', 'MBE', 'avg_qubits']).to_csv(METRICS_CSV,
                                                                                                          index=False)


# =========================================================
# 2) Data Manipulation Functions
# =========================================================

def load_city_data(file_path):
    """Loads and formats the time series from the JSON file."""
    with open(file_path) as f:
        data = json.load(f)
    props = data['properties']['parameter']
    df = pd.DataFrame()
    for var in VARIABLES_TO_PREDICT:
        s = pd.Series(props[var], name=var)
        s.index = pd.to_datetime(s.index, format='%Y%m%d')
        df[var] = s
    return df


def aggregate_decendial(df):
    """Aggregates daily data into 10-day (decile) averages."""
    df = df.copy()
    df['year'] = df.index.year
    df['month'] = df.index.month
    df['day'] = df.index.day

    def get_decile(day):
        if day <= 10:
            return 1
        elif day <= 20:
            return 2
        else:
            return 3

    df['decile_month'] = df['day'].apply(get_decile)
    df['decile'] = (df['month'] - 1) * 3 + df['decile_month']

    # Group by year and decile
    dec_group = df.groupby(['year', 'decile']).mean(numeric_only=True).reset_index()
    return dec_group


def calculate_metrics(y_true, y_pred, p):
    """Calculates performance metrics: RMSE, Adjusted R2, and Mean Bias Error (MBE)."""
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    n = len(y_true)

    if n == 0: return {'RMSE': 0, 'R2_adj': 0, 'MBE': 0}

    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    r2 = 1 - ss_res / ss_tot if ss_tot != 0 else 0

    # Adjusted R2 penalized by the number of estimated parameters (p)
    if n - p - 1 > 0:
        r2_adj = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    else:
        r2_adj = r2

    mbe = np.mean(y_pred - y_true)  # Mean Bias Error (Systematic Error)
    return {'RMSE': rmse, 'R2_adj': r2_adj, 'MBE': mbe}


# =========================================================
# 3) Quantum Core and Hybrid Functions
# =========================================================

def create_quantum_circuit(n_qubits):
    """Creates the VQC (Variational Quantum Circuit) with a Hardware-Efficient ansatz."""
    dev = qml.device("lightning.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def quantum_circuit(params):
        # Rotation Layer
        for i in range(n_qubits):
            qml.RY(params[i], wires=i)
        # Entanglement Layer (linear CNOTs)
        for i in range(n_qubits - 1):
            qml.CNOT(wires=[i, i + 1])
        # Measurement in the Z basis
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    return quantum_circuit


def hybrid_objective_function(quantum_params, circuit, train_data, order, seasonal_order):
    """Hybrid cost function: in-sample MSE of SARIMA with quantum coefficients."""
    p, d, q = order
    P, D, Q, m = seasonal_order

    # Circuit execution to obtain coefficients
    coeffs = circuit(quantum_params)

    # Mapping coefficients to SARIMA polynomials
    ar_coeffs = pnp.array(coeffs[:p])
    ma_coeffs = pnp.array(coeffs[p:p + q])
    sar_coeffs = pnp.array(coeffs[p + q:p + q + P])
    sma_coeffs = pnp.array(coeffs[p + q + P:])

    # Preparation of the differenced series
    y = pd.Series(train_data)
    if D > 0: y = y.diff(m).dropna()
    if d > 0: y = y.diff(d).dropna()
    y = y.values

    y_pred = pnp.zeros_like(y, dtype=float)
    errors = pnp.zeros_like(y, dtype=float)
    max_lag = max(p, q, m * P, m * Q) + m * D + d

    # SARIMA process simulation (in-sample)
    try:
        for t in range(max_lag, len(y)):
            ar_val = pnp.sum(ar_coeffs * y[t - p:t][::-1]) if p > 0 else 0
            ma_val = pnp.sum(ma_coeffs * errors[t - q:t][::-1]) if q > 0 else 0
            sar_val = pnp.sum(sar_coeffs * y[t - m * P:t:m][::-1]) if P > 0 else 0
            sma_val = pnp.sum(sma_coeffs * errors[t - m * Q:t:m][::-1]) if Q > 0 else 0

            y_pred[t] = ar_val + sar_val + ma_val + sma_val
            errors[t] = y[t] - y_pred[t]

        mse = pnp.mean(errors[max_lag:] ** 2)
    except Exception:
        mse = 1e9  # High penalty in case of instability

    return mse


def invert_differencing(history, preds_diff, d, D, m):
    """Reverts seasonal and non-seasonal differencing to obtain the final forecast."""
    hist = pd.Series(history).reset_index(drop=True)

    # Revert seasonal differencing
    hist_seasonal = hist.copy()
    for _ in range(D):
        hist_seasonal = hist_seasonal.diff(periods=m)
    hist_seasonal = hist_seasonal.dropna().reset_index(drop=True)

    # Revert simple differencing
    levels = [hist_seasonal.copy().reset_index(drop=True)]
    for _ in range(d):
        nxt = levels[-1].diff().dropna().reset_index(drop=True)
        levels.append(nxt)

    # Reconstruction
    preds = list(preds_diff)
    for k in range(d, 0, -1):
        prev_level = levels[k - 1]
        last_value = float(prev_level.iloc[-1]) if len(prev_level) > 0 else 0.0
        inv = []
        cur = last_value
        for v in preds:
            cur = cur + v
            inv.append(cur)
        preds = inv

    history_ext = list(hist.values)
    for _ in range(D):
        inv = []
        for v in preds:
            base = history_ext[-m] if len(history_ext) >= m else 0.0
            next_val = v + base
            inv.append(next_val)
            history_ext.append(next_val)
        preds = inv

    return preds


def manual_sarima_predict(full_data, n_periods, coeffs, order, seasonal_order):
    """Generates out-of-sample forecasts using the optimized coefficients."""
    p, d, q = order
    P, D, Q, m = seasonal_order
    ar_coeffs, ma_coeffs, sar_coeffs, sma_coeffs = coeffs

    # Preparation of safe numpy arrays
    ar_coeffs = np.asarray(ar_coeffs) if ar_coeffs is not None else np.array([])
    ma_coeffs = np.asarray(ma_coeffs) if ma_coeffs is not None else np.array([])
    sar_coeffs = np.asarray(sar_coeffs) if sar_coeffs is not None else np.array([])
    sma_coeffs = np.asarray(sma_coeffs) if sma_coeffs is not None else np.array([])

    history = list(full_data.values) if hasattr(full_data, 'values') else list(full_data)

    # Apply differencing to the historical series
    y_hist = pd.Series(history)
    if D > 0:
        for _ in range(D): y_hist = y_hist.diff(m)
    if d > 0:
        for _ in range(d): y_hist = y_hist.diff()

    y_trans = list(y_hist.dropna().values)
    n_trans = len(y_trans)
    errors = [0.0] * n_trans
    max_lag = max(p, q, m * P, m * Q, 0)

    # Helper function for lags
    def get_last(arr, k, step=1):
        vals = []
        for i in range(1, k + 1):
            idx = -i * step
            if len(arr) >= i * step:
                vals.append(arr[idx])
            else:
                vals.append(0.0)
        return np.array(vals)

    # Calculation of historical residuals (needed for MA components)
    for i in range(max_lag, n_trans):
        ar_val = np.sum(ar_coeffs * get_last(y_trans[:i], p)) if p > 0 else 0.0
        ma_val = np.sum(ma_coeffs * get_last(errors[:i], q)) if q > 0 else 0.0

        sar_val = 0.0
        if P > 0:
            sar_lags = [y_trans[i - j * m] if (i - j * m) >= 0 else 0.0 for j in range(1, P + 1)]
            sar_val = np.sum(sar_coeffs * np.array(sar_lags))

        sma_val = 0.0
        if Q > 0:
            sma_lags = [errors[i - j * m] if (i - j * m) >= 0 else 0.0 for j in range(1, Q + 1)]
            sma_val = np.sum(sma_coeffs * np.array(sma_lags))

        pred_i = ar_val + sar_val + ma_val + sma_val
        errors[i] = y_trans[i] - pred_i

    # Recursive Forecasting
    y_ext = list(y_trans)
    errors_ext = list(errors)
    preds_diff = []

    for t in range(n_periods):
        ar_val = np.sum(ar_coeffs * get_last(y_ext, p)) if p > 0 else 0.0
        ma_val = np.sum(ma_coeffs * get_last(errors_ext, q)) if q > 0 else 0.0

        sar_val = 0.0
        if P > 0:
            sar_lags = [y_ext[-j * m] if len(y_ext) >= j * m else 0.0 for j in range(1, P + 1)]
            sar_val = np.sum(sar_coeffs * np.array(sar_lags))

        sma_val = 0.0
        if Q > 0:
            sma_lags = [errors_ext[-j * m] if len(errors_ext) >= j * m else 0.0 for j in range(1, Q + 1)]
            sma_val = np.sum(sma_coeffs * np.array(sma_lags))

        y_pred_diff = float(ar_val + sar_val + ma_val + sma_val)
        y_ext.append(y_pred_diff)
        errors_ext.append(0.0)  # Assuming zero error for the future
        preds_diff.append(y_pred_diff)

    # Series reconstruction
    return pd.Series(invert_differencing(history, preds_diff, d=d, D=D, m=m))


# =========================================================
# 4) Optimization and Validation Process
# =========================================================

def run_quantum_sarima_task(task_params):
    """Executes the complete training and forecasting pipeline for a given window."""
    start_time = time.time()
    climate, var, test_year, input_dec, horizon_dec, ts = (
        task_params['climate'], task_params['variable'], task_params['test_year'],
        task_params['input_dec'], task_params['horizon_dec'], task_params['time_series']
    )

    all_years = sorted(ts.index.get_level_values('year').unique())
    min_year = all_years[0]
    local_predictions = []

    # Hybrid Window Definition (Expanding -> Sliding)
    if (test_year - min_year + 1) <= MAX_WINDOW_YEARS:
        start_year = min_year
    else:
        start_year = test_year - MAX_WINDOW_YEARS + 1

    # Minimum data check
    if (test_year - start_year) < MIN_TRAIN_YEARS:
        return []

    # Selection of training data
    if start_year <= (test_year - 1):
        train_data_hist = ts.loc[start_year:(test_year - 1)]
    else:
        train_data_hist = pd.Series([], dtype=float)

    train_data_current = ts.loc[test_year, 1:input_dec] if input_dec > 0 else pd.Series([], dtype=float)
    train_data_full = pd.concat([train_data_hist, train_data_current])

    # Real data for validation
    y_true = ts.loc[test_year, (input_dec + 1):(input_dec + horizon_dec)]

    if len(train_data_full) < SEASONAL_PERIOD * MIN_TRAIN_YEARS:
        return []

    try:
        # 1. Structure Discovery (AutoARIMA)
        auto_arima_model = pm.auto_arima(
            train_data_full,
            m=SEASONAL_PERIOD,
            seasonal=True,
            max_p=MAX_QUBITS, max_q=MAX_QUBITS,
            max_P=MAX_QUBITS, max_Q=MAX_QUBITS,
            suppress_warnings=True,
            stepwise=True,
            trace=False,
            with_intercept=False,
            max_order=MAX_QUBITS
        )

        order = auto_arima_model.order
        seasonal_order = auto_arima_model.seasonal_order
        p, d, q = order
        P, D, Q, m = seasonal_order
        n_coeffs = p + q + P + Q

        if n_coeffs == 0 or n_coeffs > MAX_QUBITS:
            del auto_arima_model
            gc.collect()
            return []

        # 2. Quantum Optimization (VQC + COBYLA)
        quantum_circuit = create_quantum_circuit(n_coeffs)
        objective_func_wrapper = lambda params: hybrid_objective_function(
            params, quantum_circuit, train_data_full, order, seasonal_order
        )

        init_params = pnp.random.uniform(0, 2 * pnp.pi, size=n_coeffs, requires_grad=True)

        optimizer_result = minimize(
            objective_func_wrapper,
            init_params,
            method='COBYLA',
            options={'maxiter': QUANTUM_OPTIMIZER_MAXITER}
        )

        # Extraction of optimized coefficients
        q_coeffs_raw = quantum_circuit(optimizer_result.x)
        final_q_coeffs = {
            'ar': np.array(q_coeffs_raw[:p]),
            'ma': np.array(q_coeffs_raw[p:p + q]),
            'sar': np.array(q_coeffs_raw[p + q:p + q + P]),
            'sma': np.array(q_coeffs_raw[p + q + P:])
        }

        # 3. Out-of-Sample Forecasting
        y_pred = manual_sarima_predict(
            train_data_full, horizon_dec, list(final_q_coeffs.values()), order, seasonal_order
        )
        y_pred.index = y_true.index

        # Result storage
        for i in range(len(y_true)):
            current_decile = input_dec + 1 + i
            local_predictions.append({
                'climate': climate, 'variable': var, 'test_year': test_year,
                'input_dec': input_dec, 'decile': current_decile,
                'observed': float(y_true.iloc[i]), 'predicted': float(y_pred.iloc[i]),
                'qubits': int(n_coeffs), 'order': f"{order}{seasonal_order}"
            })

        if len(local_predictions) > 0:
            pd.DataFrame(local_predictions).to_csv(RAW_PRED_CSV, mode='a', header=False, index=False)

        # Memory cleanup
        del auto_arima_model, quantum_circuit, optimizer_result, q_coeffs_raw, final_q_coeffs, y_pred, local_predictions
        gc.collect()

        elapsed = time.time() - start_time
        return {'status': 'ok', 'climate': climate, 'var': var, 'test_year': test_year, 'time': elapsed}

    except Exception as e:
        print(f"Error in task: {climate}-{var}-{test_year}. Error: {e}")
        return {'status': 'error', 'error': str(e)}


# =========================================================
# 5) Main Execution
# =========================================================

def main():
    tasks_to_run = []

    # Data Loading and Preparation
    dfs = []
    for file, cl in CITY_CLIMATE_MAP.items():
        if cl == TARGET_CLIMATE:
            file_path = os.path.join(DATA_PATH, file)
            if os.path.exists(file_path):
                df = load_city_data(file_path)
                df_dec = aggregate_decendial(df)
                dfs.append(df_dec)
            else:
                print(f"Warning: File {file_path} not found.")

    if dfs:
        df_climate = pd.concat(dfs)
        # Average of all cities with the same climate
        df_group = df_climate.groupby(['year', 'decile']).mean(numeric_only=True).reset_index()
        all_years = sorted(df_group['year'].unique())

        # Task Generation (Walk-Forward)
        for var in VARIABLES_TO_PREDICT:
            ts = df_group.set_index(['year', 'decile'])[var].sort_index()
            for test_year in all_years:
                for input_dec, horizon_dec in ITERATIONS:
                    tasks_to_run.append({
                        'climate': TARGET_CLIMATE, 'variable': var, 'test_year': test_year,
                        'input_dec': input_dec, 'horizon_dec': horizon_dec,
                        'time_series': ts
                    })

    print(f"Starting {len(tasks_to_run)} Q-SARIMA tasks...")

    # Parallel Execution
    Parallel(n_jobs=1)(delayed(run_quantum_sarima_task)(task) for task in tqdm(tasks_to_run))

    # Final Metrics Calculation
    print("Calculating final metrics...")
    try:
        df_predictions = pd.read_csv(RAW_PRED_CSV)
        if not df_predictions.empty:
            all_final_metrics = []
            grouped = df_predictions.groupby(['climate', 'variable', 'input_dec'])

            for (climate, var, input_d), group in grouped:
                y_t = group['observed']
                y_p = group['predicted']
                p_model = int(group['qubits'].iloc[0])

                mets = calculate_metrics(y_t, y_p, p=p_model)

                final_metric = {
                    'climate': climate, 'variable': var, 'input_dec': input_d,
                    'RMSE': mets['RMSE'], 'R2_adj': mets['R2_adj'],
                    'MBE': mets['MBE'], 'avg_qubits': group['qubits'].mean()
                }
                all_final_metrics.append(final_metric)

            df_metrics = pd.DataFrame(all_final_metrics)
            df_metrics.to_csv(METRICS_CSV, index=False)
            print(df_metrics.to_string())
    except Exception as e:
        print(f"Error processing metrics: {e}")

    gc.collect()


if __name__ == "__main__":
    main()