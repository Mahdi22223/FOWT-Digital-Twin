import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pydmd import HankelDMD
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from scipy.signal import butter, filtfilt
import glob

# ======================================================
# 1. CONFIGURATION
# ======================================================
dt = 0.0125
fs = 1 / dt
low_cutoff = 0.25; high_cutoff = 5.0; filter_order = 4   

# MODEL SETTINGS
hankel_d = 60 
target_rank = 34  
use_opt = False   

# SPLIT SETTINGS
train_case_count = 8 

# SENSOR SETTINGS
missing_sensors = [0, 8] 

# Columns
fa_acc_cols = [f'TwHt{i}ALxt_[m/s^2]' for i in range(1, 10)]
fa_mom_cols = [f'TwHt{i}MLyt_[kN-m]' for i in range(1, 10)]

file_list = sorted(glob.glob("Case_*.csv"))
if len(file_list) < 2: raise FileNotFoundError("Need multiple files.")

# ======================================================
# 2. HELPER FUNCTIONS
# ======================================================
def apply_zero_phase_filter(data, low, high, fs):
    nyq = 0.5 * fs
    b, a = butter(filter_order, [low/nyq, high/nyq], btype='band')
    return filtfilt(b, a, data, axis=0)

def load_process_stack(files, scaler=None, fit_scaler=False):
    """Loads, Filters, Stacks, and Scales data."""
    snapshots_list = []
    for f in files:
        df = pd.read_csv(f, sep=None, engine='python')
        acc = apply_zero_phase_filter(df[fa_acc_cols].values, low_cutoff, high_cutoff, fs)
        mom = apply_zero_phase_filter(df[fa_mom_cols].values, low_cutoff, high_cutoff, fs)
        # Trim artifacts (1000 steps)
        snapshots_list.append(np.hstack((acc[1000:-1000], mom[1000:-1000])))

    X_raw = np.vstack(snapshots_list).T
    
    if fit_scaler:
        scaler = StandardScaler()
        scaler.fit(X_raw.T)
    
    X_scaled = scaler.transform(X_raw.T).T
    return X_raw, X_scaled, scaler

# ======================================================
# 3. STEP 1: TRAINING (Run Once)
# ======================================================
def train_generalization_model():
    print(f"\n{'='*20} STEP 1: TRAINING MODEL {'='*20}")
    
    train_files = file_list[:train_case_count]
    test_files = file_list[train_case_count:]
    
    print(f"Training on {len(train_files)} cases...")
    print(f"Testing set has {len(test_files)} cases (Unseen).")
    
    # Load Training Data
    _, X_train_scaled, scaler = load_process_stack(train_files, fit_scaler=True)
    
    # Fit Model
    print(f"Fitting Hankel DMD (d={hankel_d}, Rank={target_rank})...")
    hdmd = HankelDMD(svd_rank=target_rank, tlsq_rank=0, exact=True, opt=use_opt, d=hankel_d)
    hdmd.fit(X_train_scaled)
    
    # Load Test Data (Ready for prediction loop)
    print("Preparing Test Data...")
    X_test_raw, X_test_scaled, _ = load_process_stack(test_files, scaler=scaler)
    
    return {
        "model": hdmd,
        "scaler": scaler,
        "X_test_raw": X_test_raw,
        "X_test_scaled": X_test_scaled
    }

# ======================================================
# 4. STEP 2: PREDICTION LOOP (Run with different updates)
# ======================================================
def run_prediction_loop(model_data, update_every_seconds=1.0, duration=300):
    print(f"\n{'='*20} STEP 2: PREDICTION LOOP {'='*20}")
    print(f"Update Horizon: {update_every_seconds}s")
    
    # Unpack Data
    hdmd = model_data['model']
    scaler = model_data['scaler']
    X_test_scaled = model_data['X_test_scaled']
    X_test_raw = model_data['X_test_raw']
    
    # Matrices
    Phi_full = hdmd.modes
    Eigs = hdmd.eigs
    Phi_phys = Phi_full[0:18, :]
    
    # Sparse Basis Setup
    all_indices = np.arange(18)
    avail_indices = np.setdiff1d(all_indices, missing_sensors)
    
    valid_hankel_rows = []
    for lag in range(hankel_d):
        offset = lag * 18
        valid_hankel_rows.extend(avail_indices + offset)
        
    Phi_sparse = Phi_full[valid_hankel_rows, :]
    Phi_sparse_pinv = np.linalg.pinv(Phi_sparse)
    
    # Prediction Params
    total_steps = min(int(duration/dt), X_test_scaled.shape[1] - hankel_d)
    update_steps = max(1, int(update_every_seconds / dt))
    
    X_virtual_pred = np.zeros((18, total_steps))
    current_step = 0
    
    print(f"Simulating {duration}s... (Sensors {missing_sensors} Hidden)")
    
    while current_step < total_steps:
        end_step = min(current_step + update_steps, total_steps)
        pred_len = end_step - current_step
        
        # A. OBSERVE (From Test Data)
        X_hist_full = X_test_scaled[:, current_step : current_step + hankel_d]
        x_hankel_full = X_hist_full.flatten(order='F')
        
        # MASK
        x_hankel_sparse = x_hankel_full[valid_hankel_rows]
        
        # B. CALIBRATE
        b_new = np.dot(Phi_sparse_pinv, x_hankel_sparse)
        
        # C. PREDICT
        Vandermonde = np.vander(Eigs, pred_len, increasing=True)
        Dynamics_window = b_new[:, None] * Vandermonde
        
        X_pred_scaled = np.dot(Phi_phys, Dynamics_window).real
        X_pred_phys = scaler.inverse_transform(X_pred_scaled.T).T
        
        X_virtual_pred[:, current_step:end_step] = X_pred_phys
        current_step = end_step
        
    return {
        "pred": X_virtual_pred,
        "truth": X_test_raw[:, :total_steps],
        "update_s": update_every_seconds
    }

# ======================================================
# 5. STEP 3: PLOTTING (Run to visualize)
# ======================================================
def plot_prediction_results(prediction_result, plot_duration=None):
    pred = prediction_result['pred']
    truth = prediction_result['truth']
    update_s = prediction_result['update_s']
    
    total_steps = pred.shape[1]
    t = np.arange(total_steps) * dt
    
    # Handle Plot Zoom
    if plot_duration:
        plot_steps = min(int(plot_duration/dt), total_steps)
    else:
        plot_steps = total_steps
        
    print(f"\n--- Results (Update: {update_s}s) ---")
    
    for idx in missing_sensors:
        truth_s = truth[idx, :plot_steps]
        pred_s = pred[idx, :plot_steps]
        t_s = t[:plot_steps]
        
        # Metrics (Global)
        r2 = r2_score(truth[idx], pred[idx])
        rmse = np.sqrt(mean_squared_error(truth[idx], pred[idx]))
        nrmse = (rmse / (np.max(truth[idx])-np.min(truth[idx]))) * 100
        
        print(f"Sensor {idx}: R2={r2:.4f} | NRMSE={nrmse:.2f}%")
        
        plt.figure(figsize=(12, 4))
        plt.plot(t_s, truth_s, 'k-', alpha=0.5, label='Actual (Hidden)')
        plt.plot(t_s, pred_s, 'r--', linewidth=1.2, label=f'Virtual (Update {update_s}s)')
        
        plt.title(f"Generalization Test: Sensor {idx} (Zoom: {plot_duration}s)")
        plt.ylabel("Value")
        plt.xlabel("Time (s)")
        plt.legend(loc='upper right')
        plt.grid(True, alpha=0.3)
        
        # Add update lines if zoomed in enough
        if plot_duration and plot_duration < 100:
            for ut in np.arange(0, t_s[-1], update_s):
                plt.axvline(ut, color='g', linestyle=':', alpha=0.5)
                
        plt.show()

# ======================================================
# EXECUTION EXAMPLE
# ======================================================
if __name__ == "__main__":
    
    # 1. TRAIN (Do this once)
    model_data = train_generalization_model()
    
    # 2. PREDICT (Try different update speeds)
    # Experiment A: Fast Update (1.0s)
    res_1s = run_prediction_loop(model_data, update_every_seconds=1.0, duration=50)
    
    # Experiment B: Slower Update (5.0s)
    res_5s = run_prediction_loop(model_data, update_every_seconds=2, duration=50)
    
    # 3. PLOT (Compare results)
    print("\n>>> PLOTTING 1-SECOND UPDATE RESULTS <<<")
    plot_prediction_results(res_1s, plot_duration=50) # Zoom to see waves
    
    print("\n>>> PLOTTING 5-SECOND UPDATE RESULTS <<<")
    plot_prediction_results(res_5s, plot_duration=50)

# ----------------------------------------------------------------------
# LEGACY KNOWN ISSUE: the original file ended with a markdown code-fence
# (triple backticks) followed by a "### How to use this:" section. Those
# tokens are valid markdown but invalid Python and therefore caused a
# SyntaxError on import. The original block is preserved verbatim below as
# Python comments so the file is now parseable without losing provenance.
# Numerical methodology is unchanged: the train/predict/plot logic above is
# byte-for-byte identical to the original; the cleaned counterpart is
# src/eftwin/virtual_sensing.py:train_generalization_model and
# run_generalization_prediction.
#
# ```
#
# ### How to use this:
# 1.  Run the whole script. It will train the model once.
# 2.  It will automatically run predictions for **1s updates** and **5s updates**.
# 3.  It will plot both, so you can easily compare how the accuracy degrades
#     as you increase the update window, exactly as you wanted to investigate.
# ----------------------------------------------------------------------