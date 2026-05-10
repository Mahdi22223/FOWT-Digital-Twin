import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

def run_virtual_sensing(analysis_result, missing_sensors=[0, 8], update_every=1, duration=300):
    hdmd = analysis_result['model']
    scaler = analysis_result['scaler']
    X_true_all = analysis_result['X_raw']
    name = analysis_result['name']
    
    hankel_d = getattr(hdmd, 'd', 60)
    
    print(f"\n{'='*20} VIRTUAL SENSING TEST: {name} {'='*20}")
    print(f"Simulating failure of sensors at indices: {missing_sensors}")
    
    # 1. Setup Matrices
    # Full Modes (Stacked Hankel)
    Phi_full = hdmd.modes
    # Physical Modes (First 18 rows)
    Phi_phys_full = Phi_full[0:18, :]
    Eigs = hdmd.eigs
    Rank = Phi_full.shape[1]
    
    # 2. CREATE SPARSE BASIS (The Key Step)
    # We identify which rows in the huge Hankel matrix correspond to the AVAILABLE sensors
    # Total rows = 18 * d
    
    all_indices = np.arange(18)
    available_indices = np.setdiff1d(all_indices, missing_sensors)
    
    print(f"Available sensors: {len(available_indices)} / 18")
    
    # We need to construct a mask for the FULL Hankel stack
    # The stack is [Feature1_t0, Feature2_t0... | Feature1_t1, Feature2_t1...]
    # Wait, PyDMD stacks: [Snapshot_t, Snapshot_t+1...] vertically.
    # So the order is: 
    # Rows 0-17: t=0
    # Rows 18-35: t=1
    # ...
    
    valid_hankel_rows = []
    for lag in range(hankel_d):
        offset = lag * 18
        # Add the indices for this lag block, skipping the missing sensors
        valid_rows_for_lag = available_indices + offset
        valid_hankel_rows.extend(valid_rows_for_lag)
        
    valid_hankel_rows = np.array(valid_hankel_rows)
    
    # Slice the Mode Matrix to keep ONLY rows corresponding to available data
    Phi_sparse = Phi_full[valid_hankel_rows, :]
    
    # Pre-compute Pseudo-Inverse of the SPARSE matrix
    # This matrix map "Available Partial History" -> "Amplitudes b"
    Phi_sparse_pinv = np.linalg.pinv(Phi_sparse)
    
    # 3. ROLLING RECONSTRUCTION LOOP
    dt = 0.0125
    update_steps = int(update_every / dt)
    total_steps = min(int(duration/dt), X_true_all.shape[1] - hankel_d)
    
    X_virtual_pred = np.zeros((18, total_steps))
    
    current_step = 0
    
    while current_step < total_steps:
        end_step = min(current_step + update_steps, total_steps)
        window_len = end_step - current_step
        
        # --- A. OBSERVE (PARTIAL DATA) ---
        # Get the history window
        X_hist_full = X_true_all[:, current_step : current_step + hankel_d]
        
        # Scale (using full scaler, assuming we know the scaling factors from training)
        X_hist_scaled = scaler.transform(X_hist_full.T).T
        
        # Stack into Hankel Vector
        x_hankel_full = X_hist_scaled.flatten(order='F')
        
        # MASK THE INPUT (Simulate missing sensors)
        # We only keep the rows corresponding to valid sensors
        x_hankel_sparse = x_hankel_full[valid_hankel_rows]
        
        # --- B. CALIBRATE (Find 'b' from Partial Data) ---
        # b = pinv(Phi_sparse) * x_sparse
        b_new = np.dot(Phi_sparse_pinv, x_hankel_sparse)
        
        # --- C. PREDICT (Reconstruct FULL Data) ---
        # We use 'b' (found from sparse data) and multiply by Phi_phys_full (ALL sensors)
        # This fills in the blanks!
        
        Vandermonde = np.vander(Eigs, window_len, increasing=True)
        Dynamics_window = b_new[:, None] * Vandermonde
        
        X_pred_scaled = np.dot(Phi_phys_full, Dynamics_window).real
        X_pred_phys = scaler.inverse_transform(X_pred_scaled.T).T
        
        X_virtual_pred[:, current_step:end_step] = X_pred_phys
        current_step = end_step

    # 4. VERIFICATION
    # Verify SPECIFICALLY the missing sensors
    t = np.arange(total_steps) * dt
    
    for missing_idx in missing_sensors:
        truth = X_true_all[missing_idx, :total_steps]
        pred = X_virtual_pred[missing_idx, :total_steps]
        
        rmse = np.sqrt(mean_squared_error(truth, pred))
        r2 = r2_score(truth, pred)
        range_val = np.max(truth) - np.min(truth)
        nrmse = (rmse / range_val) * 100
        
        print(f"Sensor {missing_idx} Reconstruction: R2={r2:.4f}, NRMSE={nrmse:.2f}%")
        
        plt.figure(figsize=(12, 4))
        plt.plot(t, truth, 'k-', alpha=0.5, label='Actual (Hidden)')
        plt.plot(t, pred, 'r--', linewidth=1.2, label='Virtual Sensing (Recovered)')
        plt.title(f"Virtual Sensing Recovery: Sensor Index {missing_idx}")
        plt.ylabel("Value")
        plt.xlabel("Time (s)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

# ======================================================
# EXECUTION
# ======================================================
if __name__ == "__main__":
    # Test removing Sensor 0 (Base Accel) and Sensor 8 (Top Accel)
    # Indices 0 to 8 are Acceleration. Indices 9 to 17 are Moment.
    
    sensors_to_remove = [0, 8] 
    
    if 'results_fa' in locals():
        run_virtual_sensing(results_fa, missing_sensors=sensors_to_remove, update_every=1, duration=50)
    elif 'results_ss' in locals():
        run_virtual_sensing(results_ss, missing_sensors=sensors_to_remove, update_every=1, duration=50)
    else:
        print("Please run analysis first.")