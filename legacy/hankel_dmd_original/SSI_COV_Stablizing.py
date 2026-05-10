import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from scipy.linalg import svd, pinv
import glob

# ======================================================
# 1. CONFIGURATION
# ======================================================
dt = 0.0125
fs = 1 / dt
low_cutoff = 0.25; high_cutoff = 5.0; filter_order = 4   
ssi_i = 60  

# STABILIZATION SETTINGS
min_order = 2
max_order = 60
step_order = 2

# Tolerances for "Stability"
tol_freq = 0.01  # 1% Frequency deviation
tol_damp = 0.05  # 5% Damping deviation
tol_mac  = 0.98  # 98% MAC match

# Columns
fa_acc_cols = [f'TwHt{i}ALxt_[m/s^2]' for i in range(1, 10)]
fa_mom_cols = [f'TwHt{i}MLyt_[kN-m]' for i in range(1, 10)]

ss_acc_cols = [f'TwHt{i}ALyt_[m/s^2]' for i in range(1, 10)]
ss_mom_cols = [f'TwHt{i}MLxt_[kN-m]' for i in range(1, 10)]

file_list = sorted(glob.glob("Case_*.csv"))
if not file_list: raise FileNotFoundError("No Case_*.csv files found.")

# ======================================================
# 2. HELPER FUNCTIONS
# ======================================================
def apply_zero_phase_filter(data, low, high, fs):
    nyq = 0.5 * fs
    b, a = butter(filter_order, [low/nyq, high/nyq], btype='band')
    return filtfilt(b, a, data, axis=0)

def calculate_mac_single(v1, v2):
    v1 = v1.flatten(); v2 = v2.flatten()
    num = np.abs(np.vdot(v1, v2))**2
    den = np.vdot(v1, v1).real * np.vdot(v2, v2).real
    return num / den if den != 0 else 0.0

# ======================================================
# 3. EFFICIENT SSI CLASS
# ======================================================
class SSIMultiOrder:
    def __init__(self, dt, block_rows=60):
        self.dt = dt
        self.i = block_rows
        self.U = None
        self.S = None
        self.T1 = None
        self.n_sensors = 0
        
    def prepare_svd(self, data):
        """Computes Toeplitz and SVD once (Heavy lifting)"""
        self.n_sensors, n_samples = data.shape
        print(f"  -> Calculating Covariance & SVD (Lags={2*self.i})...")
        
        # 1. Covariance
        R_matrices = []
        for k in range(1, 2*self.i + 1):
            Y_future = data[:, k:]
            Y_past   = data[:, :-k]
            R_k = (Y_future @ Y_past.T) / (n_samples - k)
            R_matrices.append(R_k)
            
        # 2. Toeplitz
        T_rows = []
        for r in range(self.i):
            row_blk = np.hstack(R_matrices[r : r + self.i])
            T_rows.append(row_blk)
        self.T1 = np.vstack(T_rows)
        
        # 3. SVD
        self.U, self.S, _ = svd(self.T1, full_matrices=False)
        print("  -> SVD Complete.")

    def solve_for_order(self, order):
        """Extracts modes for a specific order using pre-computed SVD"""
        # Truncate
        U1 = self.U[:, :order]
        S1 = np.diag(self.S[:order])
        
        # Observability
        Oi = U1 @ np.sqrt(S1)
        rows_per_block = self.n_sensors
        O_up = Oi[: -rows_per_block, :]
        O_down = Oi[rows_per_block :, :]
        
        # Matrices
        A = pinv(O_up) @ O_down
        C = Oi[:rows_per_block, :]
        
        # Eigen
        mu, Psi = np.linalg.eig(A)
        lambda_c = np.log(mu) / self.dt
        
        freqs = np.abs(lambda_c) / (2 * np.pi)
        damps = -lambda_c.real / np.abs(lambda_c)
        mode_shapes = C @ Psi
        
        return freqs, damps, mode_shapes

# ======================================================
# 4. RUN STABILIZATION DIAGRAM
# ======================================================
def run_stabilization(direction_name, acc_cols, mom_cols):
    print(f"\n{'='*20} SSI STABILIZATION: {direction_name} {'='*20}")
    
    # Load Data
    snapshots_list = []
    for f in file_list:
        df = pd.read_csv(f, sep=None, engine='python')
        acc = apply_zero_phase_filter(df[acc_cols].values, low_cutoff, high_cutoff, fs)
        mom = apply_zero_phase_filter(df[mom_cols].values, low_cutoff, high_cutoff, fs)
        snapshots_list.append(np.hstack((acc, mom)))
    
    X = np.vstack(snapshots_list).T
    X_sub = X[:, ::4] 
    
    # Init Engine
    ssi = SSIMultiOrder(dt=dt*4, block_rows=ssi_i)
    ssi.prepare_svd(X_sub)
    
    all_poles = [] 
    
    # --- Loop through Orders ---
    prev_modes = [] 
    
    print(f"Scanning Model Orders {min_order} to {max_order}...")
    for order in range(min_order, max_order + 1, step_order):
        freqs, damps, shapes = ssi.solve_for_order(order)
        
        current_modes = []
        
        for k in range(len(freqs)):
            f = freqs[k]
            d = damps[k]
            v = shapes[:, k]
            
            # Filter Noise (Physical Constraints)
            if f < 0.25 or f > 5.0 or d < 0 or d > 0.3:
                continue
                
            # Check Stability against Previous Order
            stable_freq = False
            stable_damp = False
            stable_mac = False
            
            if prev_modes:
                # Find closest mode in previous order
                best_err = 100
                best_match = None
                
                for pm in prev_modes:
                    err = abs(f - pm['f'])
                    if err < best_err:
                        best_err = err
                        best_match = pm
                
                if best_match:
                    if abs(f - best_match['f']) / best_match['f'] < tol_freq:
                        stable_freq = True
                    if abs(d - best_match['d']) / (best_match['d'] + 1e-6) < tol_damp:
                        stable_damp = True
                    
                    mac = calculate_mac_single(v, best_match['v'])
                    if mac > tol_mac:
                        stable_mac = True
            
            # Determine Status Label
            if stable_freq and stable_damp and stable_mac:
                status = 'stable'
            elif stable_freq and stable_mac:
                status = 'stable_freq_mac'
            elif stable_freq:
                status = 'stable_freq'
            else:
                status = 'new'
            
            pole_data = {'order': order, 'f': f, 'd': d, 'v': v, 'status': status}
            all_poles.append(pole_data)
            current_modes.append(pole_data)
            
        prev_modes = current_modes 

    # --- PLOTTING ---
    plt.figure(figsize=(12, 8))
    
    orders = [p['order'] for p in all_poles]
    freqs = [p['f'] for p in all_poles]
    statuses = [p['status'] for p in all_poles]
    
    colors = []
    sizes = []
    for s in statuses:
        if s == 'stable': 
            colors.append('green'); sizes.append(30)
        elif s == 'stable_freq_mac':
            colors.append('blue'); sizes.append(20)
        elif s == 'stable_freq':
            colors.append('cyan'); sizes.append(10)
        else:
            colors.append('grey'); sizes.append(5) 
            
    plt.scatter(freqs, orders, c=colors, s=sizes, alpha=0.7)
    
    # Legend
    plt.scatter([], [], c='green', s=30, label='Stable (Freq+Damp+MAC)')
    plt.scatter([], [], c='blue', s=20, label='Stable (Freq+MAC)')
    plt.scatter([], [], c='grey', s=10, label='Unstable / Noise')
    
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Model Order")
    plt.title(f"SSI Stabilization Diagram: {direction_name}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xlim(0, 5.0)
    plt.tight_layout()
    plt.show()

# ======================================================
# 5. EXECUTION
# ======================================================
if __name__ == "__main__":
    # Run Fore-Aft
    run_stabilization("Fore-Aft", fa_acc_cols, fa_mom_cols)
    
    # Run Side-to-Side (Added this call)
    run_stabilization("Side-to-Side", ss_acc_cols, ss_mom_cols)