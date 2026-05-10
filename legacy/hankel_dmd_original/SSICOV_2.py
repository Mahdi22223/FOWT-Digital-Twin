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
low_cutoff = 0.25 
high_cutoff = 5.0 
filter_order = 4   
ssi_i = 60   
model_order = 40 

fa_acc_cols = [f'TwHt{i}ALxt_[m/s^2]' for i in range(1, 10)]
fa_mom_cols = [f'TwHt{i}MLyt_[kN-m]' for i in range(1, 10)]
ss_acc_cols = [f'TwHt{i}ALyt_[m/s^2]' for i in range(1, 10)]
ss_mom_cols = [f'TwHt{i}MLxt_[kN-m]' for i in range(1, 10)]

file_list = sorted(glob.glob("Case_*.csv"))
if not file_list: raise FileNotFoundError("No Case_*.csv files found.")

# ======================================================
# 2. HELPER FUNCTIONS & CLASS
# ======================================================
def apply_zero_phase_filter(data, low, high, fs):
    nyq = 0.5 * fs
    b, a = butter(filter_order, [low/nyq, high/nyq], btype='band')
    return filtfilt(b, a, data, axis=0)

class SSICOV:
    def __init__(self, dt, block_rows=60):
        self.dt = dt
        self.i = block_rows
        
    def fit(self, data, order):
        n_sensors, n_samples = data.shape
        print(f"   -> Calculating Covariance (Lags={2*self.i})...")
        
        # 1. Correlations (R matrices)
        R_matrices = []
        for k in range(1, 2*self.i + 1):
            Y_future = data[:, k:]
            Y_past   = data[:, :-k]
            # Simple covariance estimate
            R_k = (Y_future @ Y_past.T) / (n_samples - k)
            R_matrices.append(R_k)
            
        # 2. Build Toeplitz
        T_rows = []
        for r in range(self.i):
            row_blk = np.hstack(R_matrices[r : r + self.i])
            T_rows.append(row_blk)
        T1 = np.vstack(T_rows)
        
        # 3. SVD
        print("   -> Performing SVD...")
        U, S, Vt = svd(T1)
        U1 = U[:, :order]
        S1 = np.diag(S[:order])
        
        # 4. Identification
        Oi = U1 @ np.sqrt(S1)
        rows_per_block = n_sensors
        O_up = Oi[: -rows_per_block, :]
        O_down = Oi[rows_per_block :, :]
        
        A = pinv(O_up) @ O_down
        C = Oi[:rows_per_block, :]
        
        # 5. Eigendecomposition
        mu, Psi = np.linalg.eig(A)
        lambda_c = np.log(mu) / self.dt
        
        freqs = np.abs(lambda_c) / (2 * np.pi)
        damps = -lambda_c.real / np.abs(lambda_c)
        mode_shapes = C @ Psi
        
        return freqs, damps, mode_shapes

# ======================================================
# 3. RUN FUNCTION (Plots & Returns DataFrame)
# ======================================================
def run_ssi_validation(direction_name, acc_cols, mom_cols):
    print(f"\n{'='*20} SSI-COV VALIDATION: {direction_name} {'='*20}")
    
    # 1. Load Data
    snapshots_list = []
    for f in file_list:
        df = pd.read_csv(f, sep=None, engine='python')
        acc = apply_zero_phase_filter(df[acc_cols].values, low_cutoff, high_cutoff, fs)
        mom = apply_zero_phase_filter(df[mom_cols].values, low_cutoff, high_cutoff, fs)
        snapshots_list.append(np.hstack((acc, mom)))
    
    X = np.vstack(snapshots_list).T
    X_sub = X[:, ::4] # Downsample for speed
    
    # 2. Run SSI Algorithm
    ssi = SSICOV(dt=dt*4, block_rows=ssi_i)
    freqs, damps, modes = ssi.fit(X_sub, order=model_order)
    
    # 3. Filter & Sort Results
    valid_results = []
    for idx, f in enumerate(freqs):
        d = damps[idx]
        if 0.3 < f < 5.0 and 0.0 < d < 0.2:
            valid_results.append({"Freq": f, "Damp": d, "Mode": modes[:, idx]})
            
    valid_results.sort(key=lambda x: x["Freq"])
    
    # Dedup
    clean_results = []
    seen_freqs = []
    for res in valid_results:
        if not any(abs(res["Freq"] - x) < 0.05 for x in seen_freqs):
            clean_results.append(res)
            seen_freqs.append(res["Freq"])
    
    # 4. Print Table
    print(f"\n--- SSI Identified Modes ({direction_name}) ---")
    print(f"{'Freq (Hz)':<10} | {'Damping':<10} | {'Period (s)'}")
    print("-" * 35)
    for res in clean_results:
        print(f"{res['Freq']:<10.3f} | {res['Damp']:<10.4f} | {1/res['Freq']:.3f}")
    print("-" * 35)

    # 5. Plot Mode Shapes (Displacement)
    plt.figure(figsize=(8, 6))
    tower_h = np.linspace(10, 87.6, 9)
    
    for res in clean_results[:6]:  # only plots 7 modes
        f = res['Freq']
        # Accel indices: 0-8
        phi_acc = res['Mode'][0:9]
        # Convert to Displacement
        w = 2 * np.pi * f
        phi_disp = -phi_acc / (w**2)
        
        # Normalize
        max_val = phi_disp[np.argmax(np.abs(phi_disp))]
        phi_norm = (phi_disp / max_val).real
        
        plt.plot(phi_norm, tower_h, 'o-', label=f"{f:.3f} Hz")
        
    plt.title(f"{direction_name} SSI-COV Mode Shapes")
    plt.xlabel("Norm. Displacement")
    plt.ylabel("Tower Height (m)")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # 6. Return DataFrame (Moments) for MAC
    # Moment indices: 9-17
    mode_dict = {
        f"Freq_{res['Freq']:.3f}Hz": res['Mode'][9:18] 
        for res in clean_results
    }
    
    return pd.DataFrame(mode_dict)

# ======================================================
# 4. EXECUTION
# ======================================================
if __name__ == "__main__":
    
    # Run SSI Analysis
    SSI_fa_modes = run_ssi_validation("Fore-Aft", fa_acc_cols, fa_mom_cols)
    SSI_ss_modes = run_ssi_validation("Side-to-Side", ss_acc_cols, ss_mom_cols)
    
    print("\nSSI Data Ready for MAC Analysis.")
    print("FA Modes:", SSI_fa_modes.columns.tolist())
    print("SS Modes:", SSI_ss_modes.columns.tolist())
    
df_SSI_FA = pd.DataFrame(
                SSI_fa_modes.values,
                columns=SSI_fa_modes.columns,
                index=SSI_fa_modes.index)

df_SSI_SS = pd.DataFrame(
                SSI_ss_modes.values,
                columns=SSI_ss_modes.columns,
                index=SSI_ss_modes.index)
                
    
    
    
    