import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pydmd import HankelDMD
from sklearn.preprocessing import StandardScaler
from scipy.signal import butter, filtfilt
import glob

# ======================================================
# 1. CONFIGURATION
# ======================================================
dt = 0.0125
fs = 1 / dt
low_cutoff = 0.25; high_cutoff = 5.0; filter_order = 4   
target_rank = 35 

# Columns
fa_acc_cols = [f'TwHt{i}ALxt_[m/s^2]' for i in range(1, 10)]
fa_mom_cols = [f'TwHt{i}MLyt_[kN-m]' for i in range(1, 10)]
ss_acc_cols = [f'TwHt{i}ALyt_[m/s^2]' for i in range(1, 10)]
ss_mom_cols = [f'TwHt{i}MLxt_[kN-m]' for i in range(1, 10)]

file_list = sorted(glob.glob("Case_*.csv"))
if not file_list: raise FileNotFoundError("No Case_*.csv files found.")

# Heights for plotting
tower_h = np.linspace(10, 87.6, 20)[[0,2,4,6,9,12,14,16,19]] 

# ======================================================
# 2. CORE FUNCTIONS
# ======================================================
def apply_zero_phase_filter(data, low, high, fs):
    nyq = 0.5 * fs
    b, a = butter(filter_order, [low/nyq, high/nyq], btype='band')
    return filtfilt(b, a, data, axis=0)

def run_analysis_param(acc_cols, mom_cols, d_val):
    # Load and Stack
    snapshots_list = []
    for f in file_list:
        df = pd.read_csv(f, sep=None, engine='python')
        acc = apply_zero_phase_filter(df[acc_cols].values, low_cutoff, high_cutoff, fs)
        mom = apply_zero_phase_filter(df[mom_cols].values, low_cutoff, high_cutoff, fs)
        snapshots_list.append(np.hstack((acc, mom)))

    X = np.vstack(snapshots_list).T
    scaler = StandardScaler()
    scaler.fit(X.T)
    X_scaled = scaler.transform(X.T).T
    
    # Fit Hankel DMD
    hdmd = HankelDMD(svd_rank=target_rank, tlsq_rank=0, exact=True, opt=False, d=d_val)
    hdmd.fit(X_scaled)
    
    return hdmd

def extract_mode_data(hdmd, target_freq, search_tol=0.1):
    """
    Extracts freq, damping, and SHAPE for the mode closest to target_freq.
    """
    eigs = hdmd.eigs
    if eigs is None: return None
    
    log_eigs = np.log(eigs)
    omega = log_eigs / dt
    freqs_hz = np.abs(omega.imag) / (2 * np.pi)
    damping = -omega.real / np.abs(omega)
    
    # Find closest mode
    # Filter first to avoid noise
    valid_idxs = np.where((freqs_hz > target_freq - search_tol) & (freqs_hz < target_freq + search_tol))[0]
    
    if len(valid_idxs) == 0:
        return None
    
    best_local = np.argmin(np.abs(freqs_hz[valid_idxs] - target_freq))
    idx = valid_idxs[best_local]
    
    # Extract Shape (Physical Displacement)
    # 1. Get full stacked mode
    phi_full = hdmd.modes[:, idx]
    # 2. Get first time step (Physical sensors)
    phi_phys = phi_full[0:18]
    # 3. Get Acceleration part (0-8)
    acc_complex = phi_phys[0:9]
    # 4. Convert to Displacement
    w = 2 * np.pi * freqs_hz[idx]
    if w == 0: w = 1e-6
    disp_complex = -acc_complex / (w**2)
    
    # 5. Normalize (Rotate to Real) for consistent plotting
    # Find max magnitude element
    max_idx = np.argmax(np.abs(disp_complex))
    # Rotate so max element is purely Real and Positive
    phase_shift = np.angle(disp_complex[max_idx])
    disp_norm = disp_complex * np.exp(-1j * phase_shift)
    # Scale so max is 1.0
    disp_norm = disp_norm / np.abs(disp_norm[max_idx])
    
    return {
        "Freq": freqs_hz[idx],
        "Damping": damping[idx],
        "Shape": disp_norm.real # Return Real part of normalized shape
    }

# ======================================================
# 3. SENSITIVITY LOOP
# ======================================================
d_range = [10, 20, 30, 35, 40, 45, 50, 55, 57,63, 65, 70, 75, 80, 90]
target_fa = 0.54
target_ss = 0.52

results_fa = []
results_ss = []
shapes_fa = [] # Store shapes for probabilistic plot
shapes_ss = []

print(f"Running Sensitivity Analysis on d={d_range}...")

for val_d in d_range:
    print(f"  -> Testing d={val_d}...", end="\r")
    
    # --- Fore-Aft ---
    mdl_fa = run_analysis_param(fa_acc_cols, fa_mom_cols, val_d)
    data_fa = extract_mode_data(mdl_fa, target_fa)
    
    if data_fa:
        results_fa.append({'d': val_d, 'Freq': data_fa['Freq'], 'Damping': data_fa['Damping']})
        # Only collect shapes if d is in the "Convergence Zone" (e.g. 45-75) to avoid bad outliers
        if 45 <= val_d <= 75:
            shapes_fa.append(data_fa['Shape'])
            
    # --- Side-to-Side ---
    mdl_ss = run_analysis_param(ss_acc_cols, ss_mom_cols, val_d)
    data_ss = extract_mode_data(mdl_ss, target_ss)
    
    if data_ss:
        results_ss.append({'d': val_d, 'Freq': data_ss['Freq'], 'Damping': data_ss['Damping']})
        if 45 <= val_d <= 75:
            shapes_ss.append(data_ss['Shape'])

df_res_fa = pd.DataFrame(results_fa)
df_res_ss = pd.DataFrame(results_ss)
print("\nDone.")

# ======================================================
# 4. PLOTTING FUNCTIONS
# ======================================================

def plot_sensitivity(df, title):
    if df.empty: return
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Frequency
    color = 'tab:blue'
    ax1.set_xlabel('Hankel Delay ($d$)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Freq (Hz)', color=color, fontsize=12, fontweight='bold')
    ax1.plot(df['d'], df['Freq'], color=color, marker='o', label='Freq')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim(0, 1.0)
    ax1.grid(True, alpha=0.3)

    # Damping
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Damping Ratio', color=color, fontsize=12, fontweight='bold')
    ax2.plot(df['d'], df['Damping'], color=color, marker='s', linestyle='--', label='Damping')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim(0, 0.5)
    
    # Highlight Region
    plt.axvspan(40, 63, color='gray', alpha=0.15, label='Convergence Zone')
    plt.axvline(60, color='k', linestyle=':', label='Selected d=60')
    
    plt.title(f"{title}: Parameter Convergence", fontsize=14)
    plt.tight_layout()
    plt.show()

def plot_probabilistic_shape(shape_list, title):
    """
    Plots Mean Shape + Confidence Interval from the collected shapes.
    """
    if not shape_list: return
    
    # Convert list to matrix (N_samples x N_sensors)
    # shape_list is list of arrays of shape (9,)
    S = np.array(shape_list) 
    
    # Calculate Statistics
    mean_shape = np.mean(S, axis=0)
    std_shape = np.std(S, axis=0)
    
    # 95% Confidence Interval (approx 2 std devs)
    ci_upper = mean_shape + 1.96 * std_shape
    ci_lower = mean_shape - 1.96 * std_shape
    
    plt.figure(figsize=(8, 10))
    
    # 1. Plot Individual Samples (Light Grey)
    for s in S:
        plt.plot(s, tower_h, color='gray', alpha=0.3, linewidth=1)
        
    # 2. Plot Confidence Interval (Shaded)
    plt.fill_betweenx(tower_h, ci_lower, ci_upper, color='blue', alpha=0.2, label='95% Confidence (Variance due to d)')
    
    # 3. Plot Mean (Bold)
    plt.plot(mean_shape, tower_h, 'b-o', linewidth=3, label='Mean Mode Shape')
    
    plt.title(f"{title}: Probabilistic Mode Shape\n(Derived from Convergence Zone d=[45,75])", fontsize=14)
    plt.xlabel("Normalized Displacement", fontsize=12)
    plt.ylabel("Tower Height (m)", fontsize=12)
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.5)
    plt.tight_layout()
    plt.show()

# ======================================================
# 5. EXECUTION
# ======================================================

# A. Sensitivity Plots
plot_sensitivity(df_res_fa, "Fore-Aft (1st Mode)")
plot_sensitivity(df_res_ss, "Side-to-Side (1st Mode)")

# B. Probabilistic Shape Plots
# This is the "Money Plot" for your paper
plot_probabilistic_shape(shapes_fa, "Fore-Aft 1st Mode")
plot_probabilistic_shape(shapes_ss, "Side-to-Side 1st Mode")
