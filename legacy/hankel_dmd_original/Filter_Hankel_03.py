import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pydmd import HankelDMD
from pydmd.plotter import plot_eigs
from sklearn.preprocessing import StandardScaler
from sklearn.utils.extmath import randomized_svd
from scipy.signal import butter, filtfilt
import glob

# ======================================================
# 1. CONFIGURATION
# ======================================================
dt = 0.0125
fs = 1 / dt

# FILTER SETTINGS
low_cutoff = 0.25 
high_cutoff = 5.0 
filter_order = 4   

# HANKEL SETTINGS
hankel_d = 60 # - high value cause over fitting, low value get fail to see a full wave cycle
target_rank = 24  # We check this with the SVD plot later

# Define Columns
fa_acc_cols = [f'TwHt{i}ALxt_[m/s^2]' for i in range(1, 10)]
fa_mom_cols = [f'TwHt{i}MLyt_[kN-m]' for i in range(1, 10)]

ss_acc_cols = [f'TwHt{i}ALyt_[m/s^2]' for i in range(1, 10)]
ss_mom_cols = [f'TwHt{i}MLxt_[kN-m]' for i in range(1, 10)]

file_list = sorted(glob.glob("Case_*.csv"))
if not file_list:
    raise FileNotFoundError("No Case_*.csv files found.")

# ======================================================
# 2. HELPER FUNCTIONS
# ======================================================

def apply_zero_phase_filter(data, low, high, fs):
    nyq = 0.5 * fs
    b, a = butter(filter_order, [low/nyq, high/nyq], btype='band')
    return filtfilt(b, a, data, axis=0)

tower_h = np.linspace(10, 87.6, 20)[[0,2,4,6,9,12,14,16,19]] 

# ======================================================
# 3. ANALYSIS FUNCTION
# ======================================================

def run_analysis(direction_name, acc_cols, mom_cols):
    print(f"\n================ {direction_name} COMPUTATION START ================")
    
    snapshots_list = []
    for f in file_list:
        df = pd.read_csv(f, sep=None, engine='python')
        acc = df[acc_cols].values
        mom = df[mom_cols].values
        
        acc_clean = apply_zero_phase_filter(acc, low_cutoff, high_cutoff, fs)
        mom_clean = apply_zero_phase_filter(mom, low_cutoff, high_cutoff, fs)
        
        # Stack: [Acc (0-8) | Mom (9-17)]
        snapshots_list.append(np.hstack((acc_clean, mom_clean)))

    X = np.vstack(snapshots_list).T
    
    scaler = StandardScaler()
    scaler.fit(X.T)
    X_scaled = scaler.transform(X.T).T
    
    print(f"Fitting Hankel DMD (d={hankel_d}, Rank={target_rank})...")
    hdmd = HankelDMD(svd_rank=target_rank, tlsq_rank=0, exact=True, opt=False, d=hankel_d)
    hdmd.fit(X_scaled)
    
    print(f"{direction_name} Fit Complete.")
    
    # RETURN ALL DATA FOR POST-PROCESSING
    return {
        "model": hdmd,
        "scaler": scaler,
        "X_raw": X,
        "X_scaled": X_scaled,
        "name": direction_name
    }

# ======================================================
# 4. PLOTTING FUNCTIONS
# ======================================================
def plot_eigs_custom(analysis_result, dt, font_size=14):
    """
    Custom plot for Eigenvalues. 
    Highlights 1st and 2nd modes in RED and labels them with text descriptions.
    Uses smaller font size (6) for annotations as requested.
    """
    hdmd = analysis_result['model']
    name = analysis_result['name']
    
    eigs = hdmd.eigs
    log_eigs = np.log(eigs)
    freqs = np.abs(log_eigs.imag) / (dt * 2 * np.pi)
    
    # Define ranges to highlight (Physics)
    # 1st Mode: 0.4 - 0.6 Hz
    # 2nd Mode: 1.5 - 2.5 Hz
    highlight_mask = ((freqs > 0.4) & (freqs < 0.6)) | ((freqs > 1.5) & (freqs < 2.5))
    
    # Separate data
    eigs_highlight = eigs[highlight_mask]
    freqs_highlight = freqs[highlight_mask]
    eigs_normal = eigs[~highlight_mask]
    
    # --- PLOTTING ---
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # 1. Unit Circle
    theta = np.linspace(0, 2*np.pi, 200)
    ax.plot(np.cos(theta), np.sin(theta), 'k--', linewidth=1.5, alpha=0.6, label='Stability Limit')
    
    # 2. Plot Normal (Blue)
    ax.scatter(eigs_normal.real, eigs_normal.imag, s=60, c='dodgerblue', alpha=0.6, label='Other Modes', edgecolors='k', linewidth=0.5)
    
    # 3. Plot Highlight (Red)
    ax.scatter(eigs_highlight.real, eigs_highlight.imag, s=150, c='red', marker='*', label='Tower Candidate Modes(1st/2nd)', zorder=10, edgecolors='k')
    
    # 4. Text Annotations (Descriptive)
    
    # -- 1st Mode (Low Freq ~0.5 Hz) --
    mask_1 = (freqs_highlight > 0.4) & (freqs_highlight < 0.6) & (eigs_highlight.imag > 0)
    if np.any(mask_1):
        # Pick the representative point
        subset_eigs = eigs_highlight[mask_1]
        best_idx = np.argmax(subset_eigs.real) 
        eig_1 = subset_eigs[best_idx]
        
        # Place text inside
        ax.annotate("1st Mode\nCandidate", 
                    xy=(eig_1.real, eig_1.imag), 
                    xytext=(eig_1.real * 0.75, eig_1.imag * 0.75),
                    fontsize=6, weight='bold', color='darkred',  # Reduced Font Size
                    ha='center', va='center',
                    arrowprops=dict(arrowstyle="-|>", color='gray', lw=1.0))

    # -- 2nd Mode (High Freq ~1.7 - 2.0 Hz) --
    mask_2 = (freqs_highlight > 1.5) & (freqs_highlight < 2.5) & (eigs_highlight.imag > 0)
    if np.any(mask_2):
        # Pick representative
        subset_eigs = eigs_highlight[mask_2]
        best_idx = np.argmax(subset_eigs.real)
        eig_2 = subset_eigs[best_idx]
        
        ax.annotate("2nd Mode\nCandidate", 
                    xy=(eig_2.real, eig_2.imag), 
                    xytext=(eig_2.real * 0.75, eig_2.imag * 0.75),
                    fontsize=6, weight='bold', color='darkred', # Reduced Font Size
                    ha='center', va='center',
                    arrowprops=dict(arrowstyle="-|>", color='gray', lw=1.0))

    # 5. Styling
    ax.set_title(f"{name}: Stability Map (Eigenvalues)", fontsize=font_size+2, weight='bold')
    ax.set_xlabel("Real Part", fontsize=font_size)
    ax.set_ylabel("Imaginary Part", fontsize=font_size)
    ax.tick_params(axis='both', which='major', direction='out', labelsize=12)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(fontsize=12, loc='lower left')
    ax.set_aspect('equal')
    
    # Zoom slightly to focus on the circle
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    
    plt.tight_layout()
    plt.show()
    
def plot_svd_rank_stability(analysis_result, d=60, max_rank_check=60):
    """
    Plots the Singular Values (Energy) vs Rank.
    Helps determine if Rank 24 is sufficient.
    """
    name = analysis_result['name']
    X = analysis_result['X_scaled']
    
    print(f"\n--- Computing SVD Rank Analysis ({name}) ---")
    
    # 1. Build Implicit Hankel Matrix Specs
    n_features, n_time = X.shape
    # To save memory, we don't build the full Hankel matrix.
    # We downsample in time for the SVD check (statistically valid for finding dominant modes)
    # Taking every 5th snapshot preserves the singular value distribution shape.
    X_sub = X[:, ::5] 
    
    # Construct Hankel Matrix manually for SVD
    # Size: (features*d) x (time_sub - d)
    H_rows = n_features * d
    H_cols = X_sub.shape[1] - d + 1
    
    if H_cols <= 0:
        print("Data too short for Hankel SVD check.")
        return

    # Build H (Stacked)
    H = np.zeros((H_rows, H_cols))
    for i in range(d):
        H[i*n_features : (i+1)*n_features, :] = X_sub[:, i : i+H_cols]
    
    print(f"Hankel Matrix Shape for SVD: {H.shape}")
    
    # 2. Compute Total Energy (Variance)
    # Frobenius norm squared = sum of all singular values squared
    total_energy = np.sum(H**2)
    
    # 3. Compute Top Singular Values (Randomized SVD is fast)
    U, S, Vt = randomized_svd(H, n_components=max_rank_check, random_state=42)
    
    # 4. Calculate Cumulative Variance
    explained_variance_ratio = (S**2) / total_energy
    cumulative_variance = np.cumsum(explained_variance_ratio) * 100
    
    # 5. Plot
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Plot SVs
    color = 'tab:blue'
    ax1.set_xlabel('Rank (Mode Index)')
    ax1.set_ylabel('Singular Value (Log Scale)', color=color)
    ax1.semilogy(np.arange(1, len(S)+1), S, 'o-', color=color, markersize=3, alpha=0.6)
    ax1.tick_params(axis='y', labelcolor=color)
    
    # Draw cutoff line
    ax1.axvline(target_rank, color='k', linestyle='--', label=f'Current Rank ({target_rank})')
    ax1.legend(loc='lower left')
    
    # Plot Cumulative Energy
    ax2 = ax1.twinx()
    color = 'tab:orange'
    ax2.set_ylabel('Cumulative Energy (%)', color=color)
    ax2.plot(np.arange(1, len(S)+1), cumulative_variance, 's-', color=color, markersize=4)
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim(0, 105)
    
    # Annotation
    if target_rank <= len(cumulative_variance):
        energy_at_rank = cumulative_variance[target_rank-1]
        ax2.annotate(f'{energy_at_rank:.1f}% Variance', 
                     xy=(target_rank, energy_at_rank), 
                     xytext=(target_rank+5, energy_at_rank-10),
                     arrowprops=dict(facecolor='black', shrink=0.05))
    
    plt.title(f"{name}: Hankel SVD Spectrum (d={d})")
    plt.grid(True, which='both', alpha=0.3)
    plt.show()


def plot_physical_modes(analysis_result, low_f=0.3, high_f=5.0):
    hdmd = analysis_result['model']
    direction_name = analysis_result['name']
    
    print(f"\n--- Plotting Physical Mode Shapes ({direction_name}) ---")
    
    # Calculate Physics
    eigs = hdmd.eigs
    log_eigs = np.log(eigs)
    omega = log_eigs / dt
    freqs_hz = np.abs(omega.imag) / (2 * np.pi)
    damping = -omega.real / np.abs(omega)
    
    # Filter
    valid_mask = (freqs_hz > low_f) & (freqs_hz < high_f) 
    valid_idx = np.where(valid_mask)[0]
    sorted_idx = valid_idx[np.argsort(freqs_hz[valid_idx])]
    
    unique_freqs = []
    extracted_modes = {}
    mode_stats = []
    
    # --- PLOT SETUP (Styling) ---
    plt.figure(figsize=(10, 7)) # Slightly wider to fit legend
    plt.rcParams.update({'font.size': 14})
    
    # Generate distinct colors for up to 10 modes
    colors = plt.cm.jet(np.linspace(0, 1, min(10, len(sorted_idx))))
    
    plot_count = 0 
    
    for idx in sorted_idx:
        f = freqs_hz[idx]
        
        if not any(abs(f - x) < 0.1 for x in unique_freqs):
            unique_freqs.append(f)
            # Save Stats (for Sensitivity Analysis)
            mode_stats.append({
                "Freq_Hz": f,
                "Damping": damping[idx],
                "Index": idx
            })
            
            # Extract Hankel Mode (Physical part only)
            phi_full = hdmd.modes[:, idx]
            phi_phys = phi_full[0:18]
            
            # 1. Save Moment (for MAC calculation)
            mom_complex = phi_phys[9:18]
            extracted_modes[f"Freq_{f:.3f}Hz_Mode{idx}"] = mom_complex
            
            # 2. Calculate Displacement (for Plotting)
            acc_complex = phi_phys[0:9]
            w = 2 * np.pi * f
            if w == 0: w = 1e-6
            disp_complex = -acc_complex / (w**2)
            
            # Normalize for plotting
            max_val = disp_complex[np.argmax(np.abs(disp_complex))]
            if max_val != 0: disp_norm = disp_complex / max_val
            else: disp_norm = disp_complex

            # --- PLOT COMMAND (Styled) ---
            if plot_count < 10: 
                # Dashed for 2nd modes (> 1.0 Hz), Solid for 1st modes
                ls = '--' if f > 1.0 else '-'
                mk = 's' if f > 1.0 else 'o'
                
                plt.plot(disp_norm.real, tower_h, linestyle=ls, marker=mk, 
                         linewidth=2, label=f'{f:.2f} Hz', color=colors[plot_count])
                
                plot_count += 1

    # Finalize Plot Styling
    plt.title(f"{direction_name}: DMD Mode Shapes", fontweight='bold')
    plt.xlabel("Normalized Displacement")
    plt.ylabel("Tower Height (m)")
    
    # Legend outside the plot to prevent covering data
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left') 
    
    plt.grid(True, alpha=0.5)
    plt.tight_layout()
    plt.show()
    
    # Return DataFrames for post-processing
    return pd.DataFrame(extracted_modes), pd.DataFrame(mode_stats)

def plot_dynamics_manual(analysis_result, mode_indices=None):
    hdmd = analysis_result['model']
    direction_name = analysis_result['name']
    
    print(f"\n--- Plotting Dynamics ({direction_name}) ---")
    dynamics = hdmd.dynamics
    time = np.arange(dynamics.shape[1]) * dt
    
    if mode_indices is None:
        energy = np.linalg.norm(dynamics, axis=1)
        mode_indices = np.argsort(energy)[::-1][:3]

    plt.figure(figsize=(12, 5))
    plt.rcParams.update({'font.size': 14})
    
    for idx in mode_indices:
        log_eig = np.log(hdmd.eigs[idx])
        freq = np.abs(log_eig.imag) / (dt * 2 * np.pi)
        plt.plot(time, dynamics[idx, :].real, label=f'Mode {idx} ({freq:.2f} Hz)', linewidth=1.5)
        
    plt.title(f"{direction_name}: Mode Dynamics", fontweight='bold')
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend(loc='upper right', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 30) 
    plt.show()

# ======================================================
# 5. EXECUTION
# ======================================================
if __name__ == "__main__":
    
    # 1. Compute
    results_fa = run_analysis("Fore-Aft", fa_acc_cols, fa_mom_cols)
    results_ss = run_analysis("Side-to-Side", ss_acc_cols, ss_mom_cols)
    results_fa.keys()
    # 2. Visualize - SVD Rank Stability (NEW)
    plot_svd_rank_stability(results_fa, d=hankel_d)
    plot_svd_rank_stability(results_ss, d=hankel_d)
    
    # 3. Visualize - Eigenvalues
    print("\nDisplaying Custom Eigenvalues...")
    plot_eigs_custom(results_fa, dt, font_size=16)
    plot_eigs_custom(results_ss, dt, font_size=16)

    # 4. Visualize - Shapes
    df_fa_modes, stats_FA = plot_physical_modes(results_fa, low_f=0.3, high_f=3.0)
    df_ss_modes, stats_SS = plot_physical_modes(results_ss, low_f=0.3, high_f=3.0)
    
    print("--- Fore-Aft Mode Statistics ---")
    print(stats_FA)
    
    print("--- Side-to-Side Mode Statistics ---")
    print(stats_SS)

# range(1, 24)
    plot_dynamics_manual(results_fa, mode_indices=(12,22)) # 12, 22
    plot_dynamics_manual(results_ss, mode_indices=(10, 18)) # 18, 10
    
    
 
#     # ---- Check Convergence of Damping and Natural Frequency for Time Delay
    

# # ==========================================
# # SENSITIVITY ANALYSIS (Convergence Check)
# # ==========================================
# results_sensitivity = []
# d_range = [5,10,20,25,30,35,40,45,50,55,58,62,65,70,75,80,90]  # Delay Standard range to show stability

# for test_d in d_range:
#     print(f"\n--- Sensitivity Test: d={test_d} ---")
    
#     # 1. Update global variable
#     hankel_d = test_d 
    
#     # 2. Run Analysis
#     res = run_analysis("Fore-Aft", fa_acc_cols, fa_mom_cols)
  
#     # 3. Get Stats (Unpack the tuple: _, df_stats)
#     # Target the 1st Mode (~0.54 Hz)
#     _, df_stats = plot_physical_modes(res, low_f=0.4, high_f=0.6)
    
#     # 4. Extract data
#     if not df_stats.empty:
#         # Find the mode closest to 0.54 Hz
#         best_idx = (df_stats['Freq_Hz'] - 0.54).abs().idxmin()
        
#         results_sensitivity.append({
#             'd': test_d, 
#             'Freq': df_stats.loc[best_idx, 'Freq_Hz'], 
#             'Damping': df_stats.loc[best_idx, 'Damping']
#         })

# df_res = pd.DataFrame(results_sensitivity)
# print("\n--- Sensitivity Results ---")
# print(df_res)

# # ==========================================
# # PROFESSIONAL CONVERGENCE PLOT
# # ==========================================
# if not df_res.empty:
#     fig, ax1 = plt.subplots(figsize=(10, 6))

#     # --- LEFT AXIS (Frequency) ---
#     color = 'tab:blue'
#     ax1.set_xlabel('Hankel Delay Parameter ($d$)', fontsize=12, fontweight='bold')
#     ax1.set_ylabel('Natural Frequency (Hz)', color=color, fontsize=12, fontweight='bold')
#     df_plot = df_res.drop(df_res.index[8])
#     ax1.plot(df_plot['d'], df_plot['Freq'], color=color, marker='o', linewidth=2, label='Frequency')
#     ax1.tick_params(axis='y', labelcolor=color)
#     ax1.grid(True, alpha=0.3)

#     # >>> MANUAL CONTROL 1: Left Y-Axis (Frequency) <<<
#     # Adjust these values to zoom in on your frequency (e.g., 0.4 to 0.7 Hz)
#     ax1.set_ylim(0, 1) 

#     # >>> MANUAL CONTROL 2: X-Axis (Delay) <<<
#     ax1.set_xlim(0, 100) 

#     # --- RIGHT AXIS (Damping) ---
#     ax2 = ax1.twinx()
#     color = 'tab:red'
#     ax2.set_ylabel('Damping Ratio (-)', color=color, fontsize=12, fontweight='bold')
#     ax2.plot(df_res['d'], df_res['Damping'], color=color, marker='s', linestyle='--', linewidth=2, label='Damping')
#     ax2.tick_params(axis='y', labelcolor=color)

#     # >>> MANUAL CONTROL 3: Right Y-Axis (Damping) <<<
#     # Adjust this to fit your damping range (e.g., 0 to 10% -> 0.0 to 0.1)
#     ax2.set_ylim(0, 0.8) 

#     # Highlight Convergence Zone (d=60)
#     plt.axvline(x=55, color='gray', linestyle=':', linewidth=2, label='Selected $d=55$')
    
#     # Add simple shading to highlight stability
#     if len(df_res) > 2:
#         plt.axvspan(45, 75, color='gray', alpha=0.1, label='Convergence Region')

#     plt.title('Parameter Convergence: Hankel Delay Sensitivity', fontsize=14)
    
#     # Fix Legend
#     fig.legend(loc="upper right", bbox_to_anchor=(0.92, 0.92), frameon=True)
#     plt.tight_layout()
#     plt.show()
# else:
#     print("No modes found for plotting.")
    
#     plot_svd_rank_stability(res, d=hankel_d)
#     plot_eigs_custom(res, dt, font_size=16)
#     plot_physical_modes(res, low_f=0.3, high_f=3.0)
#     plot_dynamics_manual(res, mode_indices=(25, 14,13))
