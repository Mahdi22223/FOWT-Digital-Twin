import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from sklearn.neighbors import NearestNeighbors
import glob

# ==========================================
# 1. CONFIGURATION
# ==========================================
dt = 0.0125
fs = 1 / dt
low_cutoff = 0.25; high_cutoff = 5.0; filter_order = 4   

# Parameters for Lyapunov Estimation
embedding_dim = 10  # Dimension of phase space (approx 2*n_modes + 1)
time_delay = 10     # Lag for embedding (not same as Hankel 'd')
max_t_steps = 200   # How far into future to track divergence (200 * 0.0125 = 2.5s)

# Columns (We use Top Tower Accel as the proxy for chaos)
fa_acc_cols = [f'TwHt{i}ALxt_[m/s^2]' for i in range(1, 10)]

file_list = sorted(glob.glob("Case_*.csv"))
if not file_list: raise FileNotFoundError("No Case_*.csv files found.")

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================
def apply_zero_phase_filter(data, low, high, fs):
    nyq = 0.5 * fs
    b, a = butter(filter_order, [low/nyq, high/nyq], btype='band')
    return filtfilt(b, a, data, axis=0)

def embed_series(series, m, d):
    """Create Time-Delay Embedding (Phase Space Reconstruction)"""
    N = len(series)
    # Result shape: (N - (m-1)*d, m)
    Y = np.zeros((N - (m-1)*d, m))
    for i in range(m):
        Y[:, i] = series[i*d : i*d + Y.shape[0]]
    return Y

# ==========================================
# 3. ROSENSTEIN ALGORITHM
# ==========================================
def calculate_lyapunov(data_series):
    print("Reconstructing Phase Space...")
    
    # 1. Embed the time series
    # This creates the 'Attractor' geometry
    X_emb = embed_series(data_series, m=embedding_dim, d=time_delay)
    n_points = X_emb.shape[0]
    
    # 2. Find Nearest Neighbors
    # For every point i, find point j that is closest (but not too close in time)
    print("Finding Neighbors...")
    nbrs = NearestNeighbors(n_neighbors=2, algorithm='kd_tree').fit(X_emb)
    distances, indices = nbrs.kneighbors(X_emb)
    
    # indices[:, 0] is the point itself. indices[:, 1] is the nearest neighbor.
    nearest_idx = indices[:, 1]
    
    # Filter out temporal neighbors (points that are close just because they are t+1)
    # We want "Recurrence" (history matches current), not "Continuity"
    # Simple check: ignore if index difference is small
    valid_pairs = []
    min_separation = 100 # steps
    
    for i, j in enumerate(nearest_idx):
        if abs(i - j) > min_separation:
            valid_pairs.append((i, j))
            
    print(f"Found {len(valid_pairs)} valid pairs for tracking.")
    
    # 3. Track Divergence
    # Measure distance between pair (i, j) as they evolve k steps into future
    divergence = np.zeros(max_t_steps)
    counts = np.zeros(max_t_steps)
    
    for k in range(max_t_steps):
        dist_sum = 0
        n_valid = 0
        
        for i, j in valid_pairs:
            if (i + k < n_points) and (j + k < n_points):
                # Calculate Euclidean distance at step k
                dist = np.linalg.norm(X_emb[i+k] - X_emb[j+k])
                if dist > 1e-10: # Avoid log(0)
                    dist_sum += np.log(dist)
                    n_valid += 1
        
        if n_valid > 0:
            divergence[k] = dist_sum / n_valid
            counts[k] = n_valid

    return divergence

# ==========================================
# 4. EXECUTION & PLOTTING
# ==========================================
if __name__ == "__main__":
    # Load just one case (e.g. Case 12 - High Wind) for chaos analysis
    # or concatenate all. Let's use the most turbulent one (last file)
    print(f"Analyzing chaos in: {file_list[-1]}")
    
    df = pd.read_csv(file_list[-1], sep=None, engine='python')
    acc = df[fa_acc_cols].values
    
    # Filter
    acc_clean = apply_zero_phase_filter(acc, low_cutoff, high_cutoff, fs)
    
    # Use Top Tower Acceleration (Sensor 8) as proxy for system chaos
    signal = acc_clean[:, 8]
    
    # Calc Lyapunov
    div_curve = calculate_lyapunov(signal)
    
    # Create Time Axis
    t_axis = np.arange(max_t_steps) * dt
    
    # --- FIT LINEAR SLOPE ---
    # The slope of the linear region is Lambda
    # Usually the first few seconds (e.g. 0 to 1s)
    fit_region = int(1.0 / dt) 
    coeffs = np.polyfit(t_axis[:fit_region], div_curve[:fit_region], 1)
    lambda_max = coeffs[0]
    
    # Calculate Lyapunov Time
    # TL = 1 / lambda
    lyapunov_time = 1.0 / lambda_max if lambda_max > 0 else 999.0
    
    print(f"\nRESULTS:")
    print(f"Lyapunov Exponent (Lambda): {lambda_max:.4f} (1/s)")
    print(f"Lyapunov Time (Prediction Horizon): {lyapunov_time:.4f} seconds")
    
    # --- PLOT ---
    plt.figure(figsize=(10, 6))
    plt.plot(t_axis, div_curve, 'k-', lw=2, label='Average Divergence')
    
    # Plot Fit
    fit_line = coeffs[0] * t_axis + coeffs[1]
    plt.plot(t_axis, fit_line, 'r--', label=f'Fit (Slope = {lambda_max:.2f})')
    
    plt.axvline(lyapunov_time, color='b', linestyle=':', label=f'Lyapunov Time ({lyapunov_time:.2f}s)')
    
    # --- CORRECTED TITLE ---
    plt.title(f"Lyapunov Exponent Estimation (Top Tower Acceleration)", fontsize=14, fontweight='bold')
    # ----------------------
    
    plt.xlabel("Time (s)", fontsize=12)
    plt.ylabel("ln(Divergence)", fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()