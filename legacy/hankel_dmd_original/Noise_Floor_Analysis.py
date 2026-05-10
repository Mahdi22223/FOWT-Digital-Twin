import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.signal import butter, filtfilt
from sklearn.utils.extmath import randomized_svd
import glob

# ======================================================
# 1. SETUP (Must match your DMD setup)
# ======================================================
dt = 0.0125
fs = 1 / dt
low_cutoff = 0.25; high_cutoff = 5.0; filter_order = 4   
hankel_d = 60  # Your chosen delay

# Columns
fa_acc_cols = [f'TwHt{i}ALxt_[m/s^2]' for i in range(1, 10)]
fa_mom_cols = [f'TwHt{i}MLyt_[kN-m]' for i in range(1, 10)]

file_list = sorted(glob.glob("Case_*.csv"))
if not file_list: raise FileNotFoundError("No Case_*.csv files found.")

def apply_zero_phase_filter(data, low, high, fs):
    nyq = 0.5 * fs
    b, a = butter(filter_order, [low/nyq, high/nyq], btype='band')
    return filtfilt(b, a, data, axis=0)

# ======================================================
# 2. BUILD HANKEL MATRIX (Manual Construction)
# ======================================================
print("Building Hankel Matrix for SVD check...")

snapshots_list = []
for f in file_list:
    df = pd.read_csv(f, sep=None, engine='python')
    acc = apply_zero_phase_filter(df[fa_acc_cols].values, low_cutoff, high_cutoff, fs)
    mom = apply_zero_phase_filter(df[fa_mom_cols].values, low_cutoff, high_cutoff, fs)
    snapshots_list.append(np.hstack((acc, mom)))

X = np.vstack(snapshots_list).T
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X.T).T

# Downsample time to save RAM for SVD (statistically valid)
X_sub = X_scaled[:, ::5] 

n_features, n_time = X_sub.shape
H_rows = n_features * hankel_d
H_cols = n_time - hankel_d + 1

# Construct matrix H
H = np.zeros((H_rows, H_cols))
for i in range(hankel_d):
    H[i*n_features : (i+1)*n_features, :] = X_sub[:, i : i+H_cols]

print(f"Hankel Matrix Shape: {H.shape}")

# ======================================================
# 3. COMPUTE SINGULAR VALUES (The Energy)
# ======================================================
print("Computing SVD... (This tells us the true rank)")
U, S, Vt = randomized_svd(H, n_components=200, random_state=100)

# ======================================================
# 4. PLOT THE "ELBOW"
# ======================================================
plt.figure(figsize=(10, 6))

rank_axis = np.arange(1, len(S) + 1)

# Plot singular values
plt.semilogy(rank_axis, S, 'ko-', markersize=3, linewidth=1, label='Singular Values')

# Plot your Test Ranks (Vertical Lines)
plt.axvline(24, color='r', linestyle='--', label='Rank 24')
plt.axvline(35, color='g', linestyle='--', label='Rank 35')

plt.xlabel("Rank (k)")
plt.ylabel("Energy (Singular Value)")
plt.title(f"SVD Scree Plot (d={hankel_d})\nNoise Floor Analysis")
plt.legend()
plt.grid(True, which="both", alpha=0.3)
plt.tight_layout()
plt.show()

# ======================================================
# 5. AUTOMATIC ESTIMATION (Gavish-Donoho)
# ======================================================
# A simple heuristic to guess the cutoff
# We assume the tail is noise.
beta = H.shape[0] / H.shape[1]
cutoff = np.median(S) * 2.858 # Approximate threshold for white noise
optimal_rank = np.sum(S > cutoff)

print(f"\n--- ESTIMATED OPTIMAL RANK ---")
print(f"Optimal Hard Threshold Rank: ~{optimal_rank}")
print(f"Look at the plot. You want to be BEFORE the flat tail.")



def binary_search(arr, target):
    """
    Perform a binary search on a sorted array to find the target value.

    Args:
        arr (list): A sorted list of elements.
        target (any): The value to search for.

    Returns:
        int: The index of the target element if found, otherwise -1.
    """
    left, right = 0, len(arr) - 1

    while left <= right:
        mid = (left + right) // 2
        mid_val = arr[mid]

        if mid_val == target:
            return mid
        elif mid_val < target:
            left = mid + 1
        else:
            right = mid - 1

    return -1

# Example usage:
if __name__ == "__main__":
    my_list = [1, 3, 5, 7, 9, 11, 13, 15]
    target_value = 7
    
    result = binary_search(my_list, target_value)
    
    if result != -1:
        print(f"Element {target_value} found at index {result}.")
    else:
        print(f"Element {target_value} not found in the list.")
        
        

# ======================================================
# 4. PLOT THE "ELBOW" & SPECTRAL ZONES
# ======================================================
plt.figure(figsize=(10, 6))

rank_axis = np.arange(1, len(S) + 1)

# --- ADDED: Shadow Highlight Zones ---
# Zone 1: OMA/SID (Rank 24-34) - Moderate rank for extracting physics
plt.axvspan(24, 34, color='b', alpha=0.2, label=r'Zone I: OSystem Identification[$r \in (24, 34)$]')

# Zone 2: Virtual Sensing (Rank 50-90) - High rank for accuracy
plt.axvspan(50, 90, color='g', alpha=0.2, label=r'Zone II: Virtual Sensing [$r \in (50, 90)$]')
# -------------------------------------

# Plot singular values
plt.semilogy(rank_axis, S, 'ko-', markersize=3, linewidth=1, label='Singular Values')



plt.xlabel("Rank (k)")
plt.ylabel("Energy (Singular Value)")
plt.title(f"SVD Scree Plot (d={hankel_d})\nNoise Floor Analysis")
plt.legend()
plt.grid(True, which="both", alpha=0.3)
plt.tight_layout()
plt.show()