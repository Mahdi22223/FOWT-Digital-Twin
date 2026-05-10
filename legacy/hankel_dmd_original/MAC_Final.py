import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import re

# ==========================================
# 1. HELPER FUNCTIONS
# ==========================================
def parse_complex(s):
    if isinstance(s, (int, float, complex)): return complex(s)
    s = str(s).replace('(', '').replace(')', '').replace('i', 'j')
    try: return complex(s)
    except: return 0j
    
def extract_freq(name):
    match = re.search(r"(\d+\.\d+)Hz", str(name))
    return float(match.group(1)) if match else -1.0

def calculate_complex_mac(v1, v2):
    """Standard Complex MAC"""
    v1 = np.array(v1).flatten()
    v2 = np.array(v2).flatten()
    if len(v1) != len(v2): return 0.0
    num = np.abs(np.vdot(v1, v2))**2
    den = np.vdot(v1, v1).real * np.vdot(v2, v2).real
    return num / den if den != 0 else 0.0

def calculate_real_projected_mac(v_dmd, v_of):
    """Robust MAC: Projects complex DMD mode onto Real axis (Standing Wave)."""
    v_dmd = np.array(v_dmd).flatten()
    v_of = np.array(v_of).flatten()
    
    # 1. Rotate DMD to Max Real
    idx_max = np.argmax(np.abs(v_dmd))
    angle = np.angle(v_dmd[idx_max])
    v_dmd_rot = (v_dmd * np.exp(-1j * angle)).real
    
    # 2. OpenFAST (Abs/Real)
    v_of_shape = np.abs(v_of)
    
    # 3. MAC
    num = np.dot(v_dmd_rot, v_of_shape)**2
    den = np.dot(v_dmd_rot, v_dmd_rot) * np.dot(v_of_shape, v_of_shape)
    return num / den if den != 0 else 0.0

# ==========================================
# 2. CONFIGURATION
# ==========================================

# FORE-AFT TARGETS (Hz) DMD
fa_targets = {
    "1st_Tower_Mode": {"OF": 0.477, "DMD": 0.541},
    "2nd_Tower_Mode": {"OF": 1.542, "DMD": 1.674} # Matches your validated results
}

# SIDE-TO-SIDE TARGETS (Hz)
ss_targets = {
    "1st_Tower_Mode": {"OF": 0.477, "DMD": 0.524},
    "2nd_Tower_Mode": {"OF": 2.00, "DMD": 2.042} # Trying to catch the 2.04 Hz mode
}


# FORE-AFT TARGETS (Hz) SSI COV
# fa_targets = {
#     "1st_Tower_Mode": {"OF": 0.477, "SSI": 0.350},
#     "2nd_Tower_Mode": {"OF": 1.542, "SSI": 1.995} # Matches your validated results
# }

# # SIDE-TO-SIDE TARGETS (Hz)
# ss_targets = {
#     "1st_Tower_Mode": {"OF": 0.477, "SSI": 0.638},
#     "2nd_Tower_Mode": {"OF": 2.00, "SSI": 2.054} # Trying to catch the 2.04 Hz mode
# }

tolerance = 0.9 # Wide enough to capture the physical shifts

Name_method = "DMD"
# Name_method = "SSI"
# ==========================================
# 3. ANALYSIS & PLOTTING FUNCTION
# ==========================================


def analyze_and_plot(direction, target_dict, df_of_in, df_dmd_in, Name_method):
    print(f"\n{'='*40} {direction} VALIDATION {'='*40}")
    print(f"{'Target Name':<20} | {'OpenFAST Ref':<35} | {Name_method + ' Found':<35} | {'Freq Diff':<10} | {'Complex MAC':<11} | {'Real MAC':<8}")
    print("-" * 140)

    # Lists to store selected columns for Heatmap
    sel_of_cols = []
    sel_dmd_cols = []

    for target_name, freqs in target_dict.items():
        t_of = freqs["OF"]
        t_dmd = freqs[Name_method]
        
        # --- Find Best OpenFAST Candidate ---
        best_of_col = None
        min_diff_of = 100
        for col in df_of_in.columns:
            f = extract_freq(col)
            diff = abs(f - t_of)
            if diff < min_diff_of and diff < tolerance:
                min_diff_of = diff
                best_of_col = col
        
        # --- Find Best DMD Candidate ---
        best_dmd_col = None
        min_diff_dmd = 100
        for col in df_dmd_in.columns:
            f = extract_freq(col)
            diff = abs(f - t_dmd)
            if diff < min_diff_dmd and diff < tolerance:
                min_diff_dmd = diff
                best_dmd_col = col

        # --- Evaluate Match ---
        if best_of_col and best_dmd_col:
            # Add to list for heatmap
            if best_of_col not in sel_of_cols: sel_of_cols.append(best_of_col)
            if best_dmd_col not in sel_dmd_cols: sel_dmd_cols.append(best_dmd_col)

            # Calculate metrics
            vec_of = df_of_in[best_of_col].values
            vec_dmd = df_dmd_in[best_dmd_col].values
            
            f_diff = abs(extract_freq(best_of_col) - extract_freq(best_dmd_col))
            mac_c = calculate_complex_mac(vec_dmd, vec_of)
            mac_r = calculate_real_projected_mac(vec_dmd, vec_of)
            
            # Print row
            print(f"{target_name:<20} | {best_of_col[:35]:<35} | {best_dmd_col[:35]:<35} | {f_diff:.3f} Hz   | {mac_c:.4f}      | {mac_r:.4f}")
        else:
            print(f"{target_name:<20} | {'NO MATCH FOUND':<35} | {'-':<35} | -          | -           | -")

    print("-" * 140)

    # --- Generate Heatmap ---
    if not sel_of_cols or not sel_dmd_cols:
        print("Not enough matches to generate heatmap.")
        return

    mac_mat = pd.DataFrame(index=sel_of_cols, columns=sel_dmd_cols, dtype=float)
    for r in sel_of_cols:
        for c in sel_dmd_cols:
            mac_mat.loc[r, c] = calculate_real_projected_mac(df_dmd_in[c], df_of_in[r])

    plt.figure(figsize=(8, 5))
    yticklabels = [f"OF {extract_freq(x):.3f} Hz" for x in mac_mat.index]
    xticklabels = [f"DMD {extract_freq(x):.3f} Hz" for x in mac_mat.columns]
    
    sns.heatmap(mac_mat, annot=True, fmt=".2f", cmap="Blues", 
                xticklabels=xticklabels, yticklabels=yticklabels,
                vmin=0, vmax=1)
    plt.title(f"{direction}: Mode Shape Validation")
    plt.tight_layout()
    plt.show()

# ==========================================
# 4. EXECUTION EXAMPLE
# ==========================================
# Note: You need to pass the correct DataFrames.
# df_of: Loaded from Excel 'Tower_Modes' sheet
# df_dmd_fa: Result from your Fore-Aft analysis
# df_dmd_ss: Result from your Side-to-Side analysis
# ==========================================
# 2. LOAD DATA
# ==========================================
# Assuming df_of and df_dmd are already loaded in your environment.
# If not, uncomment and adapt:
# df_of = pd.read_excel('Extracted_Mode_Shapes.xlsx', index_col=0).applymap(parse_complex)
# df_dmd = df_modes # Result from run_analysis

# A. Load OpenFAST (Excel)
# assuming 'shapes' is the index column (1, 2, 3...)
df_of = pd.read_excel('Extracted_Mode_Shapes.xlsx', index_col=0)

# Convert all columns to Complex Numbers
df_of = df_of.applymap(parse_complex)

# B. Load DMD (From your previous step)
# Assuming 'df_modes' is the variable from the previous run_analysis() return
# If loading from CSV instead: df_modes = pd.read_csv('dmd_modes.csv', index_col=0).applymap(parse_complex)


# Uncomment to run:
# analyze_and_plot("Fore-Aft", fa_targets, df_of, df_fa_modes,Name_method)  # ---> df_SSI_FA
# analyze_and_plot("Side-to-Side", ss_targets, df_of, df_ss_modes,Name_method) # ----> df_SSI_SS


# Uncomment to run:
analyze_and_plot("Fore-Aft", fa_targets, df_SSI_FA, df_fa_modes,Name_method)  # ---> df_SSI_FA
analyze_and_plot("Side-to-Side", ss_targets, df_SSI_SS, df_ss_modes,Name_method) # ----> df_SSI_SS
