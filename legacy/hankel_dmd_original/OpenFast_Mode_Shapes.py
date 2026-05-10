import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

# ==========================================
# 1. CONFIGURATION
# ==========================================
low_freq_cutoff = 0.25
high_freq_cutoff = 5.0

# Exact Tower Heights (Consistent with your DMD)
tower_h = np.linspace(10, 87.6, 20)[[0,2,4,6,9,12,14,16,19]] 

def parse_complex(s):
    if isinstance(s, (int, float, complex)): return complex(s)
    s = str(s).replace('(', '').replace(')', '').replace('i', 'j')
    try: return complex(s)
    except: return 0j

def extract_freq_val(name):
    """Extracts float frequency from string."""
    match = re.search(r"(\d+\.\d+)Hz", str(name))
    return float(match.group(1)) if match else -1.0

def extract_label(name):
    """Clean label for legend."""
    match = re.search(r"(\d+\.\d+)Hz", str(name))
    return f"{match.group(1)} Hz" if match else name[:15]

# ==========================================
# 2. LOAD DATA
# ==========================================
filename = 'Extracted_Mode_Shapes.xlsx'
try:
    df_of = pd.read_excel(filename, sheet_name='Mode_Plot', index_col=0)
except ValueError:
    df_of = pd.read_excel(filename, index_col=0)

try:
    df_of = df_of.map(parse_complex)
except AttributeError:
    df_of = df_of.applymap(parse_complex)

# ==========================================
# 3. PLOTTING (With Filter)
# ==========================================
plt.figure(figsize=(10, 8))
plt.rcParams.update({'font.size': 14})

# Pre-calculate valid columns to set up colors correctly
valid_cols = []
for col in df_of.columns:
    f = extract_freq_val(col)
    if low_freq_cutoff <= f <= high_freq_cutoff:
        valid_cols.append(col)

colors = plt.cm.jet(np.linspace(0, 1, len(valid_cols)))

print(f"Found {len(valid_cols)} modes in range {low_freq_cutoff}-{high_freq_cutoff} Hz.")

for i, col_name in enumerate(valid_cols):
    mode_vec = df_of[col_name].values
    
    # Safety Check
    if len(mode_vec) != len(tower_h):
        print(f"Skipping {col_name}: Dimension mismatch.")
        continue

    # Normalize (Real Part)
    max_val = np.max(np.abs(mode_vec))
    if max_val == 0: max_val = 1.0
    norm_shape = mode_vec.real / max_val
    
    # Styling
    freq = extract_freq_val(col_name)
    label = extract_label(col_name)
    
    ls = '--' if freq > 1.0 else '-'
    marker = 's' if freq > 1.0 else 'o'

    plt.plot(norm_shape, tower_h, linestyle=ls, marker=marker, 
             linewidth=2, label=label, color=colors[i])

plt.title(f"OpenFAST Linearization Mode Shapes (Hz)", fontweight='bold')
plt.xlabel("Normalized Mode Shape")
plt.ylabel("Tower Height (m)")
plt.axvline(0, color='k', linestyle=':', alpha=0.5)

# Legend outside
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()