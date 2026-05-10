# -*- coding: utf-8 -*-
"""
OpenFAST Linearization Analysis Tool
Extracts Tower and Blade mode shapes and saves them to Excel for DMD comparison.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# --- Import FASTLinearizationFile (Robust check) ---
try:
    from openfast_toolbox.io import FASTLinearizationFile
except ImportError:
    try:
        from pyFAST.input_output import FASTLinearizationFile
    except ImportError:
        print("Error: Could not import openfast_toolbox or pyFAST. Please install one.")
        class FASTLinearizationFile:
            def __init__(self, path): raise ImportError("pyFAST not installed")

# ==============================================================================
#  MAIN ANALYSIS FUNCTION
# ==============================================================================

def analyze_modes(lin_file_path, freq_min=0.0, freq_max=5.0, plot_tower=True, plot_blade=True):
    
    print(f"\n{'='*80}")
    print(f"ANALYZING: {lin_file_path}")
    print(f"SCAN RANGE: {freq_min} Hz to {freq_max} Hz")
    print(f"{'='*80}")

    # --- 1. LOAD FILE ---
    try:
        f = FASTLinearizationFile(lin_file_path)
        A = f['A']
        C = f['C']
        
        # Robustly extract descriptions
        if isinstance(f['x_info'], dict):
            x_desc = f['x_info'].get('Description', list(f['x_info'].values()))
        else:
            x_desc = f['x_info']
            
        if isinstance(f['y_info'], dict):
            y_desc = f['y_info'].get('Description', list(f['y_info'].values()))
        else:
            y_desc = f['y_info']
            
        if isinstance(x_desc, dict): x_desc = list(x_desc.values())
        if isinstance(y_desc, dict): y_desc = list(y_desc.values())

    except Exception as e:
        print(f"Error loading file: {e}")
        return

    # --- 2. IDENTIFY STATE INDICES ---
    tower_indices = [i for i, n in enumerate(x_desc) if "ed" in str(n).lower() and ("tower" in str(n).lower() or "tw" in str(n).lower())]
    blade_indices = [i for i, n in enumerate(x_desc) if "ed" in str(n).lower() and ("blade" in str(n).lower() or "bd" in str(n).lower())]

    # --- 3. MAP SENSORS ---
    print("Mapping sensors...", end="")
    
    # A. TOWER SENSORS
    mom_indices_x, mom_indices_y, heights = [], [], []
    for i in range(1, 20):
        pat_x, pat_y = f"TwHt{i}MLxt", f"TwHt{i}MLyt"
        idx_x, idx_y = -1, -1
        for idx, name in enumerate(y_desc):
            if pat_x in name: idx_x = idx
            if pat_y in name: idx_y = idx
        if idx_x != -1 and idx_y != -1:
            mom_indices_x.append(idx_x)
            mom_indices_y.append(idx_y)
            heights.append(i)

    # B. BLADE SENSORS
    bld_flap_indices, bld_edge_indices, bld_spans = [], [], []
    for i in range(1, 50):
        pat_flap, pat_edge = f"Spn{i}MLxb1", f"Spn{i}MLyb1"
        idx_f, idx_e = -1, -1
        for idx, name in enumerate(y_desc):
            if pat_flap in name: idx_f = idx
            if pat_edge in name: idx_e = idx
        if idx_f != -1:
            bld_flap_indices.append(idx_f)
            if idx_e != -1: bld_edge_indices.append(idx_e) 
            bld_spans.append(i)

    print(f" Done. Found {len(heights)} Tower nodes and {len(bld_spans)} Blade nodes.")

    # --- 4. SOLVE EIGENVALUES ---
    print("Solving Eigenvalues...")
    evals, evecs = np.linalg.eig(A)

    all_mode_data = {}

    # --- 5. PROCESS MODES ---
    print(f"\n{'Freq (Hz)':<10} | {'Damp %':<8} | {'TwPart %':<8} | {'BdPart %':<8} | {'Dominant Name'}")
    print("-" * 110)

    for i, val in enumerate(evals):
        if val.imag < 0: continue
        freq = np.abs(val) / (2 * np.pi)

        if freq_min <= freq <= freq_max:
            damp = -val.real / np.abs(val) * 100
            v = evecs[:, i]
            
            total_energy = np.linalg.norm(v)
            tow_energy   = np.linalg.norm(v[tower_indices]) if tower_indices else 0
            bld_energy   = np.linalg.norm(v[blade_indices]) if blade_indices else 0
            
            tw_ratio = (tow_energy / total_energy) * 100 if total_energy > 0 else 0
            bd_ratio = (bld_energy / total_energy) * 100 if total_energy > 0 else 0
            
            max_idx = np.argmax(np.abs(v))
            name = str(x_desc[max_idx])

            # === CHANGED LOGIC HERE ===
            # Show if ANY relevant part is moving (Lower threshold to 0.1%)
            is_significant = (tw_ratio > 0.1) or (bd_ratio > 1.0)

            if is_significant:
                short_name = name[:50] + "..."
                print(f"{freq:<10.3f} | {damp:<8.2f} | {tw_ratio:<8.1f} | {bd_ratio:<8.1f} | {short_name}")
                
                y_phys = C @ v
                
                # --- ALWAYS SAVE TOWER SHAPE (If sensors exist) ---
                # Even if tw_ratio is low, we save it so you can see the "reaction"
                if len(heights) >= 3:
                    e_x = np.linalg.norm(y_phys[mom_indices_x])
                    e_y = np.linalg.norm(y_phys[mom_indices_y])
                    is_fa = e_y > e_x
                    t_idx = mom_indices_y if is_fa else mom_indices_x
                    direction = "Fore-Aft" if is_fa else "Side-Side"
                    
                    col_key = f"{freq:.3f}Hz_Tw_{short_name}"
                    all_mode_data[col_key] = pd.Series(y_phys[t_idx], index=heights)
                    
                    # Only PLOT if it's actually a dominant Tower mode (keeps plots clean)
                    if plot_tower and (tw_ratio > 1.0):
                        vals = np.real(y_phys[t_idx])
                        norm_vals = vals / np.max(np.abs(vals))
                        plt.figure(figsize=(4, 5))
                        plt.plot(norm_vals, heights, 'o-', linewidth=2, color='blue')
                        plt.title(f"TOWER {direction}\nFreq: {freq:.3f} Hz")
                        plt.xlabel("Norm. Moment")
                        plt.ylabel("Tower Height")
                        plt.axvline(0, c='k', ls='--', alpha=0.3)
                        plt.grid(True)
                        plt.show()
                        plt.close()

                # --- ALWAYS SAVE BLADE SHAPE (If sensors exist) ---
                if len(bld_spans) >= 3:
                    e_f = np.linalg.norm(y_phys[bld_flap_indices])
                    e_e = np.linalg.norm(y_phys[bld_edge_indices]) if bld_edge_indices else 0
                    is_flap = e_f > e_e
                    b_idx = bld_flap_indices if is_flap else bld_edge_indices
                    
                    col_key = f"{freq:.3f}Hz_Bd_{short_name}"
                    all_mode_data[col_key] = pd.Series(y_phys[b_idx], index=bld_spans)
                        
    # --- 6. EXPORT TO EXCEL ---
    print("-" * 110)
    
    if all_mode_data:
        filename = "Extracted_Mode_Shapes.xlsx"
        print(f"Exporting modes to {filename}...")
        
        # Split Data by Type based on KEY Name
        tower_dict = {k: v for k, v in all_mode_data.items() if "_Tw_" in k}
        blade_dict = {k: v for k, v in all_mode_data.items() if "_Bd_" in k}
        
        # Create DataFrames
        df_tower = pd.concat(tower_dict, axis=1) if tower_dict else pd.DataFrame()
        df_blade = pd.concat(blade_dict, axis=1) if blade_dict else pd.DataFrame()
        
        # Save using ExcelWriter
        try:
            with pd.ExcelWriter(filename) as writer:
                if not df_tower.empty:
                    df_tower.astype(str).to_excel(writer, sheet_name='Tower_Modes')
                    print(f" -> Saved sheet: Tower_Modes ({df_tower.shape[1]} modes)")
                    
                if not df_blade.empty:
                    df_blade.astype(str).to_excel(writer, sheet_name='Blade_Modes')
                    print(f" -> Saved sheet: Blade_Modes ({df_blade.shape[1]} modes)")
                    
            print(f"Export Complete. File saved at: {os.path.abspath(filename)}")
            
        except Exception as e:
            print(f"Error saving Excel file: {e}")
            print("Check if the file is open in another program.")
            
    else:
        print("No modes were stored.")

# ==============================================================================
#  EXECUTION BLOCK
# ==============================================================================

if __name__ == "__main__":
    lin_file_path = 'outputs_lin_No_SubD/parametric/case_1.1.lin'
    
    if os.path.exists(lin_file_path):
        analyze_modes(lin_file_path, freq_min=0.0, freq_max=5.0)
    else:
        print(f"File not found: {lin_file_path}")