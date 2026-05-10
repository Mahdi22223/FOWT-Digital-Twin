import numpy as np
import pandas as pd
try:
    # Try importing from the new library location
    from pyFAST.input_output import FASTOutputFile
except ImportError:
    # Fallback for older installations
    from openfast_toolbox.io import FASTOutputFile
    
output_file_path = 'outputs - Org/parametric/case_1.outb' 
csv_output_path = 'outputs - Org/Case_1.csv'

def extract_for_dmd(output_file_path,csv_output_path):
    print(f"Reading OpenFAST output: {output_file_path}...")
    
    # 1. Load the Output File (.out or .outb) into a Pandas DataFrame
    # This handles the binary or text formatting automatically
    try:
        df = FASTOutputFile(output_file_path).toDataFrame()
    except Exception as e:
        print(f"Error reading file. Make sure you point to a .out or .outb file, not .fst or .lin! \n{e}")
        return

    # 2. Select Acceleration and Moment Columns
    # We filter columns based on standard OpenFAST naming conventions:
    # 'NcIMU' = Nacelle IMU (Acceleration)
    # 'TwHt'  = Tower Height nodes (Acceleration)
    # 'Ptfm'  = Platform (Acceleration/Motion)
    # 'TwrBs' = Tower Base (Moments)
    # 'RootM' = Blade Root (Moments)
    
    # Define keywords for the columns you want
    # "Ta" or "AL" usually indicates acceleration
    # "Myt", "Mxt" indicates Bending Moments
    # target_keywords = [
    #     'Time',       # Always keep Time
    #     'TwHt',       # Tower Node Accels (if you requested them in ElastoDyn)
    #     'TwrBsM',     # Tower Base Moments (Mxt, Myt, Mzt)
    #     'YawBrTAxp'   # (Yaw Bearing Translational Acceleration in x-direction).
    #     'YawBrTAyp'   # (Yaw Bearing Translational
    #     # 'RootM'       # Blade Root Moments
    #     #'NcIMUTA',    # Nacelle Accel (Translation)

    # ]
    
# --- 2. DEFINE THE TARGET ORDER ---
    # We list the *substrings* we want to find, in the specific order.
    targets_in_order = ['Time']

    # A. 9 Acceleration Gauges (Fore-Aft then Side-Side)
    for i in range(1, 10):
        targets_in_order.append(f"TwHt{i}ALxt") 
        targets_in_order.append(f"TwHt{i}ALyt")

    # B. 9 Moment Gauges (Side-Side then Fore-Aft)
    for i in range(1, 10):
        targets_in_order.append(f"TwHt{i}MLxt") 
        targets_in_order.append(f"TwHt{i}MLyt")

    # C. Yaw Bearing Accelerations
    targets_in_order.append('YawBrTAxp') 
    targets_in_order.append('YawBrTAyp')

    # D. Tower Base Moments
    # targets_in_order.append('TwrBsMxt')
    # targets_in_order.append('TwrBsMyt')
    # targets_in_order.append('TwrBsMzt')

    # --- 3. MATCH TARGETS TO ACTUAL COLUMNS ---
    # This logic looks for the target string inside the actual column name
    # (Handling the issue where OpenFAST adds units like [m/s^2])
    
    final_cols = []
    
    for target in targets_in_order:
        # scan all columns in the dataframe to find the one that matches
        found_match = False
        for actual_col in df.columns:
            # Check if target is inside the column name (Substring Match)
            if target in actual_col:
                final_cols.append(actual_col)
                found_match = True
                break # Stop searching for this specific target, move to next
        
        if not found_match:
            # Optional: Print warning if a specific gauge is missing
            pass 

    if not final_cols:
        print("  Warning: No matching columns found.")
        return

    dmd_df = df[final_cols]


    # # 4. Save to CSV

    dmd_df.to_csv(csv_output_path, index=False)  
  
    print("Done! Here are the first few columns extracted:")
    print(dmd_df.columns.tolist()[:10])

# -----------------------------------

# --- EXECUTE ---

# -------------------------------------
output_file_path = 'outputs - Org/parametric/case_1.outb' 
csv_output_path = 'outputs - Org/Case_1.csv'
Num_Cases = 12
# !! Close Opening Excell file to prevent Error

if __name__ == "__main__":
    
    # Loop from 1 to 12
    for i in range(1, Num_Cases+1):
        # 1. Define the input file name for this case
        input_file = f'outputs - Org/parametric/case_{i}.outb'
        
        # 2. Define the output CSV name for this case
        output_csv = f'outputs - Org/Case_{i}.csv'
        
        # 3. Call the function
        extract_for_dmd(input_file, output_csv)
    
    print("\nBatch processing complete!")
    

