""" Example to run a set of OpenFAST simulations (parametric study)

Adapted from:
    https://github.com/OpenFAST/python-toolbox/blob/main/pyFAST/case_generation/examples/Example_Parametric.py


This script uses a reference directory (`ref_dir`) which contains a reference input file (.fst)
1) The reference directory is copied to a working directory (`work_dir`).
2) All the fast input files are generated in this directory based on a list of dictionaries (`PARAMS`).
For each dictionary in this list:
   - The keys are "path" to a input parameter, e.g. `EDFile|RotSpeed`  or `FAST|TMax`.
     These should correspond to the variables used in the FAST inputs files.
   - The values are the values corresponding to this parameter
For instance:
     PARAMS[0]['EDFile|RotSpeed']       = 5
     PARAMS[0]['InflowFile|HWindSpeed'] = 10

3) The simulations are run, successively distributed on `nCores` CPUs.
4) The output files are read, and averaged based on a method (e.g. average over a set of periods,
    see averagePostPro in postpro for the different averaging methods).
   A pandas DataFrame is returned

"""
import os
import pyFAST.case_generation.case_gen as case_gen
import pyFAST.case_generation.runner as runner
import pyFAST.input_output.postpro as postpro
from pyFAST.input_output.fast_input_file import FASTInputFile

# --- Parameters for this script
FAST_EXE  = r'C:\Simulations\openfast_x64.exe'  # Location of a FAST exe (and dll)
ref_dir   = r'C:\Simulations\templateDir'       # Folder where the fast input files are located (will be copied)
main_file = 'Main.fst'            # Main file in ref_dir, used as a template
work_dir  = 'outputs/parametric/'         # Output folder (will be created)

# --- Defining the parametric study  (list of dictionnaries with keys as FAST parameters)
# --- HydroDyn Parameters to Vary ---
wave_pairs = [
    # (Hs, Tp)
    (1.0, 6.0),  (1.0, 9.0),  (1.0, 12.0), # Small waves (Short, Med, Long period)
    (2.0, 8.0),  (2.0, 10.0), (2.0, 14.0), # Medium waves
    (3.0, 10.0), (3.0, 12.0), (3.0, 16.0), # Large waves
    (4.0, 9.0), (4.0, 12.0), (4.0, 20.0)  # Extreme waves (Steep (9-acceleration) vs Swell (20-long))
   # T_p = 20 excellent for System Identification to
   # excite the low-frequency modes of a floating platform.
]

# Unpack pairs into your lists
WAVE_HS = [p[0] for p in wave_pairs]
WAVE_TP = [p[1] for p in wave_pairs]

Wave_Dir = [0, 45, 90] * 4  # Result: [0, 30, 0, 30, ...]

# --- ElastoDyn Parameters to Vary (Example from previous) ---
RPM   = [ 12 ]*12  # Result: [12,12, ...]  fixed RPM at Max
PITCH = [ 0 ]*12

BaseDict = {'TMax': 3600, 'DT': 0.0125, 'DT_Out': 0.05}
PARAMS=[]



# --- PREPARATION: Define the Output Lists for DMDc ---

# 1. Generate the 9-Node Tower Sensors
#    We will create the list for Kinematics (X) and Loads (Y)
tower_sensors = []

for n in range(1, 10):
    # --- KINEMATICS (Motion - The "State" X) ---
    tower_sensors.append(f"TwHt{n}TDxt") # Translational Deflection X (Fore-Aft)
    tower_sensors.append(f"TwHt{n}TDyt") # Translational Deflection Y (Side-Side)
    tower_sensors.append(f"TwHt{n}ALxt") # Acceleration X (Fore-Aft)
    tower_sensors.append(f"TwHt{n}ALyt") # Acceleration Y (Side-Side)
    
    # --- LOADS (Force/Moment - The "Output" Y) ---
    # Based on your Excel file keys: FLx, FLy, FLz, MLx, MLy, MLz
    
    # Forces (Shear & Axial)
    tower_sensors.append(f"TwHt{n}FLxt") # Force Local X (Fore-Aft Shear)
    tower_sensors.append(f"TwHt{n}FLyt") # Force Local Y (Side-Side Shear)
    tower_sensors.append(f"TwHt{n}FLzt") # Force Local Z (Axial/Vertical Force)
    
    # Moments (Bending & Torsion)
    tower_sensors.append(f"TwHt{n}MLxt") # Moment Local X (Side-Side Bending)
    tower_sensors.append(f"TwHt{n}MLyt") # Moment Local Y (Fore-Aft Bending)
    tower_sensors.append(f"TwHt{n}MLzt") # Moment Local Z (Torsion)

# 2. Validation & "Ground Truth" Variables (Unchanged)

# Combine all into one list for ElastoDyn
ed_outlist = tower_sensors + [
    "RotSpeed", 
    "BldPitch1",
    "QD_TFA1", "QD_TSS1" # Validation modes
]

# Variables for ServoDyn (Generator/Control)
servo_vars = [
    "GenPwr",
    "GenTq"
]

# Variables for InflowWind (Wind speeds)
inflow_vars = [
    "Wind1VelX"
]
# 3. HydroDyn Inputs (The "Control" U)
hydro_outlist = [
    "HydroFxi",  # Total Hydro Force X
    "HydroFyi",  # Total Hydro Force Y
    "Wave1Elev"  # Wave Elevation
]

# --- MAIN LOOP ---

for i, (rpm, pitch, hs, tp, wd) in enumerate(zip(RPM, PITCH, WAVE_HS, WAVE_TP, Wave_Dir)): 
    p = BaseDict.copy()
    
    # 1. Operational Conditions
    p['EDFile|RotSpeed']       = rpm
    p['EDFile|BlPitch(1)']     = pitch
    p['EDFile|BlPitch(2)']     = pitch
    p['EDFile|BlPitch(3)']     = pitch

    # 2. Tower Sensor Configuration
    p['EDFile|NTwGages'] = 9
    p['EDFile|TwrGagNd'] = [1, 3, 5, 7, 10, 13, 15, 17, 20] # Ensure these exist in your tower mesh
    p['EDFile|OutList']  = ed_outlist  # <--- Apply our big list here

    # InflowWind (Fixes the Wind1VelX error)
    p['InflowFile|OutList'] = inflow_vars
    
    # ServoDyn (Fixes the GenPwr error)
    p['ServoFile|OutList']  = servo_vars
    
  # 3. Environmental Conditions
    p['HydroFile|RdtnDT']  = BaseDict['DT']
    p['HydroFile|WaveHs']  = hs 
    p['HydroFile|WaveTp']  = tp 
    p['HydroFile|WaveDir'] = wd 
    p['HydroFile|OutList'] = hydro_outlist

    # 4. Case Naming
    p['__name__'] = 'case_' + str(i+1) 
    PARAMS.append(p)
    
    
print("Number of cases generated:", len(PARAMS))
print("Example Case 1:", PARAMS[11])

# --- Generating all files in a workdir
fastFiles=case_gen.templateReplace(PARAMS, ref_dir, outputDir=work_dir, removeRefSubFiles=True, main_file=main_file, oneSimPerDir=False)
print('Main input files written:')
print(fastFiles)

# --- Creating a batch script just in case
runner.writeBatch(os.path.join(work_dir,'_RUN_ALL.bat'), fastFiles,fastExe=FAST_EXE)
# --- Running the simulations
runner.run_fastfiles(fastFiles, fastExe=FAST_EXE, parallel=True, showOutputs=True, nCores=24)
# --- Simple Postprocessing
outFiles = [os.path.splitext(f)[0]+'.out' for f in fastFiles]
avg_results = postpro.averagePostPro(outFiles, avgMethod='periods', avgParam=1, ColMap = {'WS_[m/s]':'Wind1VelX_[m/s]'},ColSort='WS_[m/s]')
outputFile = './outputs/Parametric.csv'
print('>> Average results: (written to: {})'.format(outputFile) )
avg_results.to_csv(outputFile, index=False)








