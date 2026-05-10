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
ref_dir   = r'C:\Simulations\templateDir_lin'       # Folder where the fast input files are located (will be copied)
main_file = 'Main.fst'            # Main file in ref_dir, used as a template
work_dir  = 'outputs_lin/parametric/'         # Output folder (will be created)


# --- ElastoDyn Parameters to Vary (Example from previous) ---
RPM   = [ 12]
PITCH = [ 17.1]

BaseDict = {'TMax': 100, 'DT': 0.0125, 'DT_Out': 0.05}
PARAMS=[]

# NOTE: All lists zipped must have the same length.
for i, (rpm, pitch) in enumerate(zip(RPM, PITCH)): 
    p = BaseDict.copy()
    
    # 1. ElastoDyn (EDFile) changes
    p['EDFile|RotSpeed']       = rpm
    p['EDFile|BlPitch(1)']     = pitch
    p['EDFile|BlPitch(2)']     = pitch
    p['EDFile|BlPitch(3)']     = pitch
    # 1. Tell OpenFAST exactly how many gages you want
    p['EDFile|NTwGages'] = 9
    p['EDFile|TwrGagNd'] = [1, 3, 5, 7, 10, 13, 15, 17, 20]
        # 4. Case Naming
    p['__name__']              = 'case_'+str(i+1) 
    PARAMS.append(p)

print("Number of cases generated:", len(PARAMS))
print("Example Case 1:", PARAMS[0])

# --- Generating all files in a workdir
fastFiles=case_gen.templateReplace(PARAMS, ref_dir, outputDir=work_dir, removeRefSubFiles=True, main_file=main_file, oneSimPerDir=False)
print('Main input files written:')
print(fastFiles)

# --- Creating a batch script just in case
runner.writeBatch(os.path.join(work_dir,'_RUN_ALL.bat'), fastFiles,fastExe=FAST_EXE)
# --- Running the simulations
runner.run_fastfiles(fastFiles, fastExe=FAST_EXE, parallel=True, showOutputs=False, nCores=4)
# --- Simple Postprocessing
outFiles = [os.path.splitext(f)[0]+'.out' for f in fastFiles]
# ----------------------------------------------------------------------
# LEGACY KNOWN ISSUE: the next line in the original script contained a
# non-Python placeholder rather than valid arguments:
#     avg_results = postpro.averagePostPro(outFiles, *** Moda shapes for tower ***)
# That line caused a SyntaxError on import. The line is commented out below
# so that this file remains parseable. For linear post-processing of mode
# shapes, use legacy/openfast_original/Modal_OutPut_Final_02.py, which reads
# the .lin files produced by the linearization run defined in this script.
# Numerical methodology is unchanged: nothing has been added or modified in
# the case-generation logic above.
# avg_results = postpro.averagePostPro(outFiles, *** Moda shapes for tower ***)
# outputFile = './outputs/Parametric.csv'
# print('>> Average results: (written to: {})'.format(outputFile) )
# avg_results.to_csv(outputFile, index=False)
# print(avg_results)
# ----------------------------------------------------------------------


# ------------- 

 
    
 
    
 
    
 
    
 
    
 
    