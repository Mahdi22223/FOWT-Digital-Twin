#!/usr/bin/env bash
# OpenFAST study (Sections 3-4): full reproduction from the 12 Case_*.csv files and case_1.1.lin.
# Run from the repository root. Data location: configs/openfast.yaml (or EFTWIN_OPENFAST_DIR, EFTWIN_LIN_FILE).
# The virtual-sensing step dominates the run time.
set -euo pipefail
C="${1:-configs/openfast.yaml}"
python scripts/openfast/identify.py --config "$C"              # Table 3, Fig. A1, mode shapes
python scripts/openfast/shape_check.py --config "$C"           # Table A1
python scripts/openfast/linear_reference.py --config "$C"      # Table 3, linear column
python scripts/openfast/virtual_sensing.py FA --config "$C"    # Sections 4.4-4.8
python scripts/openfast/virtual_sensing.py SS --config "$C"
python scripts/openfast/summarise.py --config "$C"             # Tables 5, A2, A3; Figs. 11-12
python scripts/openfast/robustness.py FA --config "$C"         # Table A4 (one run per direction)
python scripts/openfast/robustness.py SS --config "$C"
python scripts/openfast/excerpts.py --config "$C"              # time series for Figs. 8 and 13
python scripts/make_tables.py vs_results l1_channels robust shape_check --openfast "$C"
python scripts/make_figures.py openfast --openfast "$C"
