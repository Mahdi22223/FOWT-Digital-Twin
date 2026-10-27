#!/usr/bin/env bash
# FOCAL Campaign IV study (Section 5): full reproduction from the four FOCAL_DLC12_*.csv files.
# Run from the repository root. Data location: configs/focal.yaml (or EFTWIN_FOCAL_DIR).
set -euo pipefail
C="${1:-configs/focal.yaml}"
python scripts/focal/identify.py --config "$C"                 # Table 6
python scripts/focal/response_dmd.py --config "$C"             # Table 6, response-domain column
python scripts/focal/rotor_check.py --config "$C"              # Section 5.2, rotor-speed conditioning
python scripts/focal/wideband.py all --config "$C"             # Fig. 14 data, second tower modes
python scripts/focal/sensor_kinematics.py constrained --config "$C"   # Section 5.1
python scripts/focal/virtual_sensing.py FA --config "$C"       # Section 5.3
python scripts/focal/virtual_sensing.py SS --config "$C"
python scripts/focal/summarise.py --config "$C"                # Table 7
python scripts/focal/coherence_bound.py --config "$C"          # Table 7, coherence bound
python scripts/focal/series.py --config "$C"                   # Figs. 15-16 data
python scripts/make_tables.py focal_sysid focal_vs --focal "$C"
python scripts/make_figures.py focal --focal "$C"
