# Open items

Points found while aligning the repository with the finalized revision code and the revised manuscript. None of
them has been changed in the code: the implementation reproduces the reported numbers as they are. For each item
the effect and what a change would require are stated. Items marked **[raw data]** need a rerun on the OpenFAST or
FOCAL data before anything is changed.

## Results that must be regenerated

1. **[raw data] OpenFAST outputs.** The revision package contains the saved FOCAL outputs (`results/focal`) but not the
   CSV/NPZ outputs of the OpenFAST study, so `results/openfast` is empty and the OpenFAST tables and figures are
   available only as the final versions in `results/reference`. Run `bash scripts/run_openfast.sh` once with the 12
   case files and `case_1.1.lin`, check `results/tables` against `results/reference/tables` (Table 3 is assembled as
   described in `docs/reproduce.md`), and commit the small CSV/NPZ files. Intermediate records
   (`results/*/intermediate`) and caches stay untracked.

## Inconsistencies between code, manuscript and response letter

2. **[raw data] Gauge heights of the shape check (Table A1).** The caption of Table A1, the OpenFAST configuration
   table and the ElastoDyn input give the gauges at tower nodes 1, 3, 5, 7, 10, 13, 15, 17, 20 of 20 (11.9-85.7 m);
   the repository uses these heights. The default of the revision script was equidistant heights 10-87.6 m, which its
   docstring describes as the heights stated in the paper. Check that `scripts/openfast/shape_check.py` reproduces
   `table_shape_check.tex` (the R^2 = 1.000 column with platform motion requires the true heights), and that the
   sensor description of the simulation model (Section 3) uses the node heights. `--heights equidistant` gives the alternative.

3. **[raw data] Numbers without a script.** The following statements are not produced by any script of the revision
   package, so their exact conventions (pooling, standardisation, band) could not be checked. Small scripts can be
   added once the convention is confirmed, and checked against the paper on the raw data:
   * `sec:rank` (effective dimension): five principal components explain 99.997 % of the variance of the 18 standardised FA
     channels, six explain 99.9999 % (SS); per OLC at least 99.99 % (FA) and 99.94 % (SS).
   * `sec:focal_vs` and response letter: three principal components explain 99.5 % (FA) / 99.4 % (SS) of the variance
     of the four corresponding simulated channels and 86.9 % / 93.3 % in the basin test.
   * `sec:focal_sysid`: coherence 0.02-0.03 between the accelerometers above 2 Hz.
   * `sec:vs_results`: "less than 0.01 change from r = 120 to r = 200" for the Kalman form; the OpenFAST grid ends at
     r = 120 (`virtual_sensing.kalman_ranks`). The FOCAL statement (r = 200 to 250) is covered by the FOCAL grid.
   * Response letter (Reviewer 4, independence of training and test data): "Compared with a standard split, the
     median R^2 changes by less than 0.02."

4. **Framework figure (Fig. 4 of the original submission).** The schematic of the virtual sensor
   (`assets/figure_4_virtual_sensing.png` in the tag `v0.2-original-submission`) shows the original estimator: the
   pseudoinverse `b_k = Phi_sparse^+ y_k` and a window prediction `Y_win = Re(Phi_phys diag(b_k) V_W)` with a rolling
   horizon. The revised Section 2.4.2 and Algorithm `alg:vs` use the Tikhonov estimate and the causal nowcast
   `Re(Phi_L b_k)` at every sample (plus the Kalman form). If the figure stays in the paper, its online part should be
   redrawn; it was removed from the repository front page.

5. **Rotor frequency in the coherence-bound composition (FOCAL).** `scripts/focal/coherence_bound.py` centres the
   rotor-harmonic windows at n x 0.1235 Hz +- 0.03 Hz (n = 3, 6, 9, 12), whereas Table 6 (note b) and Fig. 14 use
   3P = 0.366 Hz (1P = 0.122 Hz). The bound itself (Table 7) does not depend on this value; only the "share near the
   rotor harmonics" (66-81 % of the moments' in-band variance, `sec:focal_vs`) does. Keep as is, or set
   `coherence_bound.rotor_1p_hz: 0.122` and rerun `coherence_bound.py` **[raw data]**.

## Packaging

6. **Supplementary video.** The Zenodo video (10.5281/zenodo.20110438) and `assets/digital_twin_demo.mp4` (kept in the
   tag `v0.2-original-submission`, removed from the main branch) show the original-submission version.
7. **Release and DOI.** After merging, create a GitHub release (for example `v1.0.0`) so that Zenodo archives the
   revised version; check whether 10.5281/zenodo.20110415 is the concept DOI or the DOI of the original version and
   update `CITATION.cff` and the README accordingly. Making the repository public releases the revised results before
   the paper is accepted; decide on the timing.
8. **Data availability.** The OpenFAST case files and the `.lin` file are not public. Depositing them (for example on
    Zenodo, about 0.5 GB) would make the OpenFAST part reproducible by others; see `data/README.md`.
