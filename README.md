# Physics-Interpretable Digital Twin for Floating Offshore Wind Turbines (FOWT)

**Author:** Dr. Mohammad Mahdi Abaei (Aalto University)
**Status:** 🚀 Pre-release (Manuscript under preparation for RESS/MSSP)

## Overview
This repository contains the implementation of a **Physics-Interpretable Digital Twin** framework for FOWT. The approach utilizes **Rank-Optimized Hankel-Dynamic Mode Decomposition (Hankel-DMD)** to decouple structural dynamics from stochastic environmental loads (waves/wind).

This method addresses the "active but unmeasured" loading problem, enabling real-time Virtual Sensing without the need for computationally expensive PDEs.

## Key Features
* **Data-Driven:** No prior knowledge of mass/stiffness matrices required.
* **Physics-Interpretable:** Extracts explicit spectral properties (eigenvalues/modes) of the system.
* **Robust:** Validated against high-fidelity OpenFAST simulations (NREL 5MW).

## Code Availability
The full source code and datasets for this project are currently under embargo pending the peer-review process of the associated journal publication. 

**The complete codebase will be released here under the MIT License immediately upon acceptance of the paper.**

## Citation

If you use this work or code in your research, please cite our paper:
```bibtex
@misc{abaei2026equationfreedigitaltwinsnonlinear,
      title={Equation-Free Digital Twins for Nonlinear Structural Dynamics}, 
      author={Mohammad Mahdi Abaei and Ahmad BahooToroody and Arttu Polojärvi and Heikki Remes and Ulf Tyge Tygesen and Mikko Suominen and Michael Beer},
      year={2026},
      eprint={2605.00950},
      archivePrefix={arXiv},
      primaryClass={eess.SP},
      url={[https://arxiv.org/abs/2605.00950](https://arxiv.org/abs/2605.00950)}, 
}

## Contact
For inquiries or early access for collaboration, please contact:
**Dr. Mohammad Mahdi Abaei** Academy Research Fellow, Aalto University  
[Email Address] mohammad.abaei@aalto.fi
