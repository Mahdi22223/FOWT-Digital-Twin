# Repository notes

This repository is a standalone implementation of the Hankel-DMD identification and virtual-sensing framework.

Manuscript-specific wording, figure layout, table formatting, and response-to-reviewer items are not treated as software issues here.

## External data

The raw OpenFAST case files and linearisation file are not bundled with the repository. The FOCAL Campaign IV raw measurements are also distributed externally by their data owners.

See `data/README.md` for the expected file layout and environment-variable overrides. The package, synthetic quickstart, tests, and saved lightweight outputs remain usable without these raw datasets.

## Release metadata

Before creating a tagged public release, verify whether `10.5281/zenodo.20110415` is the Zenodo concept DOI or a version-specific DOI for the earlier repository state, then update `CITATION.cff` and the README if needed.

This is a release and archiving task and does not affect the numerical implementation.

Code correctness is checked by the automated test suite and CI.
