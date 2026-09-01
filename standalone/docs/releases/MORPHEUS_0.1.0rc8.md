# MORPHEUS 0.1.0rc8 release candidate

MORPHEUS 0.1.0rc8 aligns semiexperimental refinement with the current MATRIX
ownership and version contracts. `VERSION` is the common release source for
MORPHEUS, its required MATRIX components and the generated release manifest.

## Public scope

- XYZ, enriched-XYZ, job-file and legacy-MSR inputs;
- rotational-constant and moment-of-inertia objectives;
- frozen SMITH/SONIC and symmetry-adapted Cartesian coordinate models;
- primitive constraints, predicates and data-supported parameter classes;
- analytic Jacobians, covariance, influence, leave-one-out and multistart
  diagnostics;
- HTML, LaTeX, PDF, CSV and JSON results;
- ensemble class refinement and publication benchmark generation.

The implementation is divided into coordinate-model, measurement/iteration
and output/diagnostic services, with `fit.py` retaining only orchestration of
the two refinement paths and checkpoint flow.

## Installation

```bash
python3.11 -m venv /chosen/path/morpheus-venv
source /chosen/path/morpheus-venv/bin/activate
python -m pip install --find-links /path/to/release/wheels \
  matrix-morpheus==0.1.0rc8
morpheus doctor
```

The clean-install verifier exercises the installed `morpheus` entry point, the
bundled water example, a SONIC refinement and the complete report/manifest
surface outside the source checkout.

## Release gate

The formal tag is created only after the coherent wheelhouse, isolated imports,
dependency graph, complete test suite and clean MORPHEUS installation all pass
from the same committed revision.
