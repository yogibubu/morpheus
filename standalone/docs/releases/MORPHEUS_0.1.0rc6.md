# MORPHEUS 0.1.0rc6 release candidate

MORPHEUS 0.1.0rc6 freezes the publication implementation for algorithmic
construction and validation of topology-constrained semiexperimental
equilibrium-structure models.

## Public scope

- canonical XYZ, enriched-XYZ, job-file and legacy-MSR import;
- isotopologue rotational constants with vibrational/electronic corrections;
- moment-of-inertia or rotational-constant objectives with declared weights;
- SMITH/SONIC and symmetry-adapted Cartesian active spaces;
- primitive constraints, predicates and data-supported parameter classes;
- rank, conditioning, covariance, influence, leave-one-out and multistart
  diagnostics;
- final-validation gate and shareable HTML, LaTeX, PDF, CSV and JSON results;
- ensemble class refinements and checked publication benchmark generation.

The ownership boundary is explicit. ORACLE supplies frozen perception and the
primitive/B source; SMITH supplies non-redundant SONIC; LINK realizes finite
Cartesian corrections. MORPHEUS constructs and solves the inverse model and
does not duplicate those three services. ARCHITECT/ZION is outside the scope of
semiexperimental refinement.

## Installation

```bash
python3.11 -m venv /chosen/path/morpheus-venv
source /chosen/path/morpheus-venv/bin/activate
python -m pip install --find-links /path/to/release/wheels \
  matrix-morpheus==0.1.0rc6
morpheus doctor
```

The distribution exposes both `morpheus` and `matrix-morpheus`. It includes a
small standalone water example and does not require a MATRIX checkout,
`MATRIX_HOME`, or developer-machine paths.

## Validation gate

- focused MORPHEUS suite: 42 tests, zero failures before release assembly;
- a clean wheel installation runs `morpheus doctor` and a complete water fit;
- the clean fit checks the `#MORPHEUS` section, result manifest, reports and LINK
  back-transformation provenance;
- the complete MATRIX suite remains the integration gate;
- the release manifest records the exact commit, component versions and SHA256
  digest of every artifact.

The intended formal tag is `morpheus-v0.1.0rc6` after the manuscript branch and
release archive have passed the final gate.
