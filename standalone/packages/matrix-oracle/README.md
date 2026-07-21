# matrix-oracle

Public Python API and command-line interface for ORACLE molecular perception.
The scientific implementation is provided by `matrix-chem`; optional GUI and
format adapters are loaded only when requested. ORACLE freezes both molecular
perception and the ordered redundant primitive/Wilson-B source subsequently
consumed by SMITH to construct SONIC coordinates.

```bash
oracle doctor
oracle examples ./oracle-examples
oracle analyze ./oracle-examples/benzene.xyz -o benzene.xyzin \
  --report benzene.oracle.json --human-report benzene.oracle.txt
```

The enriched file contains `#PRIMITIVES` with the primitive definitions,
reference values, B-matrix rank and deterministic fingerprints. The same
contract is available from Python:

```python
from matrix_oracle import read_primitive_contract, primitive_b_matrix
from matrix_chem import read_enriched_xyz

state = read_enriched_xyz("benzene.xyzin")
contract = read_primitive_contract("benzene.xyzin")
B = primitive_b_matrix(contract.primitives, state.coordinates_angstrom)
```

ORACLE emits one synthon model, one atomic charge and one chemical bond order.
Complete QM observations select the paired CM5/Mayer level; without both
complete vectors, ORACLE uses the paired electronegativity/Pauling level. A
state therefore never mixes availability levels, atoms or bonds. The smooth
connectivity weight used to perceive the graph is not a second bond order.
The CV posterior correction is likewise unique: the radius-aware period-line
Gaussian calibration is fixed and no CLI, GUI, or Python option selects a
second calibration. Descriptor radii and UFF non-bonded radii have separate,
explicit functions tied to their physical roles rather than a runtime scheme
selector.

ORACLE also exposes accuracy-ladder-aware geometry refinement. Core--valence
(CV) is a transversal contribution available at every valence rung; BL1
conjugation and PL1 selected-pair corrections are attached only to L1. These
names deliberately replace no legacy algorithm name:

```python
from matrix_oracle import ValenceLevel, build_accuracy_ladder_plan

atomic_numbers = [6] * 6 + [1] * 6  # benzene example
plan = build_accuracy_ladder_plan(
    contract.primitives,
    atomic_numbers,
    valence_level=ValenceLevel.L1,
)
```

See `docs/ORACLE_QUICKSTART.md` in the MATRIX source distribution for the
complete installation and reproducibility contract.

ORACLE deliberately stops at perception, redundant PIC/B construction and
primitive-space structural improvement. The current L1-to-PL1 workflow is one
calibrated application of that general operation. SMITH constructs SONIC coordinates and
ARCHITECT owns Hessian reduction, force-field parameterization and ZION. The
versioned ownership contract is available as
`matrix_oracle.oracle_scope_contract()` and is embedded in every complete
analysis report.

The optional desktop command opens the same canonical ORACLE window as the
MATRIX shell, rather than the historical multi-tool project dashboard:

```bash
python -m pip install 'matrix-oracle[qt]'
oracle-gui benzene.xyzin
```

The publication wheel is deliberately narrower than the migration worktree:
it contains only the public perception, PIC/B, refinement, CLI and canonical
GUI-launcher modules. Historical dashboard, GICForge/SMITH and GF/TRINITY
helpers are excluded from the standalone artifact and are checked by the
clean-install verifier.
