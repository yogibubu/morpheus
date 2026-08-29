# MORPHEUS 0.1.0rc8 quick start

MORPHEUS constructs and solves auditable semiexperimental equilibrium-structure
models from isotopologue rotational constants, vibrational/electronic
corrections and optional structural predicates. It consumes a frozen molecular
state and SONIC coordinate contract: ORACLE owns perception and primitive/B
rows, SMITH owns the non-redundant SONIC definition, and LINK alone realizes
finite internal-coordinate corrections in Cartesian space.

## Clean installation

Choose every directory on the target machine. No source checkout or path from
the developer's computer is required.

```bash
python3.11 -m venv /chosen/path/morpheus-venv
source /chosen/path/morpheus-venv/bin/activate
python -m pip install --upgrade pip
python -m pip install --find-links /path/to/morpheus-release/wheels \
  matrix-morpheus==0.1.0rc8
morpheus doctor
```

## Run the bundled example

```bash
morpheus examples /chosen/path/morpheus-examples
morpheus fit \
  --xyz /chosen/path/morpheus-examples/water/parent.xyz \
  --observations /chosen/path/morpheus-examples/water/isotopologues.toml \
  --xyzin /chosen/path/morpheus-water.xyzin \
  --outdir /chosen/path/morpheus-water-run \
  --coordinate-model gic \
  --include-r0-report \
  --max-iter 2
```

The run writes the refined Cartesian geometry, exact SONIC definitions,
rotational residuals, covariance and influence diagnostics, a machine-readable
manifest, an HTML report, and standalone LaTeX/PDF results. The manifest must
identify the back-transformation as the LINK hybrid typed-SONIC service.
With `--include-r0-report`, the final report follows the full structural path:
input geometry, diagnostic `r0`, Kraitchman `rs` substitution coordinates and
final `reSE`. Input, `r0` and `reSE` are compared on the same primitive internal
coordinates; uncertainties are printed for both fitted structures.

The `fit` word is optional. The equivalent suite-level command is:

```bash
matrix semiexp --xyzin molecule.xyzin --outdir run
```

## Production inputs

MORPHEUS accepts either:

- a canonical enriched XYZ with `#ISOTOPOLOGUES` and optional `#MORPHEUS`;
- a parent XYZ/Gaussian Cartesian geometry plus CSV, JSON or TOML observations;
- a self-contained `.mse.toml` job;
- a legacy MSR input, imported into the same public contract.

Use moments of inertia as the default observable. A production result should
enable final validation and inspect rank, condition number, weight/influence
diagnostics, coordinate stability, predicate-width scans and multistarts:

```bash
morpheus fit --job molecule.mse.toml --outdir run \
  --final-validation --leave-one-out
```

For an auditable free/constrained comparison, request the comparison explicitly
and state every omitted measurement rather than relying on an implicit outlier
rule:

```bash
matrix semiexp --job molecule.msr.inp --outdir run \
  --compare-free-fit --sensitivity-fit-regularization-scale 3 \
  --exclude-rotational-constant iso_004:A \
  --exclude-rotational-constant iso_004:B \
  --exclude-rotational-constant iso_004:C
```

Both fits then use the same retained constants. The generated
`semiexp_fit_comparison.json` reports the free result, the constrained result,
the fixed 0.003 A structural gate, and every applied prior and exclusion.

## Coordinate ownership

- MORPHEUS selects the statistically supported active model and proposes each
  correction.
- ORACLE topology and point-group state remain frozen during an ordinary fit.
- SMITH supplies the frozen SONIC rows, including regular U-based ring
  puckering coordinates.
- LINK performs every finite internal-to-Cartesian realization. MORPHEUS does
  not maintain a second pseudoinverse or nonlinear back-transformation.
- The symmetry-adapted Cartesian route is an independent numerical control;
  chemical predicates remain expressed through primitive relations.

## Reproducibility gate

The release validator installs into a fresh virtual environment and runs the
bundled water fit without access to the source repository:

```bash
python3 verify_morpheus_install.py /path/to/MORPHEUS-0.1.0rc8
```
