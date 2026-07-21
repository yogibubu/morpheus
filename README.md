# MORPHEUS

This repository accompanies the manuscript *Algorithmic Construction and
Exploitation of Topology-Constrained Refinement Models for Semiexperimental
Equilibrium Structures*.

It contains:

- the manuscript and Supporting Information sources;
- the standalone MORPHEUS 0.1.0rc6 source distribution in `standalone/`;
- runnable semiexperimental examples and manuscript benchmark inputs;
- machine-readable camphor validation data in `data/camphor/`;
- generated tables, figures, and numerical audit material used in the paper.

The standalone snapshot was exported from MATRIX revision
`187ea913261fc8a20b4260b10175b0f37d74c87d`. It contains only the components
needed to build and run MORPHEUS; the private development monorepo is not
required.

## Quick start

See [`standalone/README.md`](standalone/README.md) for source installation,
release building, and the bundled end-to-end example. The shortest validation
path is the water example; the anhydride, glycine, norcamphor, and testosterone
directories provide the publication-scale inputs.

## Reproducibility map

- `standalone/packages/`: source of MORPHEUS and its required components.
- `standalone/packages/matrix-morpheus/examples/semiexp/`: runnable inputs.
- `standalone/benchmarks/semiexp_msr/`: legacy and paper-regression inputs.
- `data/camphor/`: accepted geometry, predicate audit, stability runs, and
  final-validation summaries.
- `generated/`: manuscript tables generated from the archived results.
- `supporting_information.tex`: representative complete input and output
  excerpts, followed by detailed numerical diagnostics.

## Citation

The coordinate-generation method used by MORPHEUS is described in:

V. Barone, *SONIC: Symmetry-Oriented Non-redundant Internal Coordinates*,
arXiv:2607.16550 (2026), <https://arxiv.org/abs/2607.16550>.

Please also cite the MORPHEUS manuscript when it becomes available.

## License

The standalone software is distributed under the BSD 3-Clause License. The
manuscript and article figures remain scholarly publication material.
