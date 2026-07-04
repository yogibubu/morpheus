Camphor model-validation data
=============================

This directory contains the camphor semiexperimental validation data used in
the MORPHEUS manuscript and Supporting Information.

The accepted reference run is the no-Kraitchman BDPCS3-predicate model from
the MATRIX semiexperimental camphor stability suite:

`/Users/vincenzobarone/Documents/git/software/matrix/working/semiexp/camphor_bdpcs3/stability_suite_full/runs/ref_no_kra_full_validation`

The retained model uses 178 BDPCS3 predicates and no fixed hydrogen
coordinates. C-H distances are treated with strict soft predicates, while
Kraitchman-derived relations are retained only as diagnostics. The archived
stability runs show that tight Kraitchman predicates and predicate sets without
hydrogen-angle information produce chemically unacceptable structures even
when the rotational residual is superficially competitive.

Key files:

- `camphor_geometry_parameters.csv`: final topological parameters, propagated
  errors, and final-minus-initial differences.
- `camphor_stability_summary.csv` and `.json`: comparison of accepted and
  rejected predicate/coordinate variants.
- `camphor_final_validation.json` and `camphor_final_validation_runs.csv`:
  final reproducibility, robustness, multistart, sigma-scan, and
  leave-predicate-group-out checks.
- `kraitchman_predicate_vs_bdpcs3.csv`: Kraitchman-derived trial predicates
  compared with BDPCS3 reference parameters.
- `camphor_se_no_kraitchman.xyz`: final accepted Cartesian structure.
