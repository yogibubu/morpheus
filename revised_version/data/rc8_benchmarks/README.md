# MORPHEUS 0.1.0rc8 benchmark audit

These machine-readable outputs back the revised p-EBN and cyclopentadiene
tables. They were generated from the inputs in
`standalone/benchmarks/semiexp_msr/inputs` with the bundled rc8 source.

- `p_ebn` is the public legacy-MSR compatibility workflow (`gic`, moment
  observable, automatic legacy profile, 120-iteration cap). The historical
  Z-matrix equality block is retained as diagnostic metadata; frozen Z-matrix
  parameters and recorded sensitivity-advisor decisions define the fit.
- `cyclopentadiene_cartesian` and `cyclopentadiene_gic` use the same
  14-isotopologue subset, omit the exactly duplicated `iso04_313` record, fit
  all three moments, and use `max_iter=120`, `damping=1e-6`, and
  `max_step=0.05`.

Each directory contains the manifest, diagnostics, and rotational-constant
comparison; the p-EBN directory additionally contains the structural,
Kraitchman, and iteration-trace CSV files used for Tables S15--S18.
