# Semiexperimental Geometry Examples

These examples are small end-to-end inputs for the ORACLE Cartesian/GIC
semiexperimental equilibrium-geometry workflow. Each case contains a parent
XYZ file and a TOML isotopologue table accepted by both the CLI and GUI.

Run one case with:

```bash
python -m matrix semiexp \
  --xyz examples/semiexp/water/parent.xyz \
  --observations examples/semiexp/water/isotopologues.toml \
  --outdir working/examples/water_semiexp
```

The outputs include the fitted Cartesian geometry, non-redundant GIC
parameters, propagated errors, residuals, Kraitchman diagnostics, an HTML
report and `semiexp_tables.tex` for manuscript tables.

The phthalic anhydride literature-data case uses a job file because it needs
weighted reBO primitive-coordinate predicates:

```bash
python -m matrix semiexp \
  --job examples/semiexp/phthalic_anhydride/phthalic_anhydride_predicates.mse.toml \
  --outdir working/semiexp/phthalic_anhydride
```

The succinic anhydride case is the heavy-atom-only member of the anhydride
series. The production comparison fixes the local hydrogen frame because the
published isotopologues do not substitute H atoms:

```bash
python -m matrix semiexp \
  --job examples/semiexp/succinic_anhydride/succinic_anhydride_fixed_h.mse.toml \
  --outdir working/semiexp/succinic_anhydride_fixed_h
```

The JPCL multi-molecule test is the parent-only anhydride ensemble.  It uses
only the parent rotational constants of maleic, phthalic and succinic
anhydrides, refines shared short/long C-C and carbonyl/single C-O stretch
corrections across the homologous series, and keeps C-H distances at the BDPCS3
reference values.  The ordinary single-molecule fits with all available
isotopologues are used only as validation references:

```bash
python -m matrix semiexp-ensemble \
  --job examples/semiexp/anhydrides_parent_only/anhydrides_parent_only.mse-ensemble.toml \
  --outdir working/semiexp/anhydrides_parent_only
```

The output includes a text report, CSV summaries, covariance/correlation
matrices, the scientific `ensemble_manifest.json`, and the workflow
`run_manifest.json` with file checksums.  Ensemble reports include an explicit
model status: `accepted` means the model passes the rank, conditioning and
support checks; `review` means the fit is usable but contains flagged class
correlations; `rejected` means the model should not be used for production
corrections without changing classes, priors or atom typing.

The older full-isotopologue anhydride ensemble, including no-prior/soft-prior/
hard-constraint comparisons and prior-strength scans, is retained as an
algorithmic stress test:

```bash
python -m matrix semiexp-ensemble-paper \
  --job examples/semiexp/anhydrides_ensemble/anhydrides_ensemble.mse-ensemble.toml \
  --paper-dir doc/papers/ensemble_jpcl \
  --outdir working/semiexp/anhydrides_full_analysis
```

Single-structure paper benchmark tables are generated from a checked snapshot:

```bash
python -m matrix semiexp-benchmark \
  --snapshot benchmarks/semiexp_msr/golden/semiexp_paper_regression.json \
  --outdir benchmarks/semiexp_msr/generated
```

The two glycine conformers can be fitted together directly from the legacy MSR
benchmark inputs.  C-H distances are kept at the BDPCS3 reference values.  The
`synthon` variant uses continuous effective atomic numbers (`Zeff`) with a
threshold to define transferable atom types across the two different conformer
numberings:

```bash
python -m matrix semiexp-ensemble \
  --job examples/semiexp/glycine_ensemble/glycine_conformers_synthon.mse-ensemble.toml \
  --outdir working/semiexp/glycine_ensemble_synthon
```

Scan the continuous atom-typing threshold with:

```bash
python -m matrix semiexp-ensemble-synthon-scan \
  --job examples/semiexp/glycine_ensemble/glycine_conformers_synthon.mse-ensemble.toml \
  --outdir working/semiexp/glycine_synthon_threshold_scan_wide \
  --threshold 0.010 --threshold 0.035 --threshold 0.075 \
  --threshold 0.100 --threshold 0.150 --threshold 0.250
```

Search and use the local `se_geometries` library for reference-assisted
starting structures with:

```bash
python -m matrix multistructure-reference-search \
  --query-xyz path/to/query.xyz \
  --outdir working/multistructure_reference_search/query

python -m matrix multistructure-build-reference-geometry \
  --query-xyz path/to/query.xyz \
  --outdir working/multistructure_reference_geometry/query
```
