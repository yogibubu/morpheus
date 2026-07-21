# Norcamphor semiexperimental fit

This example stores the norcamphor DPCS3 Cartesian geometry, experimental
rotational constants from Table 3 of `data.pdf`, and HPCS2 vibrational
corrections from the supporting-information Table S1.

The main predicate-guided MORPHEUS run is reproduced from the repository root
with:

```sh
PYTHONPATH=$(printf '%s:' packages/*/src) python -m matrix semiexp \
  --xyzin packages/matrix-morpheus/examples/semiexp/norcamphor/xyzin \
  --outdir working/semiexp/norcamphor_table3_kraitchman \
  --backend python \
  --coordinate-model gic \
  --observable rotational_constants \
  --rotational-components ABC \
  --fix-hydrogens \
  --kraitchman-predicates \
  --kraitchman-partial-predicates \
  --kraitchman-distance-sigma 0.003 \
  --kraitchman-angle-sigma 0.3 \
  --max-iter 80
```

The current benchmark gives 48 generated GICs, 18 active least-squares
directions after hydrogen-frame constraints and Kraitchman-derived predicates,
a condition number of about `5.20e3`, and a rotational RMS residual of
`0.0941 MHz`.
