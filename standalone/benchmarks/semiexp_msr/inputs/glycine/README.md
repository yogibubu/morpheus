# Glycine MSR Inputs

This directory contains MSR-compatible glycine I and II convergence tests
assembled from the glycine microwave and structure papers available locally.

## `glycine_I_cs_table5_constraints.msr`

This is the compact `Cs` counterpart to the glycine II functional-constraint
benchmark. The Cartesian atom order is:

1. methylene `C`
2. amino `N`
3. amino `H`
4. carboxyl `C`
5. carbonyl `O`
6. hydroxyl `O`
7. hydroxyl `H`
8. methylene `H`
9. methylene `H`
10. amino `H`

The five-isotopologue data set contains the parent, `13C_carboxyl`,
`13C_methylene`, `CD2`, and `15N` species. Ground-state rotational constants
and statistical uncertainties are the Table 1 values of Godfrey and Brown,
J. Am. Chem. Soc. 117, 2019-2023 (1995). The `weights` block stores inverse
variances, `1/sigma(B0)^2`; the `dbvib` rows are the Gly-Ip zero-point
corrections of Kasalova et al. The four constraints are the Gly-Ip Table 5
structural relations:

```text
ROH(Frozen,Value=0.9660)=R(6,7)
ACOH(Frozen,Value=106.04)=A(4,6,7)
HNH(Frozen,Value=104.98)=A(3,2,10)
NH2WAG(Frozen,Value=57.67)=U(1,2,3,10)
```

Run:

```bash
python -m merlino semiexp \
  --job benchmarks/semiexp_msr/inputs/glycine/glycine_I_cs_table5_constraints.msr \
  --outdir working/semiexp/glycine_I_cs_table5_gic_fortran \
  --backend fortran77 \
  --coordinate-model gic \
  --observable moments \
  --rotational-components auto \
  --max-iter 120 \
  --damping 1.0e-6 \
  --max-step 0.05
```

Reference result:

- GIC convergence: `step_tolerance` after 6 accepted steps and 5 rejected
  trial steps, with a positive stationary-point check (`minimum`).
- Final/active GIC count: 24/15; four functional constraints leave 11
  independent fitted variables.
- Condition number: about `1.02e4`.
- Rotational RMS residual: `0.027095 MHz`; maximum absolute rotational
  residual: `0.056533 MHz`.
- Experimental least-squares weights are used directly from the Table 1
  standard deviations of Godfrey and Brown.

## `glycine_II_dpcs3_table5_constraints.msr`

This is the reproducible SEfit benchmark used in the manuscript. It keeps the
non-`Cs` DPCS3 Cartesian atom order:

1. carboxyl `C`
2. hydroxyl `O`
3. carbonyl `O`
4. hydroxyl `H`
5. methylene `C`
6. amino `N`
7. methylene `H`
8. methylene `H`
9. amino `H`
10. amino `H`

The rotational constants are the 10-isotopologue Gly-IIn set and the `dbvib`
rows are the Gly-IIn zero-point corrections. The five reduced-dimensionality
relations follow the Kasalova et al. Table 5 functional form, but their
`Value=` targets are evaluated from the DPCS3 geometry. The constraints are
written as reusable Gaussian-style coordinate definitions, for example:

```text
RCH7=R(5,7)
RCH8=R(5,8)
DRCH(Frozen,Value=0.000091749296)=RCH8-RCH7
```

Run:

```bash
python -m merlino semiexp \
  --job benchmarks/semiexp_msr/inputs/glycine/glycine_II_dpcs3_table5_constraints.msr \
  --outdir working/semiexp/glycine_II_dpcs3_table5_repo_gic_fortran \
  --backend fortran77 \
  --coordinate-model gic \
  --observable moments \
  --rotational-components auto \
  --max-iter 120 \
  --damping 1.0e-6 \
  --max-step 0.05
```

Reference result:

- GIC convergence: `line_search_stalled` after 46 iterations, with a positive
  stationary-point check (`minimum`).
- Effective optimized parameters and numerical rank: 19.
- Condition number: about `1.05e4`.
- Rotational RMS residual: `0.249116 MHz`; maximum absolute rotational
  residual: `0.670672 MHz`.
- The corresponding run using the published Kasalova-type constraint targets
  has RMS `0.234383 MHz`, showing that the DPCS3 targets are close to the
  coupled-cluster structural information for this purpose.

## `glycine_II_msr.msr`

Conformer II / `gly(2)` uses the isotopologue atom numbering of McGlone et al.,
J. Mol. Struct. 485-486, 225-238 (1999), Table 10, but the Cartesian
coordinates are the `Cs` DPCS3 geometry supplied by the user. The geometry was
reordered onto the McGlone numbering:

1. `H1`, hydroxyl hydrogen
2. `O2`, hydroxyl oxygen
3. `C3`, carboxyl carbon
4. `O4`, carbonyl oxygen
5. `C5`, methylene carbon
6. `H6`, methylene hydrogen
7. `H6'`, methylene hydrogen
8. `N7`, amino nitrogen
9. `H8`, amino hydrogen
10. `H8'`, amino hydrogen

The original DPCS3 atom-order mapping is:

- MSR atom 1 <- DPCS3 atom 4
- MSR atom 2 <- DPCS3 atom 2
- MSR atom 3 <- DPCS3 atom 1
- MSR atom 4 <- DPCS3 atom 3
- MSR atom 5 <- DPCS3 atom 5
- MSR atom 6 <- DPCS3 atom 7
- MSR atom 7 <- DPCS3 atom 8
- MSR atom 8 <- DPCS3 atom 6
- MSR atom 9 <- DPCS3 atom 9
- MSR atom 10 <- DPCS3 atom 10

The rotational constants and statistical uncertainties are from McGlone et al.,
Table 11. The parent, `15N`, `13C`, `CD2`, and `OD_ND2` entries are quoted
there from Godfrey and Brown, J. Am. Chem. Soc. 117, 2019-2023 (1995). The
parent normal-species constants are close to, but not identical with, the later
Lovas et al. Astrophys. J. 455, L201-L204 (1995) constants. The `weights`
section stores inverse variances, `1/sigma(B0)^2`, as expected by Merlino.

The `dbvib` rows are the zero-point corrections `A_e-A_0`, `B_e-B_0`,
`C_e-C_0` from the `Gly-IIp` column of Table 9 supplied in the development
notes. The legacy MSR parser treats `dbvib` as additive, so the corrected
constants used by the fit are `B_e = B_0 + dbvib`. In the Kasalova numbering
`C(1)` is the methylene carbon and `C(3)` is the carboxyl carbon; the Merlino
labels `C13_methylene` and `C13_carboxyl` follow that mapping.

The input uses the `Cs` model from the starting geometry and GICForge symmetry
adaptation: the two `CH2` hydrogens and the two `NH2` hydrogens are equivalent
by symmetry. Only the two carbon-bound C-H distances are fixed to their DPCS3
values; no C-H angles, C-H torsions, backbone dihedrals, OH parameter, or NH
parameter is frozen.

## Reference Run

Use a slightly conservative initial damping and step cap. With these settings
both the Python and Fortran77 backends should converge cleanly.

```bash
python -m merlino semiexp \
  --job benchmarks/semiexp_msr/inputs/glycine/glycine_II_msr.msr \
  --outdir working/semiexp/glycine_II_cs_chdist_python \
  --coordinate-model gic \
  --observable moments \
  --rotational-components auto \
  --max-iter 120 \
  --damping 1.0e-6 \
  --max-step 0.05
```

```bash
python -m merlino semiexp \
  --job benchmarks/semiexp_msr/inputs/glycine/glycine_II_msr.msr \
  --outdir working/semiexp/glycine_II_cs_chdist_fortran \
  --backend fortran77 \
  --coordinate-model gic \
  --observable moments \
  --rotational-components auto \
  --max-iter 120 \
  --damping 1.0e-6 \
  --max-step 0.05
```

Current diagnostic result:

- GIC convergence: `line_search_stalled` after 52 iterations
- Symmetry-Cartesian convergence: `line_search_stalled` after 58 iterations
- Effective optimized parameters: 14
- Numerical rank: 14
- Primitive constraints: the two local carbon-bound C-H distance constraints,
  expanded by symmetry and projected after each accepted step
- Condition number: about 6.20e3 for GIC and 4.62e3 for symmetry-Cartesian
- Rotational RMS residual: about 1.68 MHz with the `Gly-IIp` corrections,
  experimental weights, and this static Cs model
- Stationary-point diagnostic: minimum

Merlino supports exact primitive constraints and exact linear primitive
constraints using Gaussian GIC keyword syntax, for example:

```text
DRCH(Frozen,Value=-0.000189)=R[5,7]-R[5,6]
CH2ROCK(Frozen,Value=0.1168)=A(7,5,8)+A(7,5,3)-A(6,5,8)-A(6,5,3)
```

Bond targets are in Angstrom. Angular targets are in degrees unless the suffix
`rad` is given. The command-line and TOML readers split these constraints only
at top-level separators, so commas inside primitive definitions are preserved.
More general Gaussian-style expression constraints are also accepted:

```text
QFIX=[GIC001+2*GIC002] Value=0.0
DRCH(Frozen,Value=-0.000189)=R[5,7]-R[5,6]
SINA(Frozen,Value=0.0)=sin(A(7,5,8))-sin(A(6,5,8))
CH2ROCK(Frozen,Value=0.1168)=A(7,5,8)+A(7,5,3)-A(6,5,8)-A(6,5,3)
```

The bracket form with a bare `F` freezes the initial value. Tabulated
constraints, including the Kasalova Table 5 values, must use `Value=` so that
the constrained target is independent of the starting Cartesian geometry.
`GIC###` names refer to the final GICForge coordinates, while `R/A/D/U/L` refer
to Gaussian-style primitive coordinates with one-based atom indexes.

This run is an SE-style diagnostic with perturbative vibrational corrections
and fixed DPCS3 C-H distances. It should not be compared directly with the
published `r0(Fit B)` structural uncertainties.

## `glycine_II_r0_fitB_msr.msr`

This benchmark mirrors the Kasalova et al. `r0(Fit B)` comparison:

- `dbvib` is zero, so the fit uses ground-state rotational constants.
- `[OH,NDH]` and `[OD,NDH]` are excluded, following the paper.
- No C-H or other internal-coordinate constraints are imposed.
- The same inverse-variance experimental weights are used.

Reference command:

```bash
python -m merlino semiexp \
  --job benchmarks/semiexp_msr/inputs/glycine/glycine_II_r0_fitB_msr.msr \
  --outdir working/semiexp/glycine_II_r0_fitB_fortran \
  --backend fortran77 \
  --coordinate-model gic \
  --observable moments \
  --rotational-components auto \
  --max-iter 160 \
  --damping 1.0e-6 \
  --max-step 0.05
```

Reference result:

- GIC convergence: `line_search_stalled` after 53 iterations
- Effective optimized parameters and numerical rank: 15
- Condition number: about 5.49e3
- Rotational RMS residual: about 1.38 MHz
- Propagated standard errors match the published `r0(Fit B)` values closely
  for the directly comparable primitive parameters; for example `r(C-C)`,
  `r(C-N)`, `r(C=O)`, `r(C-O)`, `r(O-H)`, `r(C-H av)`, and `r(N-H av)` are
  within about 0.001-0.002 Angstrom in standard error, and the main valence
  angles are within about 0.1 degrees.
