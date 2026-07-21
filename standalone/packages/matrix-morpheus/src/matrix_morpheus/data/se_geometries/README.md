# Semiexperimental Geometry Library

This directory contains semiexperimental accurate equilibrium geometries copied
from `/Users/vincenzobarone/Desktop/SE`.

- `xyz/`: original XYZ files, preserving the source filenames.
- `manifest.csv`: stable index with slug, molecule name, atom count, level, and
  relative XYZ path.

These geometries are intended as the local accurate-reference library for
multistructure, family-based transferable geometry corrections.

Search the library with:

```bash
oracle multistructure-reference-search \
  --query-xyz path/to/query.xyz \
  --outdir working/multistructure_reference_search/query
```

The command writes a ranked CSV/JSON match list and an ORACLE run manifest.

To construct a query geometry from the most similar supported local fragments,
then summarize transferable classes for the multistructure layer:

```bash
oracle multistructure-build-reference-geometry \
  --query-xyz path/to/query.xyz \
  --outdir working/multistructure_reference_geometry/query
```

Only fragments supported by close reference-library matches are transferred.
Unsupported fragments are left at the query value.  Fragment classes are split
by primitive type, atom signature, synthon environment, and value cluster so
chemically distinct cases such as C=O and C-O are not collapsed.
