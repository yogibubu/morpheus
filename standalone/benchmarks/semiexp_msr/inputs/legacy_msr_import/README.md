# Legacy MSR Import Benchmarks

These inputs are retained as compatibility fixtures for the Merlino MSR legacy
reader. They are not meant to reproduce the old MSR constraint logic directly.

BSR imports the Cartesian/Z-matrix geometry, isotopologue definitions,
rotational constants, vibrational/electronic corrections, and QM predicates.
Legacy MSR `constraints:` records in Z-matrix style are preserved as diagnostic
metadata only. Production constraints and parameter classes must be generated or
specified through the current BSR/GICForge model.

Baseline fits with the current GICForge definitions are:

| case | legacy constraints | rank | RMS / MHz | weighted RMS | stationary point |
|---|---:|---:|---:|---:|---|
| glycidol_conf00 | 0 | 12 | 0.00115755 | 0.0454248 | minimum |
| glycolaldehyde | 0 | 12 | 0.00406413 | 0.303441 | minimum |
| o-EBN | 15 | 15 | 0.00302850 | 0.0144200 | minimum |
| p-EBN | 12 | 6 | 0.00587314 | 0.141744 | minimum |

The class-advisor suggests shared parameter classes for all four cases. These
suggestions are diagnostic unless explicitly passed to a BSR request.
