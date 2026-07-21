# Succinic anhydride semiexperimental refinement

Data are transcribed from Jahn et al., Phys. Chem. Chem. Phys. 2020, 22, 5170-5177.

- `parent.xyz` is a C2v starting geometry reconstructed from the reported rSE parameters in Table 5.
- `isotopologues.toml` contains Table 1 B0 constants and rovibrational corrections inferred from the Table S3 semiexperimental equilibrium constants.
- `succinic_anhydride_predicates.mse.toml` reproduces the three-predicate model used in the paper for the unsubstituted hydrogen frame.
- `succinic_anhydride_four_predicates.mse.toml` adds the fourth C-C predicate reported in Table 5.
- `succinic_anhydride_fixed_h.mse.toml` is the comparison case with local hydrogen parameters fixed instead of represented by weighted predicates.
