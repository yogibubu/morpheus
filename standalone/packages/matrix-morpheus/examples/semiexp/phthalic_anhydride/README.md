# Phthalic anhydride semiexperimental refinement

Data are transcribed from Belyakov, Vogt, Demaison, Kulishenko and Oskorbin,
Chem. Phys. Lett. 795 (2022) 139540.

- `parent.xyz` is a planar C2v starting geometry reconstructed from the
  reported rse parameters in Table 1.
- `isotopologues.toml` contains Table 2 B0 constants, the B2PLYP/VTZ
  rovibrational corrections scaled by 1.11 as described in the text, and the
  tabulated electronic corrections.
- `phthalic_anhydride_predicates.mse.toml` runs the fit with reBO predicates
  from Table 1 using the conservative uncertainties stated in the paper:
  0.002 A for bond lengths and 0.2 degree for bond angles.
