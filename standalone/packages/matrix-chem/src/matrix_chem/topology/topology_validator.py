"""
Topology validator for ORACLE.

Guarantees:
- All hydrogens are bound exactly once
- No isolated atoms
"""


def validate_topology(dg):
    errors = []

    for i, Z in enumerate(dg.Z):
        deg = len(dg.adjacency[i])

        if Z == 1:
            if deg != 1:
                errors.append(f"Hydrogen {i} has degree {deg}")
        else:
            if deg == 0:
                errors.append(f"Atom {i} (Z={Z}) is isolated")

    if errors:
        raise ValueError("Topology validation failed:\n" + "\n".join(errors))
