"""
thermo_writer.py
================

Writer for thermodynamic functions.

Writes:
- translational, rotational, vibrational contributions
- total thermodynamic functions
"""

# ============================================================
# Helpers
# ============================================================
def _write_keyval(f, key, value):
    if value is not None:
        f.write(f"{key} = {value}\n")


def _ordered_items(block):
    if block is None:
        return []

    preferred = (
        "Q_dimless",
        "U_kJmol",
        "H_kJmol",
        "S_JmolK",
        "Cv_JmolK",
        "Cp_JmolK",
    )
    out = []
    seen = set()

    for k in preferred:
        if k in block:
            out.append((k, block.get(k)))
            seen.add(k)

    for k in sorted(block.keys()):
        if k in seen:
            continue
        out.append((k, block.get(k)))

    return out


def _strip_thermo_block(lines):
    out = []
    in_block = False
    for line in lines:
        s = line.strip()
        if s.upper() == "#THERMO":
            in_block = True
            continue
        if in_block and s.startswith("#"):
            in_block = False
        if not in_block:
            out.append(line)
    return out


# ============================================================
# Public API
# ============================================================
def write_thermo(xyzin_path, thermo):
    """
    Write full thermodynamic block.
    """

    with open(xyzin_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    lines = _strip_thermo_block(lines)

    with open(xyzin_path, "w", encoding="utf-8") as f:
        for l in lines:
            f.write(l if l.endswith("\n") else l + "\n")

        f.write("\n#THERMO\n")
        f.write("# keys: Q_dimless U_kJmol H_kJmol S_JmolK Cv_JmolK Cp_JmolK\n")

        for label in ("trasl", "rot", "vib", "tot"):
            block = thermo.get(label)
            if block is None:
                continue

            for key, val in _ordered_items(block):
                _write_keyval(f, f"{key}_{label}", val)
