#!/usr/bin/env sh
set -eu

BASE='packages/matrix-morpheus/examples/semiexp/testosterone'
OUT='working/semiexp/testosterone_predicates'
BASIS="$OUT/basis"
XY_SIGMA="${MATRIX_TESTOSTERONE_XY_PREDICATE_SIGMA:-0.005}"

python -m matrix semiexp \
  --xyz "$BASE/testosterone_DPCS3.xyz" \
  --observations "$BASE/isotopologues.toml" \
  --xyzin "$BASIS/xyzin" \
  --outdir "$BASIS" \
  --backend python \
  --coordinate-model gic \
  --observable rotational_constants \
  --rotational-components ABC \
  --max-iter 0

FIXED=$(python - "$BASIS/semiexp_parameters.csv" <<'PY'
from __future__ import annotations

import csv
import re
import sys

params = sys.argv[1]
cc = {
    "R(1,2)", "R(1,6)", "R(2,3)", "R(2,45)", "R(3,4)", "R(3,10)",
    "R(4,5)", "R(4,9)", "R(5,6)", "R(5,7)", "R(5,12)", "R(7,8)",
    "R(8,9)", "R(10,11)", "R(11,18)", "R(15,16)", "R(15,18)",
    "R(16,39)", "R(18,45)", "R(39,42)", "R(42,45)", "R(45,46)",
}
noncc = {
    "R(1,34)", "R(1,38)", "R(2,21)", "R(3,20)", "R(4,19)", "R(6,26)",
    "R(6,27)", "R(7,13)", "R(7,28)", "R(8,32)", "R(8,33)", "R(9,22)",
    "R(9,23)", "R(10,24)", "R(10,25)", "R(11,35)", "R(11,36)",
    "R(12,29)", "R(12,30)", "R(12,31)", "R(13,14)", "R(15,37)",
    "R(16,17)", "R(39,40)", "R(39,41)", "R(42,43)", "R(42,44)",
    "R(46,47)", "R(46,48)", "R(46,49)",
}
co = {"R(7,13)", "R(16,17)"}


def compact(label: str) -> str:
    return re.sub(r"R\(\s*(\d+),\s*(\d+)\)", r"R(\1,\2)", label)


def gic_id(label: str) -> str:
    return label.split(None, 1)[0]


def coeffs(label: str) -> dict[str, float]:
    body = compact(label)
    return {
        f"R({left},{right})": abs(float(value))
        for value, left, right in re.findall(
            r"([+-]?\d+\.\d+)\*R\(\s*(\d+),\s*(\d+)\)", body
        )
    }


selected: set[str] = set()
all_ids: list[str] = []
with open(params, newline="", encoding="utf-8") as handle:
    for row in csv.DictReader(handle):
        label = row["name"]
        gid = gic_id(label)
        all_ids.append(gid)
        normalized = compact(label)
        is_cc_baseline = any(item in normalized for item in cc) and not any(
            item in normalized for item in noncc
        )
        has_dominant_co = any(value >= 0.70 for key, value in coeffs(label).items() if key in co)
        if is_cc_baseline or has_dominant_co:
            selected.add(gid)

fixed = [gid for gid in all_ids if gid not in selected]
print(";".join(fixed))
PY
)

set -- python -m matrix semiexp \
  --xyz "$BASE/testosterone_DPCS3.xyz" \
  --observations "$BASE/isotopologues.toml" \
  --xyzin "$OUT/xyzin" \
  --outdir "$OUT" \
  --backend python \
  --coordinate-model gic \
  --observable rotational_constants \
  --rotational-components ABC \
  --max-iter 80 \
  --fixed "$FIXED"

while IFS= read -r predicate; do
  set -- "$@" --qm-predicate "$predicate"
done <<PREDICATES
$(python - "$BASE/testosterone_DPCS3.xyz" "$XY_SIGMA" <<'PY'
from __future__ import annotations

import math
import sys

xyz = sys.argv[1]
sigma = float(sys.argv[2])
pairs = (
    (1, 2), (1, 6), (2, 3), (2, 45), (3, 4), (3, 10), (4, 5), (4, 9),
    (5, 6), (5, 7), (5, 12), (7, 8), (8, 9), (10, 11), (11, 18),
    (15, 16), (15, 18), (16, 39), (18, 45), (39, 42), (42, 45), (45, 46),
    (7, 13), (16, 17),
)
coords: list[tuple[float, float, float]] = []
with open(xyz, encoding="utf-8") as handle:
    lines = handle.readlines()[2:]
for line in lines:
    parts = line.split()
    if len(parts) >= 4:
        coords.append(tuple(float(item) for item in parts[1:4]))
for left, right in pairs:
    a = coords[left - 1]
    b = coords[right - 1]
    distance = math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(3)))
    print(f"R({left},{right}):{distance:.12f}:{sigma:.6g}:QC_geometry")
PY
)
PREDICATES

"$@"
