#!/usr/bin/env sh
set -eu

BASE='packages/matrix-morpheus/examples/semiexp/testosterone'
OUT='working/semiexp/testosterone_cc_class'

CC='CC_skeleton:shared:R(1,2)|R(1,6)|R(2,3)|R(2,45)|R(3,4)|R(3,10)|R(4,5)|R(4,9)|R(5,6)|R(5,7)|R(5,12)|R(7,8)|R(8,9)|R(10,11)|R(11,18)|R(15,16)|R(15,18)|R(16,39)|R(18,45)|R(39,42)|R(42,45)|R(45,46)'
NONCC='non_CC_stretches:fixed:R(1,34)|R(1,38)|R(2,21)|R(3,20)|R(4,19)|R(6,26)|R(6,27)|R(7,13)|R(7,28)|R(8,32)|R(8,33)|R(9,22)|R(9,23)|R(10,24)|R(10,25)|R(11,35)|R(11,36)|R(12,29)|R(12,30)|R(12,31)|R(13,14)|R(15,37)|R(16,17)|R(39,40)|R(39,41)|R(42,43)|R(42,44)|R(46,47)|R(46,48)|R(46,49)'

python -m matrix semiexp \
  --xyz "$BASE/testosterone_DPCS3.xyz" \
  --observations "$BASE/isotopologues.toml" \
  --xyzin "$OUT/xyzin" \
  --outdir "$OUT" \
  --backend python \
  --coordinate-model gic \
  --observable rotational_constants \
  --rotational-components ABC \
  --max-iter 80 \
  --parameter-class "$CC" \
  --parameter-class "$NONCC" \
  --parameter-class 'Angles_fixed:fixed:AAng' \
  --parameter-class 'Torsions_fixed:fixed:ATor' \
  --parameter-class 'OOP_fixed:fixed:AOop'
