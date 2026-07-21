#!/usr/bin/env sh
set -eu

BASE='packages/matrix-morpheus/examples/semiexp/testosterone'
OUT='working/semiexp/testosterone_cc_co_classes'

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
  --primitive-class 'CC_skeleton:R(1,2)|R(1,6)|R(2,3)|R(2,45)|R(3,4)|R(3,10)|R(4,5)|R(4,9)|R(5,6)|R(5,7)|R(5,12)|R(7,8)|R(8,9)|R(10,11)|R(11,18)|R(15,16)|R(15,18)|R(16,39)|R(18,45)|R(39,42)|R(42,45)|R(45,46)' \
  --primitive-class 'CO_stretches:R(7,13)|R(16,17)' \
  --primitive-class-min 0.70 \
  --primitive-class-cross-max 0.20 \
  --primitive-class-budget auto
