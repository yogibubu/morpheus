# Two-dimensional water PES example

This is the first complete chemical LINK-SENTINEL v1 example. LINK constructs
an unsymmetrized SONIC model for water and exposes two active variables: one
O-H stretch and the H-O-H bend. The deterministic SENTINEL mock requests a 2x2
batch. LINK realizes all four Cartesian structures concurrently and evaluates
E, Cartesian G and Cartesian H on an inexpensive positive quadratic surface.

From the MATRIX repository root:

```bash
python packages/matrix-trinity/examples/link_sentinel/water_2d/run_example.py \
  --run-dir /tmp/matrix-water-link-sentinel
```

The calculation is intentionally cheap and analytically reproducible. It tests
the protocol and coordinate machinery; it is not presented as an accurate
water potential.
