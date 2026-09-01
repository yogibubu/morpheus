# Cyclopentadiene benchmark input

The rotational constants and corrections are transcribed from the legacy MSR
fixture. `isotopologues.toml` states the additive MSR correction convention
explicitly, so the corrected constants are `B_e = B_0 + DeltaBvib` after a
round trip through the canonical `xyzin` container.

The manuscript regression uses 14 records and omits `iso04_313`, the second of
the exactly duplicated C2/C3 carbon-13 records. The corrected first carbon-13
constant is 8226.053 MHz before adding its 66.046 MHz correction; the
historical 8426.053 transcription is not used.
