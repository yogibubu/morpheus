# MORPHEUS 0.1.0rc8 standalone source snapshot

This directory is a revision-pinned extraction of the components required to
build and run MORPHEUS without the private MATRIX development repository.

Source revision: `5352e8dd924ec04ba1a1f48563f521eaeaf8f574`

## Build a self-contained release

From this directory, run:

```bash
python -m pip install build
python tools/build/build_morpheus_release.py /chosen/path/MORPHEUS-0.1.0rc8
python tools/audit/verify_morpheus_install.py /chosen/path/MORPHEUS-0.1.0rc8
```

The builder creates a local wheelhouse, copies the runnable examples and
documentation, and writes a SHA-256 manifest recording every artifact and the
source revision.

## Install directly from the source snapshot

For development or inspection, install the packages into a fresh Python 3.11
environment in the order used by the release builder:

```bash
for package in matrix-core matrix-chem matrix-switch matrix-apoc matrix-link \
  matrix-numerics matrix-zaff matrix-fragments matrix-qm matrix-rovib \
  matrix-engines matrix-gaussian matrix-smith matrix-gf matrix-oracle \
  matrix-trinity matrix-morpheus; do
  python -m pip install "./packages/${package}"
done
```

Then check the installation and copy the bundled examples:

```bash
morpheus doctor
morpheus examples /chosen/path/morpheus-examples
```

The full command-line example and output inventory are documented in
`docs/MORPHEUS_QUICKSTART.md` and `docs/manuals/morpheus_manual.pdf`.

## Publication data

Runnable semiexperimental inputs are stored in
`packages/matrix-morpheus/examples/semiexp/`. The `benchmarks/semiexp_msr/`
directory contains the legacy-import inputs and the paper regression record.
Additional camphor audit tables and accepted coordinates are in the repository
root under `data/camphor/`.

## Scope

This snapshot includes the source needed by MORPHEUS: CORE, CHEM, SWITCH,
APOC, LINK, NUMERICS, ZAFF, FRAGMENTS, QM, ROVIB, ENGINES, GAUSSIAN, SMITH,
GF, ORACLE, TRINITY, and MORPHEUS.
Unrelated MATRIX applications and private development material are excluded.
