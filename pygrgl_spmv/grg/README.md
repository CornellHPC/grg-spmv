Parent docs: [Project README](../../README.md)

# GRG Artifacts

`pygrgl_spmv/grg/` contains:

- `convert(...) -> Path`
- the `.grg_spmv` save/load/scan helpers in [artifact.py](artifact.py)
- the compile pipeline in [compile.py](compile.py)
- the internal `BoundGRG` host-side API logic in [__init__.py](__init__.py)

`.grg_spmv` v6 artifacts are uncompressed and store scan metadata directly for fast metadata reads.
