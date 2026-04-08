Parent docs: [Project README](../../README.md)

# GRG Operator

This package contains `SpmvGRG`, artifact I/O, and the GRG-to-operator compile pipeline.

## Files

- [__init__.py](__init__.py): `SpmvGRG` construction, `convert()`, and operator-side memory snapshots
- [artifact.py](artifact.py): `.grg_spmv` path derivation, save, and load
- [compile.py](compile.py): GRG compilation into `CompiledOperatorState`
- [sparse.py](sparse.py): binary CSR helpers

## Construction modes

`SpmvGRG` accepts:

- a `.grg` path
- a `.grg_spmv` path
- a loaded `pygrgl.ImmutableGRG`

Behavior by source type:

- `.grg`: derive an artifact path under `artifact_dir`, load it if valid, otherwise compile and save a new `.grg_spmv`
- `.grg_spmv`: load the artifact directly
- loaded GRG object: compile in memory without assigning an artifact path

For path inputs, generic `os.PathLike` objects are accepted.

## `convert()`

`convert()` exposes the compile pipeline without constructing a backend.

- `convert("/path/to/file.grg")` compiles in memory
- `convert("/path/to/file.grg", output_dir="artifacts")` also saves `file.grg_spmv`
- `convert(grg_obj)` compiles a loaded immutable GRG in memory
- `convert(grg_obj, output_dir=..., name=...)` saves an artifact for an in-memory GRG object

The result is a `CompiledOperatorState` with init-bias arrays already populated.

## Artifact behavior

Artifact path derivation is implemented in [artifact.py](artifact.py).

- default artifact root: `./pygrgl_spmv_artifacts`
- derived paths encode the full source GRG path to avoid collisions
- artifacts are standalone `.grg_spmv` files
- artifact load preserves stored structural dtypes and logs them at INFO level

## Compile pipeline

Compilation is implemented in [compile.py](compile.py).

`compile_grg(...)`:

- supports immutable, non-empty GRGs only
- uses down edges only
- computes stable height order and level-block CSR structure
- builds mutation and missingness selectors from sorted mutation rows
- loads mutation tables and optional coalescence counts
- emits INFO-level RSS checkpoints during cache-miss `.grg` builds

## Init biases

`convert()` and cache-miss `.grg` construction precompute init-bias arrays with the reference backend.

Cached biases:

- `init_vector_up_bias`
- `init_vector_down_bias`
- `init_xtx_up_bias`
- `init_xtx_down_bias`

These arrays are retained on the operator and appear in `SpmvGRG.memory`.

## Runtime responsibilities

`SpmvGRG.matmul()` handles:

- direction parsing
- input validation
- init parsing and internal init-payload construction
- `by_individual` handling
- endpoint output finishing and node-order restoration
- operator-owned memory snapshot capture

Backends perform the traversal itself. After backend setup, `CompiledOperatorState.A_blocks` is dropped on the operator side and the remaining retained arrays are tracked in `SpmvGRG.memory`.
