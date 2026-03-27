Parent docs: [Project README](../../README.md)

# GRG Operator

This package contains `SpmvGRG`, artifact I/O, and the GRG-to-operator compile pipeline.

## Files

- [__init__.py](__init__.py): `SpmvGRG` construction, matmul orchestration, and operator-side memory snapshots
- [artifact.py](artifact.py): `.grg_spmv` artifact path derivation, save, and load
- [compile.py](compile.py): GRG compilation into `CompiledOperatorState`
- [sparse.py](sparse.py): binary CSR helpers

## Construction modes

`SpmvGRG` accepts:

- a `.grg` path
- a `.grg_spmv` path

For `.grg` input:

- derive an artifact path under `artifact_dir`
- load the artifact if it exists and is valid
- otherwise load the GRG, compile it, build init biases, save the artifact, and continue from the compiled state

For `.grg_spmv` input:

- load the artifact directly

## Artifact behavior

Artifact path derivation is implemented in [artifact.py](artifact.py).

- default artifact root is `./pygrgl_spmv_artifacts`
- derived paths encode the full source GRG path
- artifacts are stored as standalone `.grg_spmv` files

The artifact stores:

- structural arrays such as `level_offsets`, `node_perm`, and selector CSR parts
- mutation metadata tables
- retained block structure
- init-bias vectors and optional XTX init-bias vectors

## Compile pipeline

Compilation is implemented in [compile.py](compile.py).

`compile_grg(...)`:

- supports non-empty immutable GRGs only
- uses only down edges from the input GRG
- computes node heights and stable height order
- builds level-block CSR matrices directly from streamed child lists
- builds mutation and missingness selectors from sorted mutation rows; repeated `MutationID` rows are allowed when one mutation is attached to multiple nodes, and only missingness selector rows may need duplicate coalescing when those repeated rows share one missingness node
- loads mutation tables and optional coalescence counts

The result is a `CompiledOperatorState`.

## `CompiledOperatorState`

`CompiledOperatorState` holds:

- level structure and node-order mappings
- selector matrices
- mutation metadata tables
- optional coalescence counts
- sparse block structure
- cached init-bias arrays

`to_backend_setup(...)` converts the retained compile result into the normalized setup payload consumed by backends.

## Init biases

`SpmvGRG` precomputes init-bias arrays with the reference backend during artifact construction.

Cached biases:

- `init_vector_up_bias`
- `init_vector_down_bias`
- `init_xtx_up_bias`
- `init_xtx_down_bias`

These arrays are operator-owned retained memory and appear in `SpmvGRG.memory` as persistent `numpy` allocations with init roles.

## Runtime responsibilities

`SpmvGRG.matmul()` is responsible for:

- direction parsing
- input shape validation
- strict `numpy.ndarray` API boundaries for `input`, `miss`, and ndarray `init`
- init parsing and internal init-payload construction
- `by_individual` handling
- endpoint-output finishing and node-order restoration
- operator-owned call snapshot capture

Backends perform the actual traversal. `SpmvGRG` stores one retained snapshot on the operator and refreshes it only when backend-retained state changes.
Each successful `matmul(...)` overwrites `SpmvGRG.memory.last_call` with one call-local snapshot; benchmark history lives in the benchmark runner, not on the operator.
Operator call memory is local to one `matmul(...)` invocation, and backend call memory exists only inside the lexical backend capture scope used by `SpmvGRG`.

## Retained operator state after setup

After backend setup completes:

- `CompiledOperatorState.A_blocks` is dropped on the operator side
- the remaining operator-retained arrays stay in `self._compiled`
- those retained arrays are counted in `SpmvGRG.memory`
