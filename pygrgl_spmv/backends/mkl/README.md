Parent docs: [Backend Docs](../README.md)

# MKL Backend

The MKL backend provides host-only sparse traversal using Intel MKL sparse handles.

## Public API

Exports from [__init__.py](__init__.py):

- `MklBackend`
- `MklPlan`
- `MklPlanPair`

## Plans

`MklPlan` is defined in [plan.py](plan.py).

Fields:

- `store`: whether the backend stores `A` or `A.T`
- `fmt`: sparse format for stored blocks
- `n_threads`: thread count hint for MKL runtime calls
- `k_hint`: optional positive integer or `none`

`MklPlanPair` holds `plan_up` and `plan_down`. Either side may be omitted.

Storage sharing follows `MklPlan.can_share_storage_with(...)`:

- identical store/format pairs share storage
- transpose-compatible format pairs also share storage

## Runtime behavior

`MklBackend` is implemented in [backend.py](backend.py).

Setup:

- converts GRG blocks into persistent MKL sparse handles
- builds selector row/col index arrays
- configures MKL sparse-handle hints
- records per-level call-count and nnz statistics for debug logging
- borrows the operator `sample_rows` mapping directly

Execution:

- allocates a dense `node_values` array on host
- applies init state
- scatters sample or mutation inputs into node state
- propagates values level by level using MKL `mv` or `mm`
- gathers endpoint outputs on host

Threading:

- setup uses the maximum configured directional thread count
- each direction runs with its own effective MKL thread count
- `n_threads=0` resolves to `os.cpu_count()`

## Instrumentation

`instrumentation=True` keeps the normal execution path and adds CPU-side wavefront timing when debug logging is enabled.

The backend logs:

- total wavefront time
- per-level milliseconds
- per-level call count and nnz

## Memory ledger mapping

The MKL backend contributes only `numpy` allocations.

Retained backend allocations include:

- sparse-handle payload arrays for `blocks_up`
- sparse-handle payload arrays for `blocks_down` when stored separately
- selector row/col caches for mutation and missingness selectors
- `_xtx_host` when coalescence counts are present

Call-time backend allocations include:

- `node_values`
- `miss_output` for UP traversals with miss output enabled
- `level_ms` when instrumentation timing is allocated
