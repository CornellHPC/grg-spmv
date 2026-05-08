Parent docs: [Project README](../../README.md)

# Tests

## Run Modes

- Default suite:
  - `uv run pytest -q pygrgl_spmv/tests --backend all`
- Backend-focused runs:
  - `uv run pytest -q pygrgl_spmv/tests --backend mkl`
  - `uv run pytest -q pygrgl_spmv/tests --backend cusparse`
  - `uv run pytest -q pygrgl_spmv/tests --backend triton`
- Long streamed-GPU suite:
  - `uv run pytest -q pygrgl_spmv/tests --backend all --stress`

Default backend test files already include small-artifact streamed GPU mirrors for planner transitions, exactness, and overlap contracts.
`--stress` is the long run. It keeps the large streamed Triton/cuSPARSE band cases and can take around 30 minutes.
Triton autotuning is disabled in tests via monkeypatch so correctness runs stay fast.
Streamed GPU tests skip only for insufficient host RAM while constructing the synthetic large artifacts; VRAM-budget failures are treated as correctness/accounting bugs.

## Shared Contracts

The suite protects five things:

- compile and artifact correctness
- `BoundGRG` host-side API semantics
- planner byte/layout correctness
- entered-runtime ownership and lifecycle invariants
- backend numerical parity and end-to-end explicit-matrix equivalence

## File Map

- `test_compile_selectors.py`
  - Compile-time selector and stable-height block construction edge cases.
- `test_sparse_utils.py`
  - Canonical sparse helper invariants for structural dtype finalization and binary CSR rehydration.
- `test_import_surface.py`
  - CPU-safe package import surface under blocked optional GPU modules.
- `test_bench_scripts.py`
  - Minimal benchmark runner loop counts and reporting contract.
- `test_bolt_lmm_inf.py`
  - CPU Boost RNG and phenotype-simulator determinism guards plus GPU/cuSPARSE 1000 Genomes chr19-22 covariate-aware BOLT-LMM-inf smoke against official BOLT v2.5; parses stdout and `summary.json`, requires 128 matched/model SNPs with generated `PC1`-`PC20` and `SEX` covariates, and enforces strict comparison thresholds in the test.
- `runtime/test_convert.py`
  - `convert()` path/object behavior, artifact writing, init-bias persistence, and down-edge-only compilation.
- `runtime/test_artifacts.py`
  - `.grg_spmv` scan/load/block iteration correctness and direct artifact consumption by runtimes.
- `runtime/test_api.py`
  - Entered-runtime lifecycle, `runtime.grgs`, concurrency guard, fixed owned-buffer reuse, and one-runtime multi-GRG usage.
- `runtime/test_matmul_semantics.py`
  - Shared `BoundGRG.matmul()` semantics on `ReferenceRuntime`: `by_individual`, init, miss, dtype, and fast-fail contract checks.
- `runtime/test_prepare_cuda.py`
  - Direct `prepare_matmul_cuda()` contract checks: CPU rejection, GPU buffer semantics, and prepared-path correctness for init, missingness, and node emission.
- `runtime/test_emit_all_nodes.py`
  - `emit_all_nodes=True` semantics, including init modes, dtype variants, stability, and miss rejection.
- `runtime/test_traversal_correctness.py`
  - Reference traversal correctness, stable node ordering, exact binary cases, dtype coverage, and zero/stability behavior.
- `runtime/test_reference.py`
  - Reference runtime parity and exact byte accounting for resident sparse blocks, selectors, and workspaces.
- `runtime/test_mkl.py`
  - MKL runtime parity, format/thread-local behavior, separate-runtime threaded execution, LP64 ABI checks, `float32` dispatch, shared-value budgeting, and optimize-flag semantics.
- `runtime/test_plans.py`
  - CUDA device/stream parsing plus backend plan/pair validation and storage-sharing rules.
- `runtime/test_directional_layouts.py`
  - One-sided planner/runtime behavior across backends, including disabled-direction failures, reduced owned bytes, and runtime `k <= max_k` coverage.
- `runtime/test_triton.py`
  - Triton planner/runtime specifics: multi-column dense execution, three-block streamed transition matrix, streamed exactness, overlap, scratch correctness, and CUDA device/stream execution behavior.
- `runtime/test_cusparse.py`
  - cuSPARSE planner/runtime specifics: dense-view plans, shared-ones modes, three-block streamed transition matrix, streamed exactness, overlap, slot compaction, scratch correctness, and CUDA device/stream execution behavior.
- `runtime/_runtime_builders.py`
  - Shared layout-builder helpers used by the runtime-era suite.
- `runtime/_streaming_cases.py`
  - Synthetic streamed-artifact builders, three-block streamed mode tables, and analytic expected-value helpers for GPU streamed tests.
- `endtoend/conftest.py`
  - Explicit genotype-matrix and missingness helper functions used by end-to-end tests.
- `endtoend/test_matmul.py`
  - End-to-end equivalence against `pygrgl.dot_product`, explicit genotype matrices, diploid semantics, init modes, and GRG splitting.
- `endtoend/test_missing.py`
  - End-to-end missingness counts, mean-imputation semantics, and shared-site missingness equality.

## Maintenance Rule

If a new test file is added, add one line here describing the correctness contract it protects.
