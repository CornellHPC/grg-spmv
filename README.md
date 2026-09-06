# pygrgl-spmv

`pygrgl-spmv` is the core of MIKADO's formulation, which enables lightning-fast and parallel-friendly GRG operations on devices including CPUs and GPUs.


## Hardware requirements

Currently, the following hardware is supported:
- CPU: Any x86 CPU is supported, though Intel MKL is not officially supported on AMD CPUs.
- GPU: Nvidia GPUs with CUDA Toolkit Version >= 13.3.1 **is required**. Earlier versions do not gurantee correctness. In case where you cannot install a newer driver or cuda runtime, we recommend you relying on [CUDA Forward Compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/forward-compatibility.html).

## Install

Requires Python >= 3.10. To install, use:

```bash
pip install .          # install the basic dependencies; can be used with the MKL-based CPU backend
pip install '.[gpu]'   # adds support for the cuSPARSE-based GPU backend
```

To use the MKL-based CPU backend, you need to install MKL manually. We recommend using the conda package manager for this.
```bash
conda install -c conda-forge mkl mkl-devel mkl-static mkl-include
```
You can also install following Intel's official [instructions](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl-download.html).

## Basic Usage

To use `pygrgl-spmv` with `grapp` and the supported applications (GWAS, PCA, BOLT-LMM), some adaptor functions have been provided, so that users don't need to touch the lower-level APIs.

### Converting

To obtain a `.grg` file from formats such as `.vcf.gz`, please refer to the [grgl docs](https://grgl.readthedocs.io/en/stable/) for instructions.

Currently, the `.grg` file needs to go through a simple conversion step to generate a `.grg_spmv` artifact, which is the format that `pygrgl-spmv` can consume. This can be done using the `simple_convert` function. An example: 

```python
from pygrgl_spmv import simple_convert

artifact = simple_convert("chr1.grg", "artifacts/chr1.grg_spmv")
```

### Running

The adaptor hides planning, layout and runtime setup for user behind three calls: 
- choose a backend (CPU-mkl or GPU-cuSparse)
- choose a run configuration (the application)
- load artifacts into an `ExitStack` that owns the runtime.
The loaded GRG artifacts are then ready for execution with `grapp`.
A minimal example utilizing GPU (cuSparse) backend with pca: 

```python
from contextlib import ExitStack

from pygrgl_spmv import make_backend_cusparse, make_runconfig_pca, load_grg_spmv_multi
from grapp.grg_calculator import GRGSpMVCalculator
from grapp.linalg import PCs

backend = make_backend_cusparse(device=0)   # or make_backend_mkl(n_threads=0)
req = make_runconfig_pca()                  # kernel / pca / bolt / gwas

with ExitStack() as stack:
    grgs = [
        GRGSpMVCalculator(g)
        for g in load_grg_spmv_multi(artifacts, backend, req, stack)
    ]
    pcs_df, eig_vals = PCs(grgs, k=10, threads=4)
```

### Supported Backends and Parameters

The adaptor exposes two backends: MKL on CPU and cuSPARSE on GPU.

`make_backend_mkl(n_threads=0, optimize=False)`

| Parameter | Default | Meaning |
|---|---|---|
| `n_threads` | `0` | Threads per file. `0` auto-detects `physical_cores // n_files`, minimum 1. Also accepts a per-file dict. |
| `optimize` | `False` | Run MKL's inspector-executor `mkl_sparse_optimize()` on each matrix at load. Costs load time, can speed up repeated matmuls. |

`make_backend_cusparse(device=0, allow_residency=True, vram_budget_mb=0, capture=False, native=False)`

| Parameter | Default | Meaning |
|---|---|---|
| `device` | `0` | CUDA device index, or a per-file dict. Files on one device share a layout and runtime; separate devices load in parallel. |
| `allow_residency` | `True` | Keep every sparse block resident in VRAM. `False` selects streaming mode. |
| `vram_budget_mb` | `0` | VRAM cap in MiB. Used only in streaming mode, where it must be `> 0`; ignored when resident. |
| `capture` | `False` | Capture CUDA graphs after loading and return a `CapturedBoundGRG`, so matmul replays a graph instead of re-issuing kernels. |
| `native` | `False` | Keep matmul I/O on device (CuPy in, CuPy out, no host copies). Requires `capture=True` — on its own it is silently ignored. |

Both `n_threads` and `device` accept a mapping keyed by artifact file stem, which is how the JSON configs under `examples/configs/` are shaped:

```json
{"chr1": {"cuda_device": 0},        "chr2": {"cuda_device": 1}}
{"chr1": {"mkl_threads": [2, 1]},   "chr2": {"mkl_threads": [4, 1]}}
```

`mkl_threads` is `[n_up, n_down]` and is passed through as written. Note that a `0` here does *not* mean the same thing as the scalar `n_threads=0`: it skips the per-file division and falls through to MKL's own default of `os.cpu_count()`, so every file would get the whole machine.

### Supported Applications and RunConfigs

Run configurations carry the `RuntimeRequirements` for an application, which decides which buffers the runtime allocates and which graphs get captured:

| Factory | Application | Parameters beyond `force_spmm` |
|---|---|---|
| `make_runconfig_kernel` | matmul microbenchmarks | `direction` (`"up"`/`"down"`) and `k`, both required |
| `make_runconfig_pca` | PCA | `maxk` |
| `make_runconfig_bolt` | BOLT-LMM-inf | — |
| `make_runconfig_gwas` | GWAS | `maxk`, `sample_variance` |

`force_spmm=False` captures at `k=1` (the SpMV path); `True` captures at `k=2` (SpMM). Either way `k=1` callers still work — `CapturedBoundGRG.matmul()` zero-pads and truncates.
**When you're forced to use an earlier CUDA version, setting `force_spmm=True` can ensure correctness, at the cost of significant performance degradation.**
For GWAS with covariates set `maxk = n_covariates + 1` so the `X^T Q` product fits, and set `sample_variance=False` for the binomial-variance-only workload, which drops the `diag(X^T X)` graph.

## Advanced Usage

More freedom is provided when using the lower-level APIs directly. 
User may want to use them when fine-grained control or optimization is needed, or to develop new methods or applications.
An example and notes have been provided below.

### Core workflow

```python
import numpy as np

from pygrgl_spmv import RuntimeRequirements, convert
from pygrgl_spmv.backends.cusparse import CusparsePlanPair, CusparseRuntime, plan_cusparse_layout

artifact = convert("A.grg", "artifacts")
req = RuntimeRequirements(
    max_k_up=8,
    max_k_down=8,
    need_down_miss_input=True,
    need_up_miss_output=False,
    need_init_vector=True,
    need_init_matrix=False,
    need_init_xtx=True,
)
pair = CusparsePlanPair.from_dicts(
    {"store": "N", "fmt": "CSR", "opA": "N", "opB": "N", "orderB": "ROW", "orderC": "ROW", "algo": "DEFAULT", "scratch": "none"},
    {"store": "T", "fmt": "CSC", "opA": "N", "opB": "N", "orderB": "ROW", "orderC": "ROW", "algo": "DEFAULT", "scratch": "none"},
)
layout = plan_cusparse_layout(
    artifacts=[artifact],
    pair=pair,
    dtype=np.float64,
    requirements=req,
    vram_budget_bytes=8_000_000_000,
    ring_buffer_size=4,
    device=0,
    stream=0,
)

with CusparseRuntime(layout) as runtime:
    (A,) = runtime.grgs
    with A.prepare_matmul_cuda(direction="up", k=1) as op:
        op.input.copy_(op.input.new_tensor(np.ones((1, A.num_samples), dtype=np.float64)))
        op()
        y = op.output.cpu().numpy().copy()
```

### Public surface

- Package root exports:
  - `convert(...) -> Path`
  - `RuntimeRequirements`
  - `plan_reference_layout(...)`, `ReferenceRuntime`
  - `plan_mkl_layout(...)`, `MklRuntime`
- GPU backends are imported from their subpackages:
  - `pygrgl_spmv.backends.triton`
  - `pygrgl_spmv.backends.cusparse`
- GPU execution is centered on `grg.prepare_matmul_cuda(...)`; eager `grg.matmul(...)` is a NumPy convenience wrapper over that prepared path

### Notes

- planners and runtimes consume `.grg_spmv` artifacts only
- runtime-owned buffers are allocated in `__enter__()`
- one runtime owns one shared execution arena across all `runtime.grgs`
- concurrent calls on one runtime fail fast
- Triton and cuSPARSE support declared `max_k >= 1`
- GPU layouts can mix resident and streamed sparse blocks under a VRAM budget
- `ring_buffer_size=0` is valid for GPU layouts only when the budget keeps every sparse block resident
- the package root intentionally stays CPU-safe and does not re-export GPU runtime symbols
