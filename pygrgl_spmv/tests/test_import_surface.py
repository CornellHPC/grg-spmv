from __future__ import annotations

import json
import subprocess
import sys
import textwrap


def test_cpu_safe_import_surface_without_optional_gpu_modules():
    script = textwrap.dedent(
        """
        import importlib.abc
        import json
        import sys

        class Blocker(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if (
                    fullname == "cupy" or fullname.startswith("cupy.")
                    or fullname == "torch" or fullname.startswith("torch.")
                    or fullname == "triton" or fullname.startswith("triton.")
                ):
                    raise ModuleNotFoundError(fullname)
                return None

        sys.meta_path.insert(0, Blocker())
        import pygrgl_spmv
        import pygrgl_spmv.backends as backends
        import pygrgl_spmv.backends.mkl
        import pygrgl_spmv.backends.reference

        print(
            json.dumps(
                {
                    "root_has_cusparse": hasattr(pygrgl_spmv, "CusparseRuntime"),
                    "root_has_triton": hasattr(pygrgl_spmv, "TritonRuntime"),
                    "backends_has_cusparse": hasattr(backends, "CusparseRuntime"),
                    "backends_has_triton": hasattr(backends, "TritonRuntime"),
                }
            )
        )
        """
    )
    result = subprocess.run([sys.executable, "-c", script], check=True, capture_output=True, text=True)
    payload = json.loads(result.stdout)
    assert payload == {
        "root_has_cusparse": False,
        "root_has_triton": False,
        "backends_has_cusparse": False,
        "backends_has_triton": False,
    }
