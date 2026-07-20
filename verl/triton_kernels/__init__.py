"""Prefer the Triton kernels bundled with the installed vLLM.

The shared training environment also contains ``sgl-kernel``, which installs a
top-level ``triton_kernels`` package built for TokenSpeed Triton 3.8.  The
vLLM environment uses PyTorch's Triton 3.6; allowing that unrelated package to
win module discovery produces import errors and can load ABI-incompatible
extensions in vLLM subprocesses.  Since the repository root is first on
``PYTHONPATH``, this small namespace shim makes vLLM's version-matched bundled
copy authoritative without modifying the shared Conda environment.
"""

from vllm.third_party import triton_kernels as _bundled

__path__ = _bundled.__path__

