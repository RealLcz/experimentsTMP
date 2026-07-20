"""Opt-in diagnostics for native Python extension imports in batch jobs."""

import importlib.machinery
import os


if os.environ.get("VERL_TRACE_EXTENSION_IMPORTS") == "1":
    _original_create_module = importlib.machinery.ExtensionFileLoader.create_module

    def _traced_create_module(self, spec):
        print(f"[native-extension] loading {spec.name} from {spec.origin}", flush=True)
        module = _original_create_module(self, spec)
        print(f"[native-extension] loaded {spec.name}", flush=True)
        return module

    importlib.machinery.ExtensionFileLoader.create_module = _traced_create_module

