"""Test setup: put the CT and X-ray inference folders on sys.path. CPU only, no model weights."""
import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for sub in ("ct", "cxr"):
    if str(ROOT / sub) not in sys.path:
        sys.path.insert(0, str(ROOT / sub))


def optional(name):
    """Import an optional dependency, or None when it is missing or broken in the test environment."""
    try:
        module = importlib.import_module(name)
    except Exception:
        return None
    return module


def load_entry(modality: str):
    """Import inference/<modality>/infer.py under a unique module name (both entry points are called infer.py)."""
    import importlib.util
    name = modality + "_infer"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, ROOT / modality / "infer.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module
