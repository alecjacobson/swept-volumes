"""Make the built extension importable without an install.

If `swept_volumes` is already importable (e.g. `pip install .`), use it.
Otherwise add the in-repo `python/` dir to sys.path and copy the freshly built
`_swept_volumes` extension next to the package.
"""
import glob
import os
import shutil
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKG_DIR = os.path.join(REPO, "python")


def _ensure_extension():
    try:
        import swept_volumes  # noqa: F401
        return
    except Exception:
        pass

    pkg = os.path.join(PKG_DIR, "swept_volumes")
    have = glob.glob(os.path.join(pkg, "_swept_volumes*.so"))
    if not have:
        candidates = glob.glob(
            os.path.join(REPO, "build", "**", "_swept_volumes*.so"), recursive=True
        )
        if candidates:
            shutil.copy(candidates[0], pkg)
    sys.path.insert(0, PKG_DIR)


_ensure_extension()
