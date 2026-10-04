from __future__ import annotations

import os
from pathlib import Path
import site
import sys


_DLL_DIRECTORY_HANDLES = []
_REGISTERED_DIRECTORIES: set[str] = set()


def discover_cuda_dll_directories() -> tuple[Path, ...]:
    """Find CUDA DLL directories shipped inside the active Python environment."""
    if sys.platform != "win32":
        return ()

    site_roots = {Path(path) for path in site.getsitepackages() if path}
    site_roots.update(
        Path(path)
        for path in sys.path
        if path and (Path(path).name.casefold() in {"site-packages", "dist-packages"})
    )

    directories: list[Path] = []
    seen: set[str] = set()

    def add_if_has_dll(directory: Path) -> None:
        if not directory.is_dir() or not any(directory.glob("*.dll")):
            return
        key = str(directory.resolve()).casefold()
        if key not in seen:
            seen.add(key)
            directories.append(directory.resolve())

    for site_root in site_roots:
        # NVIDIA wheels install DLLs under package-specific bin/lib folders.
        nvidia_root = site_root / "nvidia"
        if nvidia_root.is_dir():
            for dll in nvidia_root.rglob("*.dll"):
                add_if_has_dll(dll.parent)

        # CTranslate2 ships cuDNN DLLs alongside its own extension module.
        add_if_has_dll(site_root / "ctranslate2")
        # Some CUDA dependencies installed with PyTorch live under torch/lib.
        add_if_has_dll(site_root / "torch" / "lib")

    # Also support Python environments that bundle native runtime DLLs here.
    add_if_has_dll(Path(sys.prefix) / "Library" / "bin")
    return tuple(directories)


def configure_cuda_dll_search() -> tuple[Path, ...]:
    """Expose environment-local CUDA DLLs to Windows loader and child processes."""
    directories = discover_cuda_dll_directories()
    if not directories:
        return ()

    current_path = os.environ.get("PATH", "")
    path_entries = current_path.split(os.pathsep)
    path_keys = {entry.casefold() for entry in path_entries if entry}

    for directory in directories:
        directory_text = str(directory)
        key = directory_text.casefold()
        if key not in path_keys:
            path_entries.insert(0, directory_text)
            path_keys.add(key)
        if key not in _REGISTERED_DIRECTORIES and hasattr(os, "add_dll_directory"):
            _DLL_DIRECTORY_HANDLES.append(os.add_dll_directory(directory_text))
            _REGISTERED_DIRECTORIES.add(key)

    os.environ["PATH"] = os.pathsep.join(path_entries)
    return directories
