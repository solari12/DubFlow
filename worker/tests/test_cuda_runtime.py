from __future__ import annotations

from dubflow_worker.runtime import cuda


def test_discover_cuda_dll_directories_in_active_environment(tmp_path, monkeypatch):
    site_packages = tmp_path / "Lib" / "site-packages"
    cublas_dir = site_packages / "nvidia" / "cublas" / "bin"
    cudnn_dir = site_packages / "nvidia" / "cudnn" / "bin"
    ctranslate2_dir = site_packages / "ctranslate2"
    torch_lib_dir = site_packages / "torch" / "lib"
    for directory, dll in (
        (cublas_dir, "cublas64_12.dll"),
        (cudnn_dir, "cudnn64_9.dll"),
        (ctranslate2_dir, "ctranslate2.dll"),
        (torch_lib_dir, "cudart64_12.dll"),
    ):
        directory.mkdir(parents=True, exist_ok=True)
        (directory / dll).touch()

    monkeypatch.setattr(cuda.sys, "platform", "win32")
    monkeypatch.setattr(cuda.site, "getsitepackages", lambda: [str(site_packages)])
    monkeypatch.setattr(cuda.sys, "path", [str(site_packages)])

    discovered = set(cuda.discover_cuda_dll_directories())

    assert discovered == {cublas_dir, cudnn_dir, ctranslate2_dir, torch_lib_dir}


def test_configure_cuda_dll_search_updates_path_and_registers_directories(tmp_path, monkeypatch):
    site_packages = tmp_path / "Lib" / "site-packages"
    cublas_dir = site_packages / "nvidia" / "cublas" / "bin"
    cublas_dir.mkdir(parents=True)
    (cublas_dir / "cublas64_12.dll").touch()

    registered = []
    monkeypatch.setattr(cuda.sys, "platform", "win32")
    monkeypatch.setattr(cuda.site, "getsitepackages", lambda: [str(site_packages)])
    monkeypatch.setattr(cuda.sys, "path", [str(site_packages)])
    monkeypatch.setattr(cuda.os, "add_dll_directory", lambda path: registered.append(path) or path)
    monkeypatch.setattr(cuda, "_REGISTERED_DIRECTORIES", set())
    monkeypatch.setattr(cuda, "_DLL_DIRECTORY_HANDLES", [])
    monkeypatch.setenv("PATH", "existing-path")

    result = cuda.configure_cuda_dll_search()

    assert result == (cublas_dir.resolve(),)
    assert str(cublas_dir.resolve()) in cuda.os.environ["PATH"]
    assert registered == [str(cublas_dir.resolve())]
