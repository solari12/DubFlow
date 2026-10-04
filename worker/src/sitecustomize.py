"""Configure environment-local CUDA DLL lookup before Python imports native modules."""

try:
    from dubflow_worker.runtime.cuda import configure_cuda_dll_search

    configure_cuda_dll_search()
except ImportError:
    # This source tree may be present on sys.path before the worker package is installed.
    pass
