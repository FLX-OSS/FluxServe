# Flux Scheduler

C++ scheduler core with Python bindings used by FluxServe `scheduler_policy=paged`.

## Installation`

```bash
pip install -e flux-scheduler
```

## Build Notes

The Python package is built by `scikit-build-core` using `CMakeLists.txt`.
The default editable install builds the Python module with:

```text
-DFLUX_SCHEDULER_BUILD_TESTS=OFF
-DFLUX_SCHEDULER_BUILD_PYTHON=ON
```

If CMake cannot find `nanobind`, install the Python build dependencies first:

```bash
python3 -m pip install "scikit-build-core>=0.9.5" "nanobind>=1.7.0" "cmake>=3.22" "ninja>=1.11"
```
