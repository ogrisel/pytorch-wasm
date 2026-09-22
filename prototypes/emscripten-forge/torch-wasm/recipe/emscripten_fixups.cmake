# Injected via -DCMAKE_PROJECT_INCLUDE right after the top-level project().
#
# 1) wasm CPython is statically linked (no shared libpython), so
#    find_package(Python COMPONENTS Development.Module) does not create the
#    Python::Module target that torch's cmake/Dependencies.cmake links into
#    pybind11. Provide a header-only INTERFACE stub; the actual Python symbols
#    resolve at side-module load time (like every other emscripten-forge
#    Python extension).
if(NOT TARGET Python::Module)
  add_library(Python::Module INTERFACE IMPORTED GLOBAL)
  if(DEFINED WASM_PYTHON_INCLUDE_DIR)
    set_target_properties(Python::Module PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES "${WASM_PYTHON_INCLUDE_DIR}")
  endif()
endif()
if(NOT TARGET Python::Python)
  add_library(Python::Python INTERFACE IMPORTED GLOBAL)
  if(DEFINED WASM_PYTHON_INCLUDE_DIR)
    set_target_properties(Python::Python PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES "${WASM_PYTHON_INCLUDE_DIR}")
  endif()
endif()
