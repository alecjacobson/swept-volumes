# Python bindings (nanobind). Built when -DSWEPT_VOLUMES_PYTHON=ON.
find_package(Python 3.8 COMPONENTS Interpreter Development.Module REQUIRED)

# Locate nanobind. Prefer the installed package's CMake config (guarantees the
# same version used at runtime / by scikit-build-core); otherwise fetch it.
if(NOT TARGET nanobind::nanobind)
    execute_process(
        COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
        OUTPUT_STRIP_TRAILING_WHITESPACE
        OUTPUT_VARIABLE _nanobind_cmake_dir
        RESULT_VARIABLE _nanobind_status)
    if(_nanobind_status EQUAL 0 AND EXISTS "${_nanobind_cmake_dir}")
        list(APPEND CMAKE_PREFIX_PATH "${_nanobind_cmake_dir}")
        find_package(nanobind CONFIG REQUIRED)
    else()
        include(FetchContent)
        FetchContent_Declare(
            nanobind
            GIT_REPOSITORY https://github.com/wjakob/nanobind.git
            GIT_TAG v2.2.0)
        FetchContent_MakeAvailable(nanobind)
    endif()
endif()

nanobind_add_module(_swept_volumes STABLE_ABI src/bindings.cpp)
target_link_libraries(_swept_volumes PRIVATE core igl::core igl_copyleft::core)

# Install the extension into the Python package directory (for wheels).
install(TARGETS _swept_volumes LIBRARY DESTINATION swept_volumes)
