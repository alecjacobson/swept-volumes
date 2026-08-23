cmake_minimum_required(VERSION 3.20)
project(swept_volumes LANGUAGES CXX)

# CMake 4.x drops compatibility with projects declaring cmake_minimum_required < 3.5.
# Some fetched dependencies still do; keep them configurable.
if(NOT DEFINED CMAKE_POLICY_VERSION_MINIMUM)
    set(CMAKE_POLICY_VERSION_MINIMUM 3.5)
endif()

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

list(PREPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# ---------------------------------------------------------------------------
# Dependencies (all fetched via CMake FetchContent -- no submodule/system)
# ---------------------------------------------------------------------------
include(libigl)                 # provides igl::core (+ Eigen) and igl_include()
igl_include(copyleft core)      # igl_copyleft::core  (igl/copyleft/marching_cubes.h)

# ---------------------------------------------------------------------------
# Core library: the swept-volume algorithm (no main(), no GUI)
# ---------------------------------------------------------------------------
file(GLOB SRCFILES ${CMAKE_CURRENT_SOURCE_DIR}/include/*.cpp)
add_library(core ${SRCFILES})
target_include_directories(core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include/)
target_link_libraries(core PUBLIC igl::core igl_copyleft::core)

# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------
option(SWEPT_VOLUMES_GUI      "Build the GLFW/ImGui GUI (needs porting to libigl 2.6)" OFF)
option(SWEPT_VOLUMES_PYTHON   "Build the nanobind Python bindings" OFF)
option(SWEPT_VOLUMES_BENCH    "Build the native C++ benchmark driver" ON)

if(SWEPT_VOLUMES_BENCH)
    add_executable(swept-volumes-bench ${CMAKE_CURRENT_SOURCE_DIR}/bench/native_bench.cpp)
    set_target_properties(swept-volumes-bench PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/)
    target_link_libraries(swept-volumes-bench PRIVATE core igl::core igl_copyleft::core)

    add_executable(sparse-dc-experiment ${CMAKE_CURRENT_SOURCE_DIR}/experiments/sparse_dc_static.cpp)
    set_target_properties(sparse-dc-experiment PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/)
    target_link_libraries(sparse-dc-experiment PRIVATE igl::core igl_copyleft::core)
endif()

if(SWEPT_VOLUMES_GUI)
    igl_include(glfw)
    igl_include(imgui)
    add_executable(swept-volumes-gui ${CMAKE_CURRENT_SOURCE_DIR}/swept-volumes-gui.cpp)
    set_target_properties(swept-volumes-gui PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/)
    target_link_libraries(swept-volumes-gui PRIVATE core igl::core igl::glfw igl::imgui)
endif()

if(SWEPT_VOLUMES_PYTHON)
    add_subdirectory(python)
endif()
