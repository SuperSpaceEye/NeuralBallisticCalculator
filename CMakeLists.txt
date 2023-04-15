cmake_minimum_required(VERSION 3.24)
project(CannonBallisticCalculator)

set(CMAKE_CXX_STANDARD 20)

find_package(pybind11 REQUIRED)

set(PROJECT_FILES cpp_src/main.cpp cpp_src/ballistic_functions.h cpp_src/ballistic_functions.h)

pybind11_add_module(_test_fn ${PROJECT_FILES})

#target_compile_definitions(${PROJECT_NAME} PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})
#target_include_directories(${PROJECT_NAME} PRIVATE ${PYBIND11_INCLUDE_DIRS})