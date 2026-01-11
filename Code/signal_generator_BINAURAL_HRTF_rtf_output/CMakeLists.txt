cmake_minimum_required(VERSION 3.14)
project(signal_generator LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_compile_options(-O3 -fPIC)

find_package(pybind11 REQUIRED)

pybind11_add_module(signal_generator
    bindings.cpp
    SignalGenerator.cpp
)