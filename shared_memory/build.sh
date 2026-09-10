#!/bin/bash

set -e

base_path="$(pwd)"
build_type="Debug"
#build_type="Release"

cmake -S . -B build -DCMAKE_BUILD_TYPE=$build_type && \
    cmake --build build