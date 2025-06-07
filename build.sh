#!/bin/bash

set -e

# 使用 clang-19 編譯器
export CC=clang-19
export CXX=clang++-19

# 取得當前目錄
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 重建 build 目錄
rm -rf build
mkdir build
cd build

# 執行 CMake 設定
cmake .. -DCMAKE_CXX_STANDARD=14

# 編譯專案
cmake --build . --parallel --config Release

# 返回上層目錄
cd ..

echo "✅ All builds completed successfully using clang-19."
ls
