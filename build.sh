#!/bin/bash

set -e

# 使用 clang-19 編譯器
export CC=clang-19
export CXX=clang++-19

# CMake 設定組合（與 GitHub Actions 中一致）
CONFIG_FLAGS=(
  ""
  "-DPBRT_DBG_LOGGING=True"
  "-DPBRT_FLOAT_AS_DOUBLE=True"
)

# 檢查必要的套件（非強制）
echo "Checking for required system packages..."
REQUIRED_PKGS=(
  libopenexr-dev
  libx11-dev libxcursor-dev libxrandr-dev libxinerama-dev
  libxi-dev libxext-dev libxfixes-dev libgl1-mesa-dev
  libwayland-bin libwayland-dev wayland-protocols
  libxkbcommon-dev libxkbcommon-x11-0
)
for pkg in "${REQUIRED_PKGS[@]}"; do
  dpkg -s "$pkg" &>/dev/null || echo "⚠️  Missing package: $pkg"
done

# 取得當前目錄
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 逐一編譯每個 config
for config in "${CONFIG_FLAGS[@]}"; do
  echo "==============================="
  echo "Building with config: $config"
  echo "Using compiler: $CXX"
  echo "==============================="

  # 重建 build 目錄
  rm -rf build
  mkdir build
  cd build

  # 執行 CMake 設定
  cmake .. -DCMAKE_CXX_STANDARD=14 $config

  # 編譯專案
  cmake --build . --parallel --config Release

  # 返回上層目錄
  cd ..
done

echo "✅ All builds completed successfully using clang-19."
