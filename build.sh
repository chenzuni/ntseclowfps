#!/bin/bash

set -e

# 指定編譯器
export CC=clang-14
export CXX=clang++-14

# CMake 設定
CONFIG_FLAGS=(
  ""
  "-DPBRT_DBG_LOGGING=True"
  "-DPBRT_FLOAT_AS_DOUBLE=True"
)

# 檢查必要依賴（可選）
echo "Checking required packages..."
REQUIRED_PKGS=(libopenexr-dev libx11-dev libxcursor-dev libxrandr-dev libxinerama-dev libxi-dev libxext-dev libxfixes-dev libgl1-mesa-dev libwayland-bin libwayland-dev wayland-protocols libxkbcommon-dev libxkbcommon-x11-0)
for pkg in "${REQUIRED_PKGS[@]}"; do
  dpkg -s $pkg &>/dev/null || echo "⚠️  Warning: Missing $pkg"
done

# 取得腳本路徑
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 進入專案根目錄
cd "$SCRIPT_DIR"

# 迴圈處理所有設定
for config in "${CONFIG_FLAGS[@]}"; do
  echo "==============================="
  echo "Building with config: $config"
  echo "==============================="

  # 清除並建立 build 目錄
  rm -rf build
  mkdir build
  cd build

  # 設定編譯環境
  cmake .. -DCMAKE_CXX_STANDARD=14 -DPBRT_USE_PREGENERATED_RGB_TO_SPECTRUM_TABLES=True $config

  # 編譯
  cmake --build . --parallel --config Release

  # 回到根目錄
  cd ..
done

echo "✅ All builds completed."
ls
