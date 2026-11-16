#!/usr/bin/env bash
# This installs into the current WSL user's isolated application directory.
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
runtime_root="${FALL_BASELINE_RUNTIME:-${XDG_DATA_HOME:-$HOME/.local/share}/fall-review-baseline}"
[[ -f "$project_dir/serve.py" && -f "$project_dir/requirements-baseline-runtime.txt" ]]
for asset in \
  results/runs/urfd_2026/best_erm.pt \
  data/model_weights/rtmdet_m_8xb32-300e_coco_20220719_112220-229f527c.pth \
  data/model_weights/rtmpose-m_simcc-aic-coco_pt-aic-coco_420e-256x192-63eb25f7_20230126.pth; do
  if [[ ! -f "$project_dir/$asset" ]]; then
    echo "Missing copied baseline asset: $project_dir/$asset" >&2
    exit 1
  fi
done
mkdir -p "$runtime_root"
unset VIRTUAL_ENV PYTHONHOME PYTHONPATH
export UV_CACHE_DIR="$runtime_root/cache"
export UV_PYTHON_INSTALL_DIR="$runtime_root/python"
if [[ ! -x "$runtime_root/bootstrap/bin/python" ]]; then
  python3 -m venv "$runtime_root/bootstrap"
fi
"$runtime_root/bootstrap/bin/python" -m pip install --disable-pip-version-check 'uv==0.11.15'
uv_bin="$runtime_root/bootstrap/bin/uv"
python_home="$runtime_root/cpython-3.11.18"
if [[ ! -x "$python_home/bin/python3.11" ]]; then
  archive="$runtime_root/cpython-3.11.18.tar.gz"
  curl --fail --location --retry 3 --output "$archive" \
    'https://github.com/astral-sh/python-build-standalone/releases/download/20261110/cpython-3.11.18%2B20261110-x86_64-unknown-linux-gnu-install_only.tar.gz'
  echo "ac0b758b6ae4ba1cb87f7091eb6bb17c9a383a9bcdb5c18493cdd4f34e35ae8c  $archive" | sha256sum --check
  mkdir -p "$python_home"
  tar -xzf "$archive" -C "$python_home" --strip-components=1
fi
if [[ ! -x "$runtime_root/venv/bin/python" ]]; then
  "$uv_bin" venv --seed --python "$python_home/bin/python3.11" "$runtime_root/venv"
fi
python_bin="$runtime_root/venv/bin/python"
"$uv_bin" pip install --python "$python_bin" \
  'torch==2.9.1' 'torchvision==0.24.1' --index-url https://download.pytorch.org/whl/cpu
"$uv_bin" pip install --python "$python_bin" \
  -r "$project_dir/requirements.txt" -r "$project_dir/requirements-app.txt" \
  'setuptools==75.8.0' 'wheel==0.45.1' 'ninja==1.11.1.3' 'Cython<3' \
  'mmengine==0.10.7' 'opencv-python==4.10.0.84'
# Build original MMCV against this exact CPU PyTorch instead of substituting
# an incompatible precompiled wheel or a different pose engine.
export MMCV_WITH_OPS=1 FORCE_CUDA=0 MAX_JOBS="${FALL_BUILD_JOBS:-4}"
# Standalone Python records Clang in sysconfig; use Ubuntu's actual GCC tools.
# Keep c++ as the command so Ninja can reuse completed object files on retry.
export CC=cc CXX=c++ LDSHARED='cc -shared'
if ! "$uv_bin" pip install --python "$python_bin" --no-build-isolation --no-binary mmcv 'mmcv==2.1.0' > "$runtime_root/mmcv-build.log" 2>&1; then
  tail -n 80 "$runtime_root/mmcv-build.log"
  exit 1
fi
tail -n 12 "$runtime_root/mmcv-build.log"
"$uv_bin" pip install --python "$python_bin" --no-build-isolation \
  -r "$project_dir/requirements-baseline-runtime.txt"
"$python_bin" -m pip freeze > "$runtime_root/installed-requirements.txt"
cd "$project_dir"
"$python_bin" -X utf8 scripts/check_baseline_runtime.py --initialize
echo "Local baseline interpreter: $python_bin"
