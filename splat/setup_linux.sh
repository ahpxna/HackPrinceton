#!/usr/bin/env bash
set -euo pipefail
umask 022

# Run with bash setup_linux.sh. No root or API credentials.
[[ $(uname -s) == Linux && $(uname -m) == x86_64 ]] || {
  echo 'This installer targets Linux x86_64 with an NVIDIA GPU.' >&2; exit 1;
}
command -v nvidia-smi >/dev/null || { echo 'nvidia-smi is missing.' >&2; exit 1; }
nvidia-smi
prefix="$HOME/.local/envs/hp-splat"
state="$HOME/hp_splat/send_back/proof"
setup_hash=$(sha256sum "${BASH_SOURCE[0]}" | cut -d' ' -f1)
if [[ -f "$state/setup-complete.sha256" && -f "$state/activate.sh" ]] && \
   [[ $(cat "$state/setup-complete.sha256") == "$setup_hash" ]] && \
   "$prefix/bin/python" -c 'import torch, gsplat, cv2; assert torch.cuda.is_available()' && \
   [[ -x "$prefix/bin/splat-transform" && -f "$HOME/.local/share/hp-splat-web/node_modules/@sparkjsdev/spark/dist/spark.module.js" ]]; then
  echo 'Setup already completed for this installer; reusing environment.'
  exit 0
fi
driver_cuda=$(nvidia-smi | sed -n 's/.*CUDA Version: \([0-9]*\.[0-9]*\).*/\1/p' | head -n 1)
[[ -n $driver_cuda ]] || { echo 'Cannot determine driver CUDA support.' >&2; exit 1; }
major=${driver_cuda%%.*}; minor=${driver_cuda#*.}
if (( major > 12 || (major == 12 && minor >= 8) )); then
  cuda=12.8; wheel=cu128; compiler=12
elif (( major == 12 && minor >= 6 )); then
  cuda=12.6; wheel=cu126; compiler=12
elif (( major > 11 || (major == 11 && minor >= 8) )); then
  cuda=11.8; wheel=cu118; compiler=11
else
  echo 'Driver must support CUDA 11.8 or newer. No install attempted.' >&2; exit 1
fi
available_kb=$(df -Pk "$HOME" | awk 'END {print $4}')
(( available_kb >= 60000000 )) || { echo 'Need at least 60 GB free under HOME.' >&2; exit 1; }
mkdir -p "$HOME/.local/bin" "$state"
exec > >(tee "$state/setup.log") 2>&1
echo "Driver supports CUDA $driver_cuda; using torch 2.7.1 / $wheel / toolkit $cuda."
export MAMBA_ROOT_PREFIX="$HOME/.local/share/hp-splat-mamba"
if command -v micromamba >/dev/null; then
  mamba=$(command -v micromamba)
elif [[ -x "$HOME/.local/bin/micromamba" ]]; then
  mamba="$HOME/.local/bin/micromamba"
else
  command -v curl >/dev/null || { echo 'curl is required.'; exit 1; }
  scratch=$(mktemp -d)
  trap 'rm -rf "$scratch"' EXIT
  curl --fail --location --retry 4 https://micro.mamba.pm/api/micromamba/linux-64/latest -o "$scratch/mamba.tar.bz2"
  tar -xjf "$scratch/mamba.tar.bz2" -C "$scratch" bin/micromamba
  install -m 755 "$scratch/bin/micromamba" "$HOME/.local/bin/micromamba"
  mamba="$HOME/.local/bin/micromamba"
fi
if [[ -d "$prefix" && ! -f "$prefix/conda-meta/history" ]]; then
  echo "Refusing to overwrite existing non-conda directory: $prefix"; exit 1
fi
operation=create
[[ -f "$prefix/conda-meta/history" ]] && operation=install
"$mamba" "$operation" -y -p "$prefix" --override-channels -c conda-forge -c nvidia \
  python=3.10 pip nodejs=22 "cuda-toolkit=$cuda" \
  "gcc_linux-64=$compiler" "gxx_linux-64=$compiler" ninja cmake ffmpeg
eval "$("$mamba" shell hook -s bash)"
micromamba activate "$prefix"
export CUDA_HOME="$prefix"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export TORCH_EXTENSIONS_DIR="$HOME/.cache/hp-splat/torch-extensions"
export MAX_JOBS=${MAX_JOBS:-4}
python -m pip install --no-cache-dir --upgrade pip wheel 'setuptools<81'
constraints="$state/install-constraints.txt"
cat > "$constraints" <<'EOF'
torch==2.7.1
torchvision==0.22.1
numpy==1.26.4
gsplat==1.4.0
opencv-python-headless==4.10.0.84
EOF
python -m pip install --no-cache-dir -c "$constraints" torch==2.7.1 torchvision==0.22.1 \
  --index-url "https://download.pytorch.org/whl/$wheel"
python -m pip install --no-cache-dir -c "$constraints" numpy==1.26.4 ninja packaging \
  gsplat==1.4.0 opencv-python-headless==4.10.0.84 pillow
npm install --global --prefix "$prefix" @playcanvas/splat-transform@3.11.2
npm install --prefix "$HOME/.local/share/hp-splat-web" --save-exact @sparkjsdev/spark@2.4.0 three@0.180.0
python -m pip check
# Compile/JIT and execute the real gsplat CUDA forward/backward path, not just import.
python - <<'PY'
import importlib.metadata as md
import json
import torch
from gsplat import rasterization
assert torch.cuda.is_available(), 'PyTorch cannot use the GPU'
device='cuda'
means=torch.tensor([[0.,0.,3.]],device=device,requires_grad=True)
quats=torch.tensor([[1.,0.,0.,0.]],device=device)
scales=torch.full((1,3),.1,device=device)
opacities=torch.ones(1,device=device)*.8
colors=torch.tensor([[1.,.2,.1]],device=device)
viewmats=torch.eye(4,device=device)[None]
ks=torch.tensor([[[32.,0.,16.],[0.,32.,16.],[0.,0.,1.]]],device=device)
rgb,alpha,_=rasterization(means,quats,scales,opacities,colors,viewmats,ks,32,32)
assert torch.isfinite(rgb).all() and alpha.max()>0
rgb.sum().backward()
assert means.grad is not None and torch.isfinite(means.grad).all()
result={'cuda_raster_forward_backward':'passed','GPU':torch.cuda.get_device_name(),
        'torch_cuda':torch.version.cuda,'versions':{p:md.version(p) for p in
        ['torch','torchvision','gsplat','opencv-python-headless','Pillow']}}
from pathlib import Path
Path.home().joinpath('hp_splat/send_back/proof/setup_versions.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
PY
nvcc --version
node --version
npm --version
npm list --global --prefix "$prefix" --depth=0
splat-transform --help
python -m pip freeze > "$state/setup-freeze.txt"
cat > "$state/activate.sh" <<EOF
export MAMBA_ROOT_PREFIX='$MAMBA_ROOT_PREFIX'
eval "\$("$mamba" shell hook -s bash)"
micromamba activate '$prefix'
export CUDA_HOME='$prefix'
export LD_LIBRARY_PATH="$prefix/lib:$prefix/lib64:\${LD_LIBRARY_PATH:-}"
export TORCH_EXTENSIONS_DIR='${TORCH_EXTENSIONS_DIR}'
EOF
echo "$setup_hash" > "$state/setup-complete.sha256"
echo "Setup passed. Activate later: source $state/activate.sh"
echo "Proof: $state/setup.log and setup_versions.json. No training was started."
