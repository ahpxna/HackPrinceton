"""Print capability checks without installing heavyweight CUDA components."""
import importlib.util
import shutil
import sys
from pathlib import Path
print('Python:',sys.version.split()[0])
for pkg in ('numpy','PIL','torch','nerfstudio','gsplat','plyfile','cv2'):
    print(pkg, 'available' if importlib.util.find_spec(pkg) else 'not installed')
for cmd in ('colmap','ns-train','nvidia-smi'):
    print(cmd, shutil.which(cmd) or 'not found in PATH')
try:
    import torch
    print('torch CUDA available:',torch.cuda.is_available())
    if torch.cuda.is_available(): print('GPU:',torch.cuda.get_device_name(0))
except Exception as e: print('Torch GPU check:',type(e).__name__,str(e))
