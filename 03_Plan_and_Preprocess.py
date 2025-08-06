"""
Plan and preprocess the multi-task dataset for nnUNet training.

This script runs nnUNet's planning and preprocessing pipeline on 
Dataset900_MultiTask. It must be run before training.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

# Monkey-patch to fix PyTorch + Python 3.11 importlib issue
import importlib
_orig_inv = importlib.invalidate_caches
def _safe_invalidate_caches():
    try:
        _orig_inv()
    except TypeError:
        # skip finders that want a 'cls' argument
        pass
importlib.invalidate_caches = _safe_invalidate_caches

import nnunet_config_paths  # This sets up nnUNet environment variables


def main() -> None:
    # ID of the multi‑task dataset
    dataset_id = 900
    
    # Set up library path for libffi to fix PyTorch import issues
    current_ld_path = os.environ.get('LD_LIBRARY_PATH', '')
    libffi_path = '/apps/eb/el8/2023a/skylake/software/libffi/3.4.4-GCCcore-12.3.0/lib64'
    if libffi_path not in current_ld_path:
        os.environ['LD_LIBRARY_PATH'] = f"{libffi_path}:{current_ld_path}"

    cmd = f"nnUNetv2_plan_and_preprocess -d {dataset_id} --verify_dataset_integrity"
    print(f"Running: {cmd}")
    os.system(cmd)


if __name__ == '__main__':
    main()
