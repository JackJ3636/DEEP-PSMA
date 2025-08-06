"""
Launch training of the dual‑decoder nnU‑Net model in‑process, preserving
our importlib monkey‑patch so PyTorch FSDP doesn’t crash under Python 3.11.

Usage:
    python 02_Train_MultiTask_nnUNet.py

Edit `dataset_id`, `configuration` or `fold` below to target a different
nnU‑Net task, resolution or CV fold.
"""

import os
import sys

# Ensure this script’s directory is on the import path for nnunet_config_paths
sys.path.insert(0, os.path.dirname(__file__))

# Monkey‑patch importlib.invalidate_caches to dodge the Python 3.11 +
# PyTorch FSDP TypeError in MetadataPathFinder.invalidate_caches()
import importlib
_orig_inv = importlib.invalidate_caches

def _safe_invalidate_caches():
    try:
        _orig_inv()
    except TypeError:
        pass

importlib.invalidate_caches = _safe_invalidate_caches

# Set up nnU‑Net environment (raw, preprocessed, results paths)
import nnunet_config_paths


def main() -> None:
    # ID of your multi-task dataset (from the preprocessing step)
    dataset_id = 900
    # Use the 3D full‑resolution configuration
    configuration = '3d_fullres'
    # Cross‑validation fold (0–4)
    fold = 0

    # (Optional) ensure correct libffi is loaded on your cluster
    current_ld = os.environ.get('LD_LIBRARY_PATH', '')
    libffi_path = '/apps/eb/el8/2023a/skylake/software/libffi/3.4.4-GCCcore-12.3.0/lib64'
    if libffi_path not in current_ld:
        os.environ['LD_LIBRARY_PATH'] = f"{libffi_path}:{current_ld}"

    # Import the nnU‑Net runner and directly invoke it in this process
    from nnunetv2.run.run_training import run_training

    print(f"Launching in-process nnU‑Net training: dataset={dataset_id}, "
          f"config={configuration}, fold={fold}")
    run_training(
        dataset_name_or_id=str(dataset_id),
        configuration=configuration,
        fold=str(fold),
        trainer_class_name="MultiTaskUNetTrainer",
        plans_identifier="nnUNetPlans",
    )


if __name__ == '__main__':
    main()
