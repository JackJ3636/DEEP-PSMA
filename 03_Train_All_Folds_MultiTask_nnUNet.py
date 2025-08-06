"""
Train all 5 folds of the dual‑decoder nnU‑Net model for 200 epochs each.

This script runs training for all 5 cross-validation folds sequentially,
each for 200 epochs, providing comprehensive evaluation of the multi-task
segmentation model on the DEEP-PSMA dataset.

Usage:
    python 03_Train_All_Folds_MultiTask_nnUNet.py

Each fold will train for 200 epochs and save checkpoints along the way.
"""

import os
import sys
import time
from typing import List

# Ensure this script's directory is on the import path for nnunet_config_paths
sys.path.insert(0, os.path.dirname(__file__))

# Monkey‑patch importlib.invalidate_caches to dodge the Python 3.11 +
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


def train_single_fold(dataset_id: int, configuration: str, fold: int) -> bool:
    """Train a single fold and return success status."""
    try:
        # Import the nnU‑Net runner and directly invoke it in this process
        from nnunetv2.run.run_training import run_training

        print(f"\n{'='*80}")
        print(f"Starting training for fold {fold}")
        print(f"Dataset: {dataset_id}, Configuration: {configuration}")
        print(f"Expected epochs: 200")
        print(f"{'='*80}\n")
        
        start_time = time.time()
        
        run_training(
            dataset_name_or_id=str(dataset_id),
            configuration=configuration,
            fold=str(fold),
            trainer_class_name="MultiTaskUNetTrainer",
            plans_identifier="nnUNetPlans",
        )
        
        end_time = time.time()
        training_time = end_time - start_time
        
        print(f"\n{'='*80}")
        print(f"✅ Successfully completed training for fold {fold}")
        print(f"Training time: {training_time/3600:.2f} hours")
        print(f"{'='*80}\n")
        
        return True
        
    except Exception as e:
        print(f"\n{'='*80}")
        print(f"❌ Error training fold {fold}: {str(e)}")
        print(f"{'='*80}\n")
        return False


def main() -> None:
    # Configuration for multi-task training
    dataset_id = 900
    configuration = '3d_fullres'
    folds = [0, 1, 2, 3, 4]  # All 5 folds
    
    # (Optional) ensure correct libffi is loaded on your cluster
    current_ld = os.environ.get('LD_LIBRARY_PATH', '')
    libffi_path = '/apps/eb/el8/2023a/skylake/software/libffi/3.4.4-GCCcore-12.3.0/lib64'
    if libffi_path not in current_ld:
        os.environ['LD_LIBRARY_PATH'] = f"{libffi_path}:{current_ld}"

    print(f"🚀 Starting 5-fold cross-validation training")
    print(f"Dataset: {dataset_id} (MultiTask)")
    print(f"Configuration: {configuration}")
    print(f"Folds: {folds}")
    print(f"Epochs per fold: 200")
    print(f"Total training runs: {len(folds)}")
    
    # Track results
    successful_folds: List[int] = []
    failed_folds: List[int] = []
    overall_start_time = time.time()
    
    # Train each fold
    for fold in folds:
        print(f"\n🎯 Training fold {fold}/{max(folds)}")
        
        success = train_single_fold(dataset_id, configuration, fold)
        
        if success:
            successful_folds.append(fold)
        else:
            failed_folds.append(fold)
            
        # Brief pause between folds
        if fold < max(folds):
            print("⏳ Pausing 30 seconds before next fold...")
            time.sleep(30)
    
    # Final summary
    overall_end_time = time.time()
    total_time = overall_end_time - overall_start_time
    
    print(f"\n{'='*80}")
    print(f"🏁 TRAINING COMPLETE - FINAL SUMMARY")
    print(f"{'='*80}")
    print(f"Total training time: {total_time/3600:.2f} hours")
    print(f"Successfully completed folds: {successful_folds} ({len(successful_folds)}/5)")
    
    if failed_folds:
        print(f"Failed folds: {failed_folds} ({len(failed_folds)}/5)")
        print(f"❌ Some folds failed. Check logs for details.")
    else:
        print(f"✅ All 5 folds completed successfully!")
        print(f"🎉 5-fold cross-validation training completed!")
    
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
