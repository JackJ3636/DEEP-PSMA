"""
Preprocess the DEEP‑PSMA training data for a multi‑task segmentation model.

This script aggregates the PSMA and FDG modalities into a single
four‑channel image per case and generates two separate label maps:
one for PSMA and one for FDG. Each label map encodes three classes
background (0), tumour/disease (1) and normal physiological uptake (2).

The resulting data are saved into a new nnU‑Net style dataset folder
called ``Dataset900_MultiTask`` under the configured nnU‑Net raw data
directory. The four channels correspond to PSMA PET, PSMA CT, FDG PET
and FDG CT, all resampled to the PSMA PET resolution. A minimal
``dataset.json`` file is also created. After running this script you
can use the provided training script to train the multi‑task model.

Usage:
    python 01_Preprocess_MultiTask_Dataset.py

Requires SimpleITK and numpy. The script expects that the DEEP‑PSMA
training data have been copied into the ``data`` folder using
``00_copy_deep_psma_training_data.py``. It also depends on
``nnunet_config_paths.py`` to determine where the nnU‑Net raw data
directory lives. You do not need to run ``nnUNetv2_plan_and_preprocess``
on this dataset because the provided training code does not rely on
nnU‑Net's automatic configuration.
"""

import os
from os.path import join
import json
import numpy as np
import SimpleITK as sitk

import nnunet_config_paths  # defines nnUNet data directories


def resample_to_ref(image: sitk.Image, reference: sitk.Image, default_value: float = 0.0) -> sitk.Image:
    """Resample ``image`` to match ``reference`` in spacing, origin and orientation.

    Parameters
    ----------
    image : sitk.Image
        The image to resample.
    reference : sitk.Image
        The reference image whose geometry will be used.
    default_value : float, optional
        The voxel value assigned to regions outside the original ``image``.

    Returns
    -------
    sitk.Image
        Resampled image with geometry copied from ``reference``.
    """
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(reference)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(default_value)
    return resampler.Execute(image)


def create_multitask_dataset(input_dataset_folder: str, output_dataset_id: int = 900) -> None:
    """Create a multi‑task nnU‑Net dataset combining PSMA and FDG.

    This function fuses PSMA and FDG PET/CT into a four‑channel input
    image and produces a single five‑class label map per case.  The
    label codes are:

    * 0 – background
    * 1 – PSMA tumour
    * 2 – PSMA normal uptake
    * 3 – FDG tumour
    * 4 – FDG normal uptake

    Parameters
    ----------
    input_dataset_folder : str
        Path to the folder containing the copied DEEP‑PSMA training data.
        The expected structure is ``<input_dataset_folder>/<tracer>/<modality>``.
    output_dataset_id : int, optional
        Identifier for the new nnU‑Net dataset.  Determines the name of
        the output folder ``Dataset{ID}_MultiTask`` under the nnU‑Net
        raw data directory.  Default is 900.
    """
    dataset_name = f"Dataset{output_dataset_id}_MultiTask"
    output_base = join(nnunet_config_paths.nn_raw_dir, dataset_name)
    images_dir = join(output_base, "imagesTr")
    labels_dir = join(output_base, "labelsTr")

    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    # Iterate through cases based on PSMA directory
    psma_ct_dir = join(input_dataset_folder, "PSMA", "CT")
    cases = [f for f in os.listdir(psma_ct_dir) if f.endswith(".nii.gz")]
    cases.sort()
    for case_fname in cases:
        case_id = case_fname.replace(".nii.gz", "")
        print(f"Processing {case_id}")

        # Load PSMA modalities
        psma_ct = sitk.ReadImage(join(input_dataset_folder, "PSMA", "CT", case_fname))
        psma_pet = sitk.ReadImage(join(input_dataset_folder, "PSMA", "PET", case_fname))
        psma_ttb = sitk.ReadImage(join(input_dataset_folder, "PSMA", "TTB", case_fname))
        with open(join(input_dataset_folder, "PSMA", "thresholds", case_id + ".json"), "r") as f:
            psma_thresh = json.load(f)["suv_threshold"]

        # Load FDG modalities
        fdg_ct = sitk.ReadImage(join(input_dataset_folder, "FDG", "CT", case_fname))
        fdg_pet = sitk.ReadImage(join(input_dataset_folder, "FDG", "PET", case_fname))
        fdg_ttb = sitk.ReadImage(join(input_dataset_folder, "FDG", "TTB", case_fname))
        with open(join(input_dataset_folder, "FDG", "thresholds", case_id + ".json"), "r") as f:
            fdg_thresh = json.load(f)["suv_threshold"]

        # Rescale PET intensities: SUV threshold becomes 1.0
        psma_pet_rescaled = psma_pet / float(psma_thresh)
        fdg_pet_rescaled = fdg_pet / float(fdg_thresh)

        # Resample CT to its corresponding PET resolution
        psma_ct_rs = resample_to_ref(psma_ct, psma_pet_rescaled, default_value=-1000)
        fdg_ct_rs = resample_to_ref(fdg_ct, fdg_pet_rescaled, default_value=-1000)

        # Use PSMA PET as the reference grid for the final dataset. Resample FDG PET and CT to PSMA grid.
        fdg_pet_rs_to_psma = resample_to_ref(fdg_pet_rescaled, psma_pet_rescaled, default_value=0)
        fdg_ct_rs_to_psma = resample_to_ref(fdg_ct_rs, psma_pet_rescaled, default_value=-1000)

        # Create PSMA label: 1 = tumour/disease, 2 = normal uptake
        psma_pet_arr = sitk.GetArrayFromImage(psma_pet_rescaled)
        psma_ttb_arr = sitk.GetArrayFromImage(psma_ttb)
        psma_label_arr = np.zeros_like(psma_pet_arr, dtype=np.int16)
        psma_label_arr[psma_ttb_arr > 0] = 1
        psma_label_arr[(psma_pet_arr >= 1.0) & (psma_ttb_arr == 0)] = 2
        psma_label = sitk.GetImageFromArray(psma_label_arr)
        psma_label.CopyInformation(psma_pet_rescaled)

        # Create FDG label (1 = tumour/disease, 2 = normal uptake)
        fdg_ttb_arr = sitk.GetArrayFromImage(fdg_ttb)
        fdg_label_arr = np.zeros_like(fdg_ttb_arr, dtype=np.int16)
        fdg_label_arr[fdg_ttb_arr > 0] = 1
        fdg_pet_arr = sitk.GetArrayFromImage(fdg_pet_rescaled)
        fdg_label_arr[(fdg_pet_arr >= 1.0) & (fdg_ttb_arr == 0)] = 2
        fdg_label = sitk.GetImageFromArray(fdg_label_arr)
        fdg_label.CopyInformation(fdg_pet_rescaled)
        # Resample FDG label to PSMA grid
        fdg_label_psma = resample_to_ref(fdg_label, psma_pet_rescaled, default_value=0)
        fdg_label_psma = sitk.Cast(fdg_label_psma, sitk.sitkInt16)

        # Compose combined five‑class label: 0 background, 1 PSMA tumour, 2 PSMA normal,
        # 3 FDG tumour, 4 FDG normal.
        psma_arr = sitk.GetArrayFromImage(psma_label)
        fdg_arr = sitk.GetArrayFromImage(fdg_label_psma)
        combined_arr = np.zeros_like(psma_arr, dtype=np.int16)
        # PSMA tumour and normal overwrite FDG where present
        combined_arr[psma_arr == 1] = 1
        combined_arr[psma_arr == 2] = 2
        # FDG tumour and normal only fill voxels where PSMA label is background
        combined_arr[(combined_arr == 0) & (fdg_arr == 1)] = 3
        combined_arr[(combined_arr == 0) & (fdg_arr == 2)] = 4
        combined_label = sitk.GetImageFromArray(combined_arr)
        combined_label.CopyInformation(psma_pet_rescaled)

        # Save four‑channel image. nnU‑Net expects filenames *_0000.nii.gz, *_0001.nii.gz ...
        out_chan0 = join(images_dir, f"{case_id}_0000.nii.gz")
        out_chan1 = join(images_dir, f"{case_id}_0001.nii.gz")
        out_chan2 = join(images_dir, f"{case_id}_0002.nii.gz")
        out_chan3 = join(images_dir, f"{case_id}_0003.nii.gz")
        sitk.WriteImage(psma_pet_rescaled, out_chan0)
        sitk.WriteImage(psma_ct_rs, out_chan1)
        sitk.WriteImage(fdg_pet_rs_to_psma, out_chan2)
        sitk.WriteImage(fdg_ct_rs_to_psma, out_chan3)

        # Save combined label
        sitk.WriteImage(combined_label, join(labels_dir, f"{case_id}.nii.gz"))

    # Construct a minimal dataset.json file
    json_dict = {
        "channel_names": {
            "0": "psma_pet",
            "1": "psma_ct",
            "2": "fdg_pet",
            "3": "fdg_ct"
        },
        "labels": {
            "background": 0,
            "psma_tumour": 1,
            "psma_normal": 2,
            "fdg_tumour": 3,
            "fdg_normal": 4
        },
        "numTraining": len(cases),
        "file_ending": ".nii.gz"
    }
    with open(join(output_base, "dataset.json"), "w") as f:
        f.write(json.dumps(json_dict, indent=4))
    print(f"Saved multi‑task dataset to {output_base}")


if __name__ == "__main__":
    input_dataset_folder = "data"
    create_multitask_dataset(input_dataset_folder, output_dataset_id=900)