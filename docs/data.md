# Data preparation

## Slice format

Store each high-resolution grayscale slice as a float32 array with shape (H, W) and reference intensities normalized to [0, 1]. Use .npy for normalized images or .npz for images with HU calibration. H and W must be even for ×2 reconstruction. The training patch must fit every training slice. Choose a clipping and normalization procedure appropriate for your dataset and apply it consistently across partitions before generating LR images.

A .npz slice contains:

| Field | Meaning |
|---|---|
| image | Normalized two-dimensional float32 array |
| lower_hu | HU value corresponding to normalized zero |
| width_hu | Positive HU interval corresponding to a unit change |

The calibration is HU = image × width_hu + lower_hu. For per-image min–max normalization, store that image's actual minimum and range after clipping. For a fixed normalization interval, store its lower bound and width. Use non-identifying file names and patient IDs.

~~~python
import numpy as np
hu = np.load("slice_hu.npy").astype(np.float32)
clipped = np.clip(hu, -1024, 2000)  # Choose bounds for your data.
lo, hi = float(clipped.min()), float(clipped.max())
width = max(hi - lo, 1.0)
np.savez_compressed("slice.npz", image=(clipped-lo)/width,
                    lower_hu=lo, width_hu=width)
~~~

## Manifest

Create data/slices.csv with one row per slice. Relative image paths resolve against the manifest's directory. series_id is optional and defaults to patient_id.

~~~csv
patient_id,series_id,partition,image_path
subject_a,scan_a,train,slices/a_001.npz
subject_b,scan_b,val,slices/b_001.npz
subject_c,scan_c,test,slices/c_001.npz
~~~

Include all selected slices in the manifest. Training samples use random crops, flips and right-angle rotations. Validation and testing use the listed slices without random augmentation. Set evaluation_patch_size to null to evaluate full images, or an even integer for a centered validation crop. The test command evaluates full images.

All slices and scans of a patient belong to one partition. The loader rejects patients assigned to multiple partitions and duplicate image paths. Choose partition sizes for your own dataset. A supplied split can be used directly.

To generate a split, start with a CSV containing patient_id and image_path, optionally series_id:

~~~bash
python protocol/data_split.py --input data/images.csv --output data/slices.csv --val-fraction 0.15 --test-fraction 0.15 --seed 2026
~~~

## View augmentation

The student configuration independently converts each training sample to a liver-window view with probability 0.30 (center 40 HU, width 120 HU). Conversion uses each sample's HU calibration before generating LR. Each sample contributes one view per step. The liver-view reconstruction weight is 0.50. Distillation uses the original wide views.

For calibrated CT data, provide lower_hu and width_hu in every training .npz file. For normalized images without HU calibration, set window_probability to 0 and use .npy or .npz images. Validation uses the original stored view. For inference on images already represented in the liver window, pass --window-domain liver.

## Degradation and metrics

LR images are generated from prepared HR slices using ×2 antialiased bicubic downsampling. The projection uses the same operator. PSNR uses a peak of 1 in normalized intensity coordinates. MAE and measurement residuals are reported in normalized intensity units. These units remain independent of the chosen HU interval. Predictions are evaluated without output clipping.

Metrics are averaged over slices within each series, series within each patient, then patients. The evaluation command reports its partition and checkpoint step. Test images are evaluated with a frozen checkpoint.
