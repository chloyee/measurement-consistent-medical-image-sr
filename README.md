# SPARX

**Measurement-Consistent Medical Image Super-Resolution via Sparse Large-Field Experts**

SPARX combines content-adaptive spatial experts with measurement-consistent reconstruction for ×2 CT super-resolution. The C96D14 network uses 96 channels, 14 residual blocks and expert exchange after blocks 2 and 5, with **2.759 M parameters**. Two projection refinements correct the observation residual through the adjoint of the bicubic degradation operator.

## Installation

Use Python 3.11 and PyTorch 2.4.0. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q
```

For a CUDA 12.1 environment, install the matching PyTorch wheel before the remaining requirements:

```bash
python -m pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
```

## Prepare your data

Provide your own CT slices and a CSV manifest. Each slice is a two-dimensional normalized NumPy array. Patient-level partitions may use any suitable dataset size. See [Data preparation](docs/data.md) for the format, optional HU metadata, and a configurable patient split utility.

## Train

Set the manifest path and patch size in the configuration files, then train the teacher and student:

```bash
python experiments/main_experiment.py --config configs/teacher.yaml
python experiments/main_experiment.py --config configs/student.yaml
```

Both configurations use C96D14 and a maximum of 300,000 optimizer steps with batch size 16. The independently trained teacher is frozen during student training. The student combines projected-image reconstruction, an annealed preliminary-image loss, and wide-view output distillation with weight 0.20. Details and checkpoint handling are in [Training and evaluation](docs/training.md).

## Test and reconstruct

Evaluate the selected student checkpoint on your held-out images:

```bash
python experiments/evaluate.py --checkpoint outputs/student/best.pt --manifest data/slices.csv --output outputs/test.json
```

Reconstruct one normalized low-resolution image:

```bash
python experiments/evaluate.py --checkpoint outputs/student/best.pt --input data/lr.npy --output outputs/sr.npy
```

The output retains floating-point intensities without clipping. Inference uses batch size 1 and requires no teacher.

## Method

Compact, axial and dilated experts capture complementary spatial structure. A pixel-wise router softly combines all three experts, and a learned gate controls residual exchange. Sparse placement and dilated sampling provide large-field processing within a compact backbone.

The projector computes the downsampler's adjoint with an autograd vector–Jacobian product. Conjugate gradients solve the observation-space system, and corrections remain in the range of the adjoint. LR synthesis and projection share bicubic interpolation, coordinates, antialiasing and boundary behavior under PyTorch 2.4.0.

## Repository structure

- `code/`: model, data loading and metrics.
- `configs/`: teacher and student training settings.
- `experiments/`: training, evaluation and single-image reconstruction.
- `protocol/`: configurable patient-level partitioning.
- `tests/`: numerical projection and training pipeline tests.
- `docs/`: data and workflow guides.
