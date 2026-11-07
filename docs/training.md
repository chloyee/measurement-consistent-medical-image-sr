# Training and evaluation

## Teacher pretraining

Edit configs/teacher.yaml to set data.manifest, data.patch_size and output_dir. Run:

~~~bash
python experiments/main_experiment.py --config configs/teacher.yaml
~~~

The teacher uses the C96D14 architecture with two expert exchanges and two projection refinements. It trains from initialization on the training partition, with projected-image L1 reconstruction and an annealed preliminary-image L1 loss. The preliminary weight starts at 0.10 and decreases linearly from step 80,000 to zero at step 120,000.

## Student training

Set train.teacher_checkpoint in configs/student.yaml to the teacher's best.pt, and use the same training patients and model configuration:

~~~bash
python experiments/main_experiment.py --config configs/student.yaml
~~~

The student trains with a frozen, independently pretrained teacher of identical architecture. Final SR-output L1 distillation has weight 0.20 and uses original wide-view samples. Projected and preliminary reconstruction losses use pixel-mean L1, aggregated across samples with weight 0.50 for liver-window views and 1 for original views, divided by the sum of weights. The liver-window probability is 0.30 per sample. See [Data preparation](data.md) for HU calibration.

## Optimization

| Setting | Default |
|---|---|
| Batch size | 16 |
| Optimizer | AdamW |
| Initial learning rate | 5e-5 |
| Warm-up | 5,000 steps |
| Constant-rate phase | Through step 40,000 |
| Cosine minimum learning rate | 5e-7 |
| Maximum updates | 300,000 |
| Validation interval | 5,000 steps |
| Early stopping | From step 160,000, patience 8 validations, PSNR gain 0.002 dB |

Each step performs one optimizer update without gradient accumulation. The training implementation uses FP32. Set the patch size, workers and device for your hardware. A complete batch must fit in the training partition.

Checkpoints are written at validation intervals and the final step. best.pt tracks the highest validation PSNR. last.pt stores model, optimizer, step, early-stopping state and configuration. Resume with the unchanged configuration:

~~~bash
python experiments/main_experiment.py --config configs/student.yaml --resume
~~~

## Evaluation

~~~bash
python experiments/evaluate.py --checkpoint outputs/student/best.pt --manifest data/slices.csv --output outputs/test.json
~~~

The default partition is test. Use --partition val for validation reporting. Evaluation reads a fixed checkpoint and does not update model parameters or select checkpoints. Patient identifiers are checked against the checkpoint's training patients. Reported metrics use equal patient weighting.

For a low-resolution slice:

~~~bash
python experiments/evaluate.py --checkpoint outputs/student/best.pt --input data/lr.npy --output outputs/sr.npy
~~~

The saved array is the floating-point ×2 output. The teacher is not loaded for testing or reconstruction. Add --device cpu to run on CPU. Image intensities follow the preparation convention described in [Data preparation](data.md).

## Numerical checks

~~~bash
python -m pytest -q
~~~

Tests check the adjoint identity, conjugate-gradient solves, null-space preservation, first-order gradients, inference modes, parameter count, split isolation, and a small teacher-to-student training workflow. CUDA-specific tests run when a CUDA device is available.
