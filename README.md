# MEASUREMENT-CONSISTENT MEDICAL IMAGE SUPER-RESOLUTION VIA SPARSE LARGE-FIELD EXPERTS

**SPARX** is a compact ×2 medical CT
super-resolution method with sparse large-field experts and exact measurement
projection.

## Method

SPARX consists of:

- a compact residual reconstruction backbone;
- sparse exchanges at selected depths rather than dense global processing;
- content-routed compact, axial, and dilated large-field experts;
- a learned bounded gate controlling expert-to-backbone exchange;
- dual wide-window and liver-window stems with WindowMix training;
- an autograd-adjoint bicubic affine projector enforcing `D(SR) = LR` to solver tolerance; and
- optional frozen-teacher distillation plus an annealed preliminary-image loss.

The selected C96D14 student contains 2.759 M parameters. Distillation is used
only during training; no teacher is required at inference.

## Included files

```text
code/
├── util.py                 SPARX core, large-field experts, gates, projector
├── data_utils.py           on-the-fly LR synthesis and WindowMix
└── metrics.py              PSNR, SSIM, MAE, data consistency
experiments/
├── main_experiment.py      training, validation, scheduling, KD, early stopping
├── comparison_experiment.py fine-tuned checkpoint evaluation
├── evaluate_bicubic.py     deterministic interpolation baseline
├── evaluate_external.py    patient-balanced HKU-NC/HKU-CE evaluation
├── parameter_ablation.py   model-scale comparisons and validation selection
└── module_ablation.py      module removals and consistency diagnostics
protocol/
└── data_split.py           deterministic patient-level LiTS split procedure
tests/
├── test_core_modules.py    parameter count, tensor shape, measurement projection
└── test_autograd_projector.py adjoint, gradients, consistency, CUDA checks
```

## Standardized data protocol

### LiTS17

- Use the 131 labelled training volumes only.
- Split strictly by patient/volume: 91 train, 20 validation, 20 held-out test.
- Use the 20 validation patients for early stopping, checkpoint selection, and architecture selection.
- Never split at slice level.
- Clip CT to `[-1024, 2000] HU` and losslessly encode PNG as
  `stored_uint16 = HU + 1024`.
- Evaluate eight deterministic liver-containing slices per patient, use a
  256×256 centre crop, and aggregate slice → patient → dataset.


### HKU external validation

The HKU cohort comprises 389 subjects with non-contrast (HKU-NC) and
contrast-enhanced (HKU-CE) CT. Original PNGs use the liver window `WL=40 HU, WW=120 HU`. Evaluation uses eight
deterministic slices per scan and aggregates slice → series → patient so that
patients with more scans cannot dominate.

## Main experiment protocol

| Item | Setting |
|---|---|
| Scale | ×2 |
| Backbone | C96D14 |
| Sparse exchange positions | blocks 2 and 5 |
| Batch / patch | 16 / 256×256 |
| Optimizer | AdamW |
| Initial learning rate | 5e-5 |
| Schedule | 5k warm-up, hold through 40k, cosine to 5e-7 |
| Maximum budget | 300k optimizer steps |
| Validation interval | 5k steps |
| WindowMix probability | 0.30 |
| Window-domain loss weight | 0.50 |
| KD weight | 0.20 |
| Preliminary loss | 0.10, linearly annealed to 0 at 80k–120k |
| Early stopping | start 160k, patience 8, minimum gain 0.002 dB |
| Seed | 2026 |

“300k” denotes the maximum number of optimizer updates, not epochs. The saved
state includes model, optimizer, scheduler, AMP scaler, step, best validation
PSNR, early-stopping state and timing.

### Workflow details

Training uses `val` for monitoring and checkpoint selection. In `main_experiment.py`, `--evaluate-only --checkpoint <path> --partition test` evaluates a fixed checkpoint on the held-out partition. The comparison and Bicubic scripts also accept `--partition test`. Evaluation outputs identify their partition, and learned-model LiTS evaluation records the checkpoint SHA-256. Metrics are averaged over slices within each patient, then equally over patients.

For the 120k ablation budget, set `train.max_steps: 300000`, `train.scheduler.total_steps: 300000`, and `train.stop_at_step: 120000`. Disable early stopping for this fixed budget. Full training leaves `stop_at_step` unset and retains the 160k/8-validation early-stopping rule. A step is one optimizer update.

The baseline workflow shown here contains adapters for EDSR, RCAN, OmniSR, SwinIR, DAT-S, MambaIRv2, and PFT-SR. HAT-S and OSEDiff results are included in the tables, while their adapters and third-party implementations are outside this code excerpt.


### Dataset and third-party code
remain governed by their original licenses.

## Autograd-adjoint projection

The projector in `code/util.py` evaluates the adjoint with `torch.autograd.grad` on the same bicubic downsampler (`align_corners=False`, `antialias=True`). It solves `D D^T z = residual` with conjugate gradients, then applies `D^T z`. There is no separate interpolation upsampler or dense downsampling matrix in the correction path. The existing two refinement steps and checkpoint parameter keys are preserved.

Use PyTorch **2.4.0**. FP32 defaults are relative tolerance `1e-6`, absolute tolerance `1e-7`, and at most 100 CG iterations. FP64 defaults are `1e-10` and `1e-12`. Non-converged solves raise an error. First-order training gradients, `no_grad`, and `inference_mode` are supported. The correction solve runs in FP32 or FP64, outside autocast.

Run the numerical tests before training:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
python -m pytest -q tests/test_autograd_projector.py
```

The tests cover the adjoint identity, dense-reference agreement, null-space preservation, first-order gradients, inference modes, and a 512×512 output. CUDA tests run when a CUDA device is available.

Complexity measurements use **256×256 LR inputs and 512×512 SR outputs**, batch size 1, FP32, 50 warm-up iterations, synchronized timing, and `torch.cuda.max_memory_allocated()`.

The frozen KD teacher uses the same C96D14 architecture as the student and is independently pretrained on LiTS for at most 300k optimizer steps, without HKU data. KD matches final SR outputs using L1 on wide-range LiTS views with weight 0.20. Sparsity refers to selected exchange depths and dilated sampling. All three experts participate in soft fusion.
