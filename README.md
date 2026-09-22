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
├── parameter_ablation.py   C/D capacity anchors and validation provenance
└── module_ablation.py      module removals and consistency diagnostics
protocol/
└── data_split.py           deterministic patient-level LiTS split procedure
tests/
└── test_core_modules.py    parameter count, tensor shape, exact projection
```

This repository presents selected core modules and the main, comparison, and ablation experiment workflows. Scripts use the modules in `code/`. Patient manifests, image data, local configurations, and checkpoint files are not included.

## Standardized data protocol

### LiTS17

- Use the 131 labelled training volumes only.
- Split strictly by patient/volume: 91 train, 20 validation, 20 held-out test.
- Use the 20 validation patients for early stopping, checkpoint selection, and architecture selection.
- Table 1 reports the separate 20-patient internal test partition, labelled LiTS (test).
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

## Comparison protocol

Learned baselines are fine-tuned on CT from pretrained checkpoints using public code and each method's original training configuration. They share SPARX's LiTS patient split and training images, degradation, evaluation preprocessing, and experimental environment. The comparison script evaluates the resulting fine-tuned checkpoints. RGB baselines receive a three-channel repeat of CT and
their output is reduced by channel mean. The same degradation, crop, slice selection, metric code and
aggregation are used for every method. Checkpoint SHA-256 validation is included
in `comparison_experiment.py`.

Compared methods: Bicubic, EDSR, RCAN, OmniSR, SwinIR, HAT-S, DAT-S,
MambaIRv2, PFT-SR_class, PFT-SR_light, and OSEDiff. Third-party implementations
and weights are not redistributed here.

## Ablation protocol

Parameter anchors are C64D8, C64D14, C64D18, C96D14 and C128D14. Module
ablations are no KD, no large-field exchange, ungated exchange, and no exact
projection. Ablations share a 300k-step schedule configuration and stop at
120k optimizer updates, when the preliminary-loss weight reaches zero. They use
matched initialization, patient split, seed, loss and validation interval.
Full training of the selected C96D14 model allows up to 300k steps, with the
160k/8-validation early-stopping rule above. The 120k Full row and the main
comparison therefore correspond to different training budgets.

Architecture and checkpoint selection use LiTS validation only. HKU results do not enter the selection rule. Module selection first applies a data-consistency safety gate of
`DC-HU ≤ 1e-3`, then compares validation PSNR. The no-projection experiment is
retained to measure the physical-consistency contribution even when it fails
that gate.

### Workflow details

Training uses `val` for monitoring and checkpoint selection. In `main_experiment.py`, `--evaluate-only --checkpoint <path> --partition test` evaluates a fixed checkpoint on the held-out partition. The comparison and Bicubic scripts also accept `--partition test`. Evaluation outputs identify their partition, and learned-model LiTS evaluation records the checkpoint SHA-256. Metrics are averaged over slices within each patient, then equally over patients.

For the 120k ablation budget, set `train.max_steps: 300000`, `train.scheduler.total_steps: 300000`, and `train.stop_at_step: 120000`. Disable early stopping for this fixed budget. Full training leaves `stop_at_step` unset and retains the 160k/8-validation early-stopping rule. A step is one optimizer update.

The baseline workflow shown here contains adapters for EDSR, RCAN, OmniSR, SwinIR, DAT-S, MambaIRv2, and PFT-SR. HAT-S and OSEDiff results are included in the tables, while their adapters and third-party implementations are outside this code excerpt.


### Dataset and third-party code
remain governed by their original licenses.

## Autograd-adjoint projection

The projector in `code/util.py` evaluates the adjoint with `torch.autograd.grad` on the same bicubic downsampler (`align_corners=False`, `antialias=True`). It solves `D D^T z = residual` with conjugate gradients, then applies `D^T z`. There is no separate interpolation upsampler or dense downsampling matrix in the correction path. The existing two refinement steps and checkpoint parameter keys are preserved.

Use PyTorch **2.4.0**. FP32 defaults are relative tolerance `1e-6`, absolute tolerance `1e-7`, and at most 100 CG iterations. FP64 audit defaults are `1e-10` and `1e-12`. Non-converged solves raise an error. First-order training gradients, `no_grad`, and `inference_mode` are supported. The correction solve runs in FP32 or FP64, outside autocast.

Run the supplied validation on the experiment server before training:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
python -m pytest -q tests/test_autograd_projector.py
```

The CUDA test is skipped without a CUDA device. CPU validation on PyTorch 2.4.0 passed 11 checks, including the adjoint identity, dense-reference agreement, null-space preservation, first-order gradients, inference modes, and a 512×512 output. CUDA experiments have not been run in this update.

Use separate output directories for the new runs. Re-evaluate reconstruction and downstream metrics and remeasure latency and peak allocated memory with the new implementation. Earlier measurements are not new-autograd measurements. Complexity measurements use **256×256 LR inputs and 512×512 SR outputs**, batch size 1, FP32, 50 warm-up iterations, synchronized timing, and `torch.cuda.max_memory_allocated()`.

The frozen KD teacher uses the same C96D14 architecture as the student and is independently pretrained on LiTS for at most 300k optimizer steps, without HKU data. KD matches final SR outputs using L1 on wide-range LiTS views with weight 0.20. Sparsity refers to selected exchange depths and dilated sampling. All three experts participate in soft fusion.
