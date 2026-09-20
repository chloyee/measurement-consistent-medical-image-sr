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
- an exact bicubic affine projector enforcing `D(SR) = LR` numerically; and
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
├── comparison_experiment.py official-checkpoint baseline evaluation
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

All learned baselines use author-released ×2 checkpoints in a
frozen zero-shot setting. RGB baselines receive a three-channel repeat of CT and
their output is reduced by channel mean. No baseline in the formal table is
retrained on LiTS. The same degradation, crop, slice selection, metric code and
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

The capacity rule considers LiTS accuracy, both external domains, and parameter
cost. Module selection first applies a data-consistency safety gate of
`DC-HU ≤ 1e-3`, then compares validation PSNR. The no-projection experiment is
retained to measure the physical-consistency contribution even when it fails
that gate.

### Workflow details

Training uses `val` for monitoring and checkpoint selection. In `main_experiment.py`, `--evaluate-only --checkpoint <path> --partition test` evaluates a fixed checkpoint on the held-out partition. The comparison and Bicubic scripts also accept `--partition test`. Evaluation outputs identify their partition, and learned-model LiTS evaluation records the checkpoint SHA-256. Metrics are averaged over slices within each patient, then equally over patients.

For the 120k ablation budget, set `train.max_steps: 300000`, `train.scheduler.total_steps: 300000`, and `train.stop_at_step: 120000`. Disable early stopping for this fixed budget. Full training leaves `stop_at_step` unset and retains the 160k/8-validation early-stopping rule. A step is one optimizer update.

The baseline workflow shown here contains adapters for EDSR, RCAN, OmniSR, SwinIR, DAT-S, MambaIRv2, and PFT-SR. HAT-S and OSEDiff results are included in the tables, while their adapters and third-party implementations are outside this code excerpt.

## Results

Reported PSNR, SSIM, MAE and parameter counts below follow the manuscript
precision. Data-consistency values retain the additional precision of this
review repository.

### SPARX cross-dataset performance

| Dataset | PSNR (dB) ↑ | SSIM ↑ | MAE ↓ | Data consistency ↓ |
|---|---:|---:|---:|---:|
| LiTS (test) | 47.140 | 0.9866 | 10.80 HU | **3.358e-05 HU** |
| HKU-NC | 23.926 | 0.9005 | 10.036 (8-bit) | **2.795e-06 (8-bit)** |
| HKU-CE | 26.511 | 0.9441 | 7.220 (8-bit) | **4.210e-06 (8-bit)** |

### LiTS (test) comparison

| Method | Protocol | Params | PSNR ↑ | SSIM ↑ |
|---|---|---:|---:|---:|
| Bicubic | deterministic | 0 | 42.890 | 0.9781 |
| EDSR | official zero-shot | 40.730 M | 46.673 | 0.9858 |
| RCAN | official zero-shot | 15.445 M | 46.694 | 0.9858 |
| OmniSR | official zero-shot | 0.785 M | 46.016 | 0.9836 |
| SwinIR | official zero-shot | 11.752 M | 46.714 | 0.9858 |
| HAT-S | official zero-shot | 9.473 M | 46.761 | 0.9859 |
| DAT-S | official zero-shot | 11.064 M | 46.741 | 0.9858 |
| MambaIRv2 | official zero-shot | 22.903 M | 46.767 | 0.9858 |
| PFT-SR_class | official zero-shot | 19.624 M | 46.775 | 0.9859 |
| PFT-SR_light | official zero-shot | 0.776 M | 46.732 | 0.9858 |
| OSEDiff | official cross-degradation | 1,775 M | 29.400 | 0.8416 |
| SPARX (Ours) | LiTS-trained | 2.759 M | **47.140** | **0.9866** |

### HKU external comparison

| Method | NC PSNR ↑ | NC SSIM ↑ | CE PSNR ↑ | CE SSIM ↑ |
|---|---:|---:|---:|---:|
| Bicubic | 21.575 | 0.8128 | 22.281 | 0.8643 |
| EDSR | 23.449 | 0.8925 | 25.551 | 0.9357 |
| RCAN | 23.449 | 0.8923 | 25.479 | 0.9351 |
| OmniSR | 23.486 | 0.8915 | 25.485 | 0.9346 |
| SwinIR | 23.287 | 0.8917 | 25.103 | 0.9340 |
| HAT-S | 23.415 | 0.8940 | 25.354 | 0.9364 |
| DAT-S | 23.510 | 0.8945 | 25.495 | 0.9361 |
| MambaIRv2 | 23.573 | 0.8956 | 25.685 | 0.9375 |
| PFT-SR_class | 22.726 | 0.8936 | 23.253 | 0.9307 |
| PFT-SR_light | 23.343 | 0.8913 | 25.268 | 0.9339 |
| OSEDiff | 17.574 | 0.4650 | 17.345 | 0.5472 |
| SPARX (Ours) | **23.926** | **0.9005** | **26.511** | **0.9441** |

### Parameter ablation: matched 120k protocol

| Configuration | Params | LiTS PSNR | HKU-NC PSNR | HKU-CE PSNR | Decision |
|---|---:|---:|---:|---:|---|
| C64D8 | 0.787 M | 47.0435 | 23.8216 | 26.3061 | small anchor |
| C64D14 | 1.231 M | 47.0859 | 23.8683 | 26.4026 | depth anchor |
| C64D18 | 1.526 M | 47.0991 | 23.8788 | 26.4231 | depth saturated |
| C96D14 | 2.759 M | 47.1365 | **23.9037** | **26.4630** | selected |
| C128D14 | 4.894 M | **47.1665** | 23.7825 | 26.2081 | larger anchor |

### Module ablation: C96D14, matched 120k protocol

| Configuration | LiTS PSNR | HKU-NC PSNR | HKU-CE PSNR | LiTS consistency |
|---|---:|---:|---:|---:|
| Full | **47.1365** | 23.9037 | 26.4630 | 3.351e-05 HU |
| No KD | 47.1136 | **23.9104** | 26.4723 | 3.357e-05 HU |
| No large-field exchange | 47.1199 | 23.9017 | 26.4665 | 3.354e-05 HU |
| Ungated exchange | 47.1352 | 23.9076 | **26.4748** | 3.355e-05 HU |
| No exact projection | 47.1290 | 23.8871 | 26.4259 | 1.504e-01 HU |

The tables report the manuscript results. Architecture and checkpoint selection use LiTS validation, independently of HKU reporting. Exact projection enforces measurement consistency.


Code is provided for confidential academic review. Dataset and third-party code
remain governed by their original licenses.
