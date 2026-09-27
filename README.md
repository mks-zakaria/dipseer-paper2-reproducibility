# DIPSEER Paper II — Reproducibility Package

This repository contains all Kaggle notebook outputs for **Paper II: "Multimodal Large Language Models versus Task-Specific Models for Ordinal Student Attention Recognition"**.

## Notebook Runs & Outputs

| Notebook | Kaggle Link | Purpose | Key Output |
|----------|-------------|---------|------------|
| **2a** | [dipseer-paper-2a-zero-shot-few-shot-vlm](https://www.kaggle.com/code/zakariamakhas/dipseer-paper-2a-zero-shot-few-shot-vlm) | Zero-shot & few-shot prompting of Qwen2-VL-7B | `paper2a_comparison.json`, `paper2a_fold{0-4}.json` |
| **2b** | [dipseer-paper-2b-lora](https://www.kaggle.com/code/zakariamakhas/dipseer-paper-2b-lora) | LoRA fine-tuning (fold 0 only — OOM truncated) | `paper2_comparison.json`, `paper2b_fold0.json` |
| **FullMetrics** | [dipser-mllm-fullmetrics](https://www.kaggle.com/code/zakariamakhas/dipser-mllm-fullmetrics) | Full metrics + confusion matrices for all conditions | `mllm_metrics.json`, `mllm_confusion.png` |
| **2d** | [dipseer-paper-2d-all-conditions](https://www.kaggle.com/code/zakariamakhas/dipseer-paper-2d-all-conditions) | **Corrected 5-fold LoRA + zero/few-shot on 1,248-frame slice** | `lora_5fold_summary.json` (QWK 0.282±0.236), `lora_fold{0-4}.json` |
| **2e** | [dipseer-paper-2e-finetune-sweep](https://www.kaggle.com/code/zakariamakhas/dipseer-paper-2e-finetune-sweep) | LoRA hyperparameter sweep (rank, target modules, CORAL head) | `sweep_*.json` |
| **2f** | [dipseer-paper-2f-baseline-matched](https://www.kaggle.com/code/zakariamakhas/dipseer-paper-2f-baseline-matched) | Task-specific baseline re-scored on corrected protocol | `baseline_A_summary.json`, `baseline_B_summary.json` |

## Corrected Protocol Results (Paper II)

The **corrected protocol** (2d + 2f) gives the final numbers in the paper:

| Condition | QWK | MAE | Macro-F1 | Accuracy |
|-----------|-----|-----|----------|----------|
| LoRA (5-fold) | **0.282 ± 0.236** | 0.345 | 0.428 | 0.659 |
| Task-specific baseline (5-fold) | **0.230 ± 0.154** | 0.371 | 0.406 | 0.632 |
| Zero-shot (1,248 frames) | 0.137 | 0.454 | 0.353 | 0.556 |
| Few-shot (1,248 frames) | 0.020 | 1.368 | 0.068 | 0.060 |

**Key finding**: Adapted MLLM (LoRA) ≈ Task-specific baseline; unadapted MLLM fails.

## Directory Structure

```
kaggle_runs/
├── paper2a_results/          # Zero-shot & few-shot (per-fold)
├── paper2b_results/          # LoRA fold 0 only (OOM)
├── paper2d_results/          # Corrected 5-fold LoRA + prompting
├── paper2e_results/          # Hyperparameter sweep
├── paper2f_results/          # Task-specific baseline re-scored
├── mllm_metrics.json         # Full metrics (all conditions)
├── mllm_confusion.png        # Confusion matrices
└── *.log                     # Kernel execution logs
```

## Key Configuration (pinned across runs)

- **Model**: Qwen/Qwen2-VL-7B-Instruct (4-bit NF4)
- **LoRA**: rank 8, q/v projections, 2.52M trainable params (0.05%)
- **Training**: 2 epochs, lr 2e-4, effective batch 4 (micro=1 × accum=4)
- **Data**: DIPSER slice, 32 frames/subject, 1,248 total, 20 subjects
- **Split**: GroupKFold 5, seed=0, subject-disjoint
- **Metric**: Quadratic Weighted Kappa (QWK) primary

## Reproducing

```bash
# Push corrected 5-fold LoRA (requires T4×2):
cd dipseer/paper2/notebooks
kaggle kernels push -p . --accelerator NvidiaTeslaT4
```

> **Note**: Default Kaggle accelerator is P100 (sm_60) — **will fail** for 4-bit NF4 (needs sm_75+). Must explicitly use `--accelerator NvidiaTeslaT4`.

## Citation

If you use these results, please cite:

```bibtex
@inproceedings{makhkhas2026mllm,
  title={Multimodal Large Language Models versus Task-Specific Models for Ordinal Student Attention Recognition},
  author={Makhkhas, Zakaria and Rifai, Amal and Sadiq, Abdelalim},
  booktitle={ICAIS 2026},
  year={2026}
}
```