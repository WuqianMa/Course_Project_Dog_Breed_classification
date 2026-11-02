# Does Cropping the Dog Help?

**A Study of Background Removal in Fine-Grained Breed Classification**

> Course project for the ISAO 2025–2026 *Practical Deep Learning* course, Professor Mélodie Monod.

---

### [:page_facing_up: Read the Full Report (PDF)](report.pdf)

---

## Abstract

This study asks whether automatically cropping the dog out of an image — removing background clutter with the open-vocabulary detector **Grounding-DINO** — can improve fine-grained breed classification on the **Stanford Dogs / Kaggle Dog Breed** dataset (10 222 training images, 120 breeds).

**Short answer: no.** Cropping increases test log-loss by 26% and reduces top-1 accuracy by 3.2 percentage points. The background appears to carry contextual cues that the network uses to disambiguate visually near-identical breed pairs, and foreground segmentation is not the right lever to reduce residual errors on this dataset.

## Experimental Ladder

Four runs, each differing in exactly one design lever from the previous:

| Run | Backbone | Resolution | Image | Question |
|-----|----------|-----------|-------|----------|
| E0 | ResNet-50 (frozen) + linear head | 224 | full | ImageNet features alone |
| E1 | ResNet-50 fine-tuned | 224 | full | Gain from end-to-end fine-tuning |
| E2 | ConvNeXt-Base `fb_in22k_ft_in1k_384` | 384 | full | Stronger backbone, higher resolution |
| E3 | ConvNeXt-Base `fb_in22k_ft_in1k_384` | 384 | cropped | **Principal comparison: crop vs. full** |

## Key Results

| Run | Log-loss | Top-1 | Top-5 | Macro-F1 | ECE |
|-----|----------|-------|-------|----------|-----|
| E0 (RN50 frozen) | 0.836 | 92.27% | 99.36% | 0.917 | 0.408 |
| E1 (RN50 fine-tuned) | 0.290 | 91.25% | 98.92% | 0.907 | 0.012 |
| E2 (ConvNeXt-B, full) | **0.287** | **91.88%** | **99.36%** | **0.914** | 0.018 |
| E3 (ConvNeXt-B, cropped) | 0.361 | 88.70% | 99.17% | 0.882 | 0.024 |

*E1–E3 are reported after temperature scaling; E0 uses raw probabilities.*

## Per-Class Breakdown

The aggregate result masks a heterogeneous pattern:
- **17 breeds gained** accuracy from cropping — centrally framed dogs where cropping yields a sharper crop of texture and silhouette.
- **50 breeds lost** accuracy — these cluster on look-alike pairs (*Staffordshire Bullterrier* vs. *American Staffordshire Terrier*; *Siberian Husky* vs. *Malamute*; *Greater Swiss Mountain Dog* vs. *Appenzeller*), where the full-image model relies on contextual cues that disappear inside a tight bounding box.
- **53 breeds were unchanged.**

## Repository Structure

```
├── report.pdf              ← Full written report
├── code.ipynb              ← End-to-end notebook (Colab A100)
├── requirements.txt        ← Pinned dependencies
└── output/
    ├── figures/            ← All plots used in the report
    ├── results/            ← CSV metrics (headline, per-class, confusion pairs)
    └── experiment_history/ ← Training curves for every run & sweep
```

## Reproducing

```bash
pip install -r requirements.txt
```

The notebook was developed on **Google Colab with an A100 GPU**. E0/E1 take ~5 min each; E2/E3 take ~45 min each. The dataset is downloaded automatically from Kaggle inside the notebook.

## Author

**Wuqian Ma** — ISAO 2025–2026
