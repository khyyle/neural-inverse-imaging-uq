# Neural Inverse Imaging UQ

Research code for studying uncertainty in coordinate-based neural image reconstructions under CT, cropped Fourier, and VLBI forward models. We compare deformation-based BayesRays with parameter-space Laplace and deep ensembles, and use Fourier-domain diagnostics to separate model uncertainty from what the measurements constrain.

## Repository layout

```text
.
├── src/bhuq/
│   ├── forward/          # CT / Fourier-crop / VLBI measurement models
│   ├── models/           # coordinate-based neural image models
│   ├── uq/               # deformation, Laplace, ensemble, and Fourier uncertainty quantification
│   ├── visualization/    # maps, sparsification curves, and composed figures
│   ├── model_training.py # common in-memory and saved-model workflow
│   ├── evaluation.py     # UQ orchestration and validation metrics
│   └── runs.py           # experiment artifacts and provenance
├── scripts/              # model-training entry points
├── experiments/          # uncertainty-evaluation entry points
├── data/                 # source images and telescope arrays
└── archive/              # original notebooks, checkpoints, and figures
```

## Getting started

Install the pinned Python 3.12 environment with
[uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

For example, train a five-member 64 x 64 vlbi ensemble:

```bash
uv run python scripts/train_vlbi_model.py \
  data/images/avery_sgra_eofn.txt \
  --pixel-count 64 \
  --seeds 0 1 2 3 4
```

To run uncertainty quantification methods on the saved model bundle using a sample experiment:

```bash
uv run python experiments/vlbi_image.py \
  model_cache/vlbi/avery_sgra_eofn/eht2017/64x64/neural_image/<model-id>
```
