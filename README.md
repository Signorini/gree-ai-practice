# Green AI: Model Compression for Image Classification

It trains a CNN baseline on CIFAR-10 and evaluates three compression strategies
(pruning, INT8 quantisation, knowledge distillation) against sustainability
metrics (size, MACs, latency, energy and CO2e via CodeCarbon).

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data

CIFAR-10 is read from parquet files placed in `./data/`:

```bash
mkdir -p data && cd data
curl -L -o cifar10_train.parquet \
  "https://huggingface.co/datasets/uoft-cs/cifar10/resolve/main/plain_text/train-00000-of-00001.parquet"
curl -L -o cifar10_test.parquet \
  "https://huggingface.co/datasets/uoft-cs/cifar10/resolve/main/plain_text/test-00000-of-00001.parquet"
```

(The script's `ParquetCIFAR` loader decodes the PNG bytes once into memory.)

## Run

```bash
python green_ai_experiment.py
```

Outputs are written to `results/`:

- `metrics.json` — full configuration + per-model metrics
- `results_table.csv` — flat table of the same metrics
- `tradeoff.png` — size and accuracy-vs-compute figures

## Configuration

Edit the `Config` dataclass at the top of `green_ai_experiment.py` to change
epochs, batch size, pruning sparsity, distillation temperature, dataset subset
size, or the CodeCarbon grid region. Defaults are lightweight (a few epochs) so
the full study runs on CPU in a few minutes; increase the epoch counts for
higher absolute accuracy.

## Reproducibility notes

- Fixed seed (42) for NumPy and PyTorch.
- All models trained/evaluated on CPU for comparable energy measurement.
- Energy on Apple Silicon is estimated from CPU power models (relative
  comparisons are more reliable than absolute values).
