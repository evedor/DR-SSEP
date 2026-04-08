# DR-SSEP

Dual-Reservoir Slow-Slip Event Predictor.

This repository predicts GNSS slow-slip displacement sequences (`E/N` components) using a dual-reservoir workflow.

## Pipeline

1. Load station time series from `data/*.npy`.
2. Extract high-dimensional dynamic features with `Reservoir_rnn` / `Reservoir_fnn`.
3. Build local linear mappings with delay-embedding style reconstruction.
4. Perform multi-step forecasting.
5. Save prediction tensors and evaluation plots in `results/`.

## Project Files

- `main_shallow_newzealand.ipynb`: shallow-station experiment
- `main_deep_newzealand.ipynb`: deep-station experiment
- `reservoir_rnn.py`, `reservoir_fnn.py`: reservoir feature modules
- `utils.py`: preprocessing + helper functions
- `data/`: input arrays
- `results/`: saved outputs

## Setup

Recommended: Python `3.10`.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install numpy pandas scipy matplotlib scikit-learn joblib tqdm jupyter torch fastdtw
```

## How To Run

```bash
jupyter notebook
```

Then run one notebook top-to-bottom:

- Shallow case: `main_shallow_newzealand.ipynb`
- Deep case: `main_deep_newzealand.ipynb`

Default key parameters in both notebooks:

- `predict_len = 22`
- `train_len = 35`

## Outputs

Typical outputs include:

- Prediction tensors: `results/*/pt*.npy`
- Comparison plots: `predict.pdf`
- Metrics/analysis plots: `nrmse.pdf`, `pcc.pdf`, `distance.pdf`, and station-level figures

