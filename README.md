# Algorithmic Trading

*This is an evolving collection of trading models where I continuously build and refine machine learning pipelines for market prediction.*

Each folder is a self-contained strategy with modular code and full backtesting support.

## Pipelines

- `MSFT_algo/` - Gradient Boosting classifier trained on engineered MSFT financial features; full model validation using time-series CV and backtested trading logic to assess deployability
- `SVM_pipeline/` – SVM classifier with macro features and time-series CV (tuned for tech stocks)
- `x4_pipeline/` – 4-model ensemble (LR, RF, XGBoost, LGBM); fixed parameters used to avoid overfitting from excessive tuning


## Usage

Each folder includes its own `README.md` and `requirements.txt`. Run independently.
