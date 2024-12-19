# Cargo Data Analysis and Regression Models

This repository contains code for analyzing cargo data and building regression models to predict `PricePerWeight` for air cargo loads. The primary models used include Random Forest, LightGBM, and XGBoost, with an emphasis on robust evaluation metrics.

## Files Included
- `cargo_data_analysis.ipynb`: Notebook for exploratory data analysis (EDA) of cargo data.
- `cargo_regression_task.ipynb`: Notebook for training and evaluating regression models.
- `utils.py`: Utility functions for visualization and model evaluation.

## Key Features
- **Log-Scale Transformations**: Handle skewed data effectively.
- **Custom Evaluation Metrics**: Includes RMSE and MAPE to assess monetary and percentage errors.
- **Visualization**: Uses Plotly for interactive plots of errors and metrics distributions.

## Dependencies
Install the dependencies using the following command:
```bash
pip install -r requirements.txt
```

## Usage
1. Run the `cargo_data_analysis.ipynb` for an initial exploration of the data.
2. Use `cargo_regression_task.ipynb` to train and evaluate models.

## Notes
- Ensure the datasets and `utils.py` script are in the same directory before running the notebooks.
- The models and scripts have been tested with Python 3.10 and the dependencies listed in `requirements.txt`.
