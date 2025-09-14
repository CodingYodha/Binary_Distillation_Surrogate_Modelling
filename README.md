# AI/ML Surrogate Modeling for Binary Distillation

## 1. Project Objective

This project develops and rigorously evaluates three machine learning surrogate models (Polynomial Regression, XGBoost, and an Artificial Neural Network) to predict the distillate purity (xD) and reboiler duty (QR) of an ethanol-water distillation column.

The primary goal is not just to achieve high statistical accuracy, but to identify the most robust, physically consistent, and reliable model suitable for engineering applications like process optimization. The evaluation framework includes a challenging extrapolation test and a suite of physics-based diagnostic checks to ensure the models adhere to fundamental engineering principles.

## 2. Project Structure

```
.
├── data/
│   ├── raw/                # Place raw DWSIM .csv simulation files here
│   └── processed/          # Cleaned, featured master dataset is saved here
├── models/                 # Saved model files (.joblib, .h5) and scalers
├── results/
│   ├── plots/              # All diagnostic plots are saved here
│   └── metrics.json        # Final performance metrics for all models
├── 1_prepare_data.py       # Script to ingest, clean, and feature-engineer data
├── 2_train_models.py       # Script for hyperparameter tuning and model training
├── 3_run_diagnostics.py    # Script to evaluate models and generate all results
├── utils.py                # Helper functions for data loading, saving, etc.
├── Report.pdf              # The final, detailed project report
└── README.md               # This file
```

## 3. Installation

This project is written in Python 3.12 To install the necessary dependencies, it is recommended to use a virtual environment.

### Clone the repository:
```bash
git clone <your-repository-url>
cd <repository-name>
```

### Create and activate a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

### Install the required packages:
```bash
pip install -r requirements.txt
```

### requirements.txt
```
pandas
numpy
scikit-learn
xgboost
tensorflow
scikit-optimize
matplotlib
seaborn
```

## 4. How to Run

Follow these steps in order to reproduce the entire analysis pipeline.

### Step 0: Add Raw Data
Place the raw `.csv` files generated from your DWSIM simulations into the `data/raw/` directory. The data preparation script expects filenames that include the feed mole fraction (e.g., `simulation_data_0.2.csv`, `simulation_data_0.95.csv`).

### Step 1: Prepare the Data
This script will read all raw CSVs, combine them and save a master dataset to `data/processed/`.

```bash
python 1_prepare_data.py
```

### Step 2: Train and Tune Models
This script will load the processed data, perform the block-based train-test split, tune hyperparameters using Bayesian Optimization, and train the final models. All trained models and data scalers will be saved to the `models/` directory.

```bash
python 2_train_models.py
```

### Step 3: Run Diagnostics and Generate Results
This script is the final evaluation step. It loads the trained models, evaluates them on the unseen test set, and runs all physical consistency and diagnostic checks. All output plots will be saved to `results/plots/` and the final metrics will be saved to `results/metrics.json`.

```bash
python 3_run_diagnostics.py
```

### Step 4: Review the Analysis
The comprehensive analysis, interpretation of results, and final model recommendation are detailed in `Shivaprasad_AI_distillation_surrogate_Report.pdf`.

