# MLP Boston House Prices Regression

A simple project to train and evaluate a multilayer perceptron model on the Boston House Prices data using TensorFlow, SciKeras, and Scikit-Learn. 

---

# Installation

1. Clone the repo

```bash
git clone https://github.com/yourusername/mlp-iris-classifier.git
cd mlp-boston-house-prices
```

2. Create a Conda enviornment

It is included an `environment.yml` for Conda users: 

```bash 
conda env create -f environment.yml
conda activate mlp-boston-house-prices
```

# Usage

1. Verify the dataset

The [Boston House Prices](https://www.kaggle.com/datasets/vikrishnan/boston-house-prices) from Kaggle is already included under data/sonar.csv.

2. Adjust settings

Open `config.yaml`and tweak any values you like (seed, test_size, units, etc.)

3. Run the full pipeline

```bash
python run_all.py
```

This will: 

- Train de MLP with standardized input data
- Save the model to `models` folder
- Evaluate and print train/test MAE and sample predictions

# Project Structure

```
mlp-iris-classifier/
│
├── config.yaml          # Experiment settings
├── environment.yml      # Conda environment spec
│
├── data/
│   └── housing.csv         # Boston House Prices data
│
├── models/              # (Auto-created) Trained model & params
│
├── src/
│   ├── config.py        # Loads config.yaml
│   ├── data_loader.py   # Reads & splits data
│   ├── model_builder.py # Defines the Keras MLP
│   ├── train.py         # Hyperparameter search & model saving
│   └── evaluate.py      # Loads model & prints metrics
│
└── run_all.py           # Runs train.py then evaluate.py
```

# Dependencies 

- Python 3.7+
- numpy, scikt-learn, tensorflow, scikeras, PyYAML, matplotlib.

With Conda:

```bash 
conda env create -f environment.yml
conda activate mlp-boston-house-prices
```
