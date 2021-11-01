# Project 1: Finding Higgs Boson's Signature Trace

The EPFL Machine Learning course's first project aims to solve the [Higgs Boson classification problem](https://www.aicrowd.com/challenges/epfl-machine-learning-higgs) using regression techniques. The given report and code explain our machine learning procedure in detail.

## Getting Started

Please insert `train.csv` and `test.csv` into `/data`! To produce the final result, run the `run.py` file with `python run.py`. Python3 and the libraries NumPy and matplotlib are required.

### Code Description
In the following, we describe the main functionality of the files in the repository.

```
.
├── README.md
├── data    <- Please insert train.csv and test.csv here!
│   ├── final-submission.csv
├── data_processing.py
├── experiments.ipynb
├── figs    <- Figure Outputs from cross validation
│   ├── cross_validation_least_degree.pdf
│   ├── cross_validation_log_degree.pdf
│   ├── cross_validation_ridge_lamdas.pdf
│   └── cross_validation_ridge_lamdas.pdf
├── implementations.py
├── proj1_helpers.py
└── run.py
```

### `dataprocessing.py`
Includes all functions for data splitting, cleaning, standadization and feature expension.
* *split_data*
* *build_k_indices*
* *build_poly*
* *fix_nonexisting*
* *standardize*
* *funtions for cross validations*

---

### `experiments.ipynb`

That jupyter notebook reflects the score improvement steps, which are :
* Cross-Validation for Least Square to find optimal degree for feature expansion
* Cross-Validation for Ridge Regression to find optimal λ
* Cross-Validation for Logistic Regression to find optimal degree for feature expansion
* Cross-Validation for Regularized Logistic Regression to find optimal λ
* Run of all Method in `implementations.py` to compare results without Feature Expansion
* Run of all Method in `implementations.py` to compare results with Feature Expansion
---

### `implementations.py`

List of functions that had to be implemented in the framework of the project.

* *least_squaresGD, least_squaresSGD, least_squares, ridge_regression*
* *logistic_regression, reg_logistic_regression*

---

### `proj1_helpers.py`

Functions *load_csv_data*, *predict_labels* and *create_csv_submission* that were given as helpers. 

---

### `run.py`

Running that file produces the exact .csv predictions used in the best submission on [AIrowd](https://www.aicrowd.com/challenges/epfl-machine-learning-higgs/submissions/164057).

## Authors

* Aamir Shakir
* Patricia Brandl
* Perrine Vantalon
