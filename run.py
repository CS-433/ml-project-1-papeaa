import numpy as np
from proj1_helpers import *
from data_processing import *

def main():
    # Import Data
    DATA_TRAIN_PATH = 'data/train.csv'
    DATA_TEST_PATH = 'data/test.csv'
    y, tX, ids = load_csv_data(DATA_TRAIN_PATH)
    _, tX_test, ids_test = load_csv_data(DATA_TEST_PATH)

    # Data Processing
    tX, tX_test = fix_nonexisting(tX, tX_test)
    tX, tX_test = standardize(tX, tX_test)

    # Train model with optimized parameter
    loss, w = ridge_regression(y, tX, 0.005736152510448681)

    # Write Predictions to file
    OUTPUT_PATH = 'data/final-submission.csv'
    y_pred = predict_labels(w, tX_test)
    create_csv_submission(ids_test, y_pred, OUTPUT_PATH)

if __name__ == "__main__":
    main()

