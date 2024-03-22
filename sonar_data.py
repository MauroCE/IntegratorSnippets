from pathlib import Path
import numpy as np


def get_sonar_data():
    raw_data = np.loadtxt("sonar.all-data", delimiter=",", converters={60: lambda x: 1 if x == b"R" else 0})
    response = raw_data[:, -1]
    # Preprocess predictors
    predictors = np.atleast_2d(raw_data[:, :-1])
    rescaled_predictors = 0.5 * (predictors - np.mean(predictors, axis=0)) / np.std(predictors, axis=0)
    n, p = predictors.shape
    out = np.empty((n, p + 1))
    out[:, 0] = 1.0  # intercept
    out[:, 1:] = rescaled_predictors
    predictors = out
    return predictors, response