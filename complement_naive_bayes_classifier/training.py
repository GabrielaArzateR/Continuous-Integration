"""
This module provides functions for training a machine learning model on the training data.
"""
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.naive_bayes import ComplementNB


def model_training(x_train_transf: csr_matrix, y_train: pd.Series) -> ComplementNB:
    """
    Train a machine learning model using the training data.

    Args:
        x_train_transf (scipy.sparse.csr_matrix): Transformed training input data.
        y_train (pandas.Series): Training target data.

    Returns:
        ComplementNB: The trained machine learning model.

    Example:
        To train a model, use:
        >>> trained_model = model_training(x_train_transf, y_train)
    """
    ### 5- Model Training
    best_model = ComplementNB()
    best_model.fit(x_train_transf, y_train)

    return best_model
