"""
This module provides preprocessing and modeling functions for
text classification using the Complement Naive Bayes classifier.
"""

# Importing Libraries
from collections import Counter

# This code is typically used to add the parent directory (project's root directory)
# to the Python path, allowing you to import modules
# from your project even if your script is located in a subdirectory.

from complement_naive_bayes_classifier.preprocessing import (
    load_data,
    data_segmentation,
    feature_engineering,
)
from complement_naive_bayes_classifier.training import model_training
from complement_naive_bayes_classifier.testing import model_testing
from complement_naive_bayes_classifier.visualization import model_performance


# pylint: disable=too-many-locals
def train_test(data_path):
    """
    This function loads a dataset, performs data segmentation, text data transformation,
    model training, and model testing using the Complement Naive Bayes classifier.

    Args:
    data_path (str): The file path to the dataset in CSV format.

    Returns:
    None

    Example:
    To use this function, provide the path to your dataset CSV file:
    train_test('/path/to/your/dataset.csv')
    """

    # Load the data
    data = load_data(data_path)

    # Data Segmentation
    x_train, x_test, y_train, y_test = data_segmentation(data)

    # Count Values
    value_counts = Counter(y_train)
    y_train_value_counts = Counter(y_train)
    y_test_value_counts = Counter(y_test)
    print("Values Counts before splitting:", value_counts)
    print("Values Counts after splitting: y_train:", y_train_value_counts)
    print("Values Counts after splitting: y_test", y_test_value_counts)

    # Text Data Transformation from Text to Numbers
    x_train_transf, x_test_transf = feature_engineering(x_train, x_test)

    # Print information about the transformed data
    print("Text Data Transformed to Numbers:")
    print("Training Data Shape:", x_train_transf.shape)
    print("Test Data Shape:", x_test_transf.shape)

    # Model Training
    best_model = model_training(x_train_transf, y_train)

    # Model Testing
    best_model_tested, y_test_pred, prior_probability = model_testing(best_model, x_test_transf)

    model_params = best_model.get_params()
    print("Model Parameters:", model_params)

    print("Prior Probability of Classes:")
    print("0 - Caw(negative post)", prior_probability[0])
    print("1 - Chirp(Positiv post)", prior_probability[1])

    # Model Performance Visualization Confusion of Matrix
    report, _ = model_performance(y_test, y_test_pred, best_model_tested)
    print(report)
