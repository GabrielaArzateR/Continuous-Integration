"""
This module provides a command-line interface for running the
Complement Naive Bayes Classifier on a specified file,
allowing users to analyze and classify data with the trained model.
"""

import os
import sys
import argparse

current_directory = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_directory, '..'))

# Import your `train_test` function from your module
from complement_naive_bayes_classifier.script import (  # pylint: disable=wrong-import-position
    train_test,
)


def main() -> None:
    """
    This function parses command-line arguments and runs the Complement Naive Bayes Classifier.

    It reads a file from the provided path and uses the classifier to analyze the data.

    Returns:
        None

    Example:
    To run the Complement Naive Bayes Classifier, use:
    >>> main()
    """
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(description="Complement Naive Bayes Classifier")

    # Add a command-line argument for the file path
    parser.add_argument(
        'file_path',
        type=str,
        help='Path to the file to analyze with Complement Naive Bayes Classifier',
    )

    # Parse the command-line arguments
    args = parser.parse_args()

    # You can call your train_test function here with the data object
    train_test(args.file_path)


if __name__ == '__main__':
    main()
