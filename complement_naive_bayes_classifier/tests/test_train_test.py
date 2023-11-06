"""
This module contains test cases for the complement naive Bayes classifier.
"""
from complement_naive_bayes_classifier.script import train_test


def test_train_test() -> None:
    """Tests the train_test function"""
    data_path = "data/chirper.csv"  # Replace with the path to your test data
    train_test(data_path)  # Call the train_test function with the test data


if __name__ == '__main__':
    test_train_test()
