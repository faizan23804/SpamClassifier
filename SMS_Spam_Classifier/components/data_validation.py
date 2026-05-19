import os
import sys
import pandas as pd # type: ignore
import numpy as np  # type: ignore
from SMS_Spam_Classifier.exceptions.exception import CustomException
from SMS_Spam_Classifier.logger.logging import logging



# Validation Check reference point.
BASELINE_STATS = {
    "spam_ratio_min": 0.10,      # spam should be at least 10% of data
    "spam_ratio_max": 0.20,      # spam should be at most 20% of data
    "avg_msg_length_min": 50,    # average message length in characters
    "avg_msg_length_max": 100,
    "min_records": 4000,         # dataset should have at least 4000 rows
}


class DataValidation:

    def __init__(self, X_train, X_test, y_train, y_test):
        try:
            self.X_train = X_train
            self.X_test  = X_test
            self.y_train = y_train
            self.y_test  = y_test
            self.report  = {}   # Stores all validation results
            logging.info("DataValidation initialized.")
        except Exception as e:
            raise CustomException(e, sys)

    #Check 1: Minimum record count 
    def check_minimum_records(self) -> bool:
        try:
            total = len(self.X_train) + len(self.X_test)
            passed = total >= BASELINE_STATS["min_records"]
            self.report["total_records"] = total
            self.report["min_records_check"] = "PASSED" if passed else "FAILED"
            logging.info(f"Record count check: {total} records → {'PASSED' if passed else 'FAILED'}")
            return passed
        except Exception as e:
            raise CustomException(e, sys)

    # Check 2: Label distribution drift
    def check_label_distribution(self) -> bool:
        """
        Checks if spam ratio is within expected range.
        If suddenly 50% of messages are spam, something is wrong with the data.
        """
        try:
            spam_ratio = self.y_train.mean()   # mean of 0/1 column = spam ratio
            passed = (
                BASELINE_STATS["spam_ratio_min"]
                <= spam_ratio <=
                BASELINE_STATS["spam_ratio_max"]
            )
            self.report["spam_ratio"] = round(spam_ratio, 4)
            self.report["label_distribution_check"] = "PASSED" if passed else "FAILED"
            logging.info(f"Spam ratio: {spam_ratio:.2%} → {'PASSED' if passed else 'FAILED'}")
            return passed
        except Exception as e:
            raise CustomException(e, sys)

    #Check 3: Average message length drift
    def check_message_length_drift(self) -> bool:
        """
        Detects if average message length has shifted significantly.
        A sudden change in length patterns = possible data quality issue.
        """
        try:
            avg_len = self.X_train.str.len().mean()
            passed = (
                BASELINE_STATS["avg_msg_length_min"]
                <= avg_len <=
                BASELINE_STATS["avg_msg_length_max"]
            )
            self.report["avg_message_length"] = round(avg_len, 2)
            self.report["message_length_drift_check"] = "PASSED" if passed else "FAILED"
            logging.info(f"Avg message length: {avg_len:.2f} → {'PASSED' if passed else 'FAILED'}")
            return passed
        except Exception as e:
            raise CustomException(e, sys)


    #Check 4: Vocabulary size check
    def check_vocabulary_size(self) -> bool:
        """
        Checks if the vocabulary (unique words) is within a reasonable range.
        Too few words = data too small or corrupted.
        Too many = possible encoding issues or garbage data.
        """
        try:
            all_words  = " ".join(self.X_train).split()
            vocab_size = len(set(all_words))
            passed = vocab_size >= 500   # Minimum healthy vocabulary
            self.report["vocabulary_size"] = vocab_size
            self.report["vocabulary_check"] = "PASSED" if passed else "FAILED"
            logging.info(f"Vocabulary size: {vocab_size} unique words → {'PASSED' if passed else 'FAILED'}")
            return passed
        except Exception as e:
            raise CustomException(e, sys)

    
    def initiate_data_validation(self) -> bool:
        """
        Runs all checks. Returns True only if ALL checks pass.
        Prints a clean validation report at the end.
        """
        try:
            logging.info("Data Validation started.")

            results = [
                self.check_minimum_records(),
                self.check_label_distribution(),
                self.check_message_length_drift(),
                self.check_vocabulary_size(),
            ]

            overall = all(results)
            self.report["overall_validation"] = "PASSED" if overall else "FAILED"

            # Print clean report 
            print("\n" + "="*50)
            print("       DATA VALIDATION REPORT")
            print("="*50)
            for key, value in self.report.items():
                print(f"  {key:<35} : {value}")
            print("="*50)
            print(f"  OVERALL RESULT: {self.report['overall_validation']}")
            print("="*50 + "\n")

            logging.info(f"Data Validation completed. Result: {self.report['overall_validation']}")
            return overall

        except Exception as e:
            raise CustomException(e, sys)