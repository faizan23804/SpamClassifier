import os
import sys
import pickle
import pandas as pd  # type: ignore
from sklearn.naive_bayes import MultinomialNB, BernoulliNB         # type: ignore
from sklearn.linear_model import LogisticRegression    # type: ignore
from sklearn.ensemble import RandomForestClassifier    # type: ignore
from sklearn.metrics import (                          # type: ignore
    accuracy_score, precision_score,
    recall_score, f1_score
)
from SMS_Spam_Classifier.exceptions.exception import CustomException
from SMS_Spam_Classifier.logger.logging import logging

MODEL_PATH = "models/spam_classifier.pkl"


class ModelTrainer:

    def __init__(self, X_train, X_test, y_train, y_test):
        try:
            self.X_train = X_train
            self.X_test  = X_test
            self.y_train = y_train
            self.y_test  = y_test

            self.models = {
                "MNB": MultinomialNB(alpha=0.1, fit_prior=False),
                "BNB": BernoulliNB(alpha=2.0,fit_prior=False),
                "Logistic Regression": LogisticRegression(max_iter=1000),
                "Random Forest": RandomForestClassifier(n_estimators=100,max_depth=10,min_samples_split=50,n_jobs=-1)
            }
            logging.info("ModelTrainer initialized.")
        except Exception as e:
            raise CustomException(e, sys)

    def evaluate_model(self, model) -> dict:
        """Trains one model and returns its metrics."""
        try:
            model.fit(self.X_train, self.y_train)
            y_pred = model.predict(self.X_test)
            return {
                "accuracy": f"{accuracy_score(self.y_test, y_pred):.2%}",
                "precision": f"{precision_score(self.y_test, y_pred):.2%}",
                "recall":    f"{recall_score(self.y_test, y_pred):.2%}",
                "f1_score":  f"{f1_score(self.y_test, y_pred):.2%}",
            }
        except Exception as e:
            raise CustomException(e, sys)

    def train_and_select_best(self):
        """
        Trains all models
        saves the best model to disk.
        """
        try:
            logging.info("Model training started.")
            results = {}

            print("\n" + "="*55)
            print("           MODEL EVALUATION REPORT")
            print("="*55)

            for name, model in self.models.items():
                metrics = self.evaluate_model(model)
                results[name] = {"model": model, "metrics": metrics}
                print(f"\n  {name}")
                for metric, value in metrics.items():
                    print(f"{metric:<10}: {value}")

            print("="*55 + "\n")

            # Select best model by Precision
            best_name = max(
                results,
                key=lambda name: results[name]["metrics"]["precision"]
            )
            best_model   = results[best_name]["model"]
            best_metrics = results[best_name]["metrics"]

            logging.info(f"Best model: {best_name} | Precision: {best_metrics['precision']}")
            print(f"Best Model Selected: {best_name}")
            print(f"Precision: {best_metrics['precision']}\n")

            # Save best model
            os.makedirs("models", exist_ok=True)
            with open(MODEL_PATH, "wb") as f:
                pickle.dump(best_model, f)
            logging.info(f"Best model saved to {MODEL_PATH}")

            return best_model, best_metrics

        except Exception as e:
            raise CustomException(e, sys)