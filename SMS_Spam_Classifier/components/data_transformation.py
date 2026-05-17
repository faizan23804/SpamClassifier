import os
import sys
import pickle
import numpy as np # type: ignore
import pandas as pd # type: ignore
from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
from imblearn.over_sampling import SMOTE # type: ignore
from SMS_Spam_Classifier.exceptions.exception import CustomException
from SMS_Spam_Classifier.logger.logging import logging

# Saved vectorizer path — so app.py can load it for predictions
VECTORIZER_PATH = "models/tfidf_vectorizer.pkl"


class DataTransformation:

    def __init__(self, X_train, X_test, y_train, y_test):
        try:
            self.X_train = X_train
            self.X_test  = X_test
            self.y_train = y_train
            self.y_test  = y_test

            # TF-IDF config:
            # max_features=3000  
            # ngram_range=(1,2)  
            # sublinear_tf=True  → applies log normalization to term frequency
            self.vectorizer = TfidfVectorizer(
                max_features=3000,
                ngram_range=(1, 2),
                sublinear_tf=True
            )
            self.smote = SMOTE(random_state = 42)
            logging.info("DataTransformation initialized.")
        except Exception as e:
            raise CustomException(e, sys)

    def transform(self):
        """
        Fits TF-IDF on training data.
        Transforms both train and test using the fitted vectorizer.
        Fits SMOTE on training data only
        Saves the vectorizer to disk for use in Streamlit app.
        """
        try:
            
            logging.info("Fitting TF-IDF on training data.")

            # fit_transform on train
            X_train_tfidf = self.vectorizer.fit_transform(self.X_train)

            # transform on test — uses vocabulary learned from train
            X_test_tfidf  = self.vectorizer.transform(self.X_test)

            logging.info(
                f"TF-IDF complete. "
                f"Train shape: {X_train_tfidf.shape} | "
                f"Test shape: {X_test_tfidf.shape}"
            )
            logging.info("Resampling data using SMOTE")

            #fit_resample on train
            X_train_smote, y_train_smote = self.smote.fit_resample(X_train_tfidf, self.y_train)


            # Save vectorizer so Streamlit app can vectorize new input
            os.makedirs("models", exist_ok=True)
            with open(VECTORIZER_PATH, "wb") as f:
                pickle.dump(self.vectorizer, f)
            logging.info(f"Vectorizer saved to {VECTORIZER_PATH}")

            return X_train_smote, X_test_tfidf, y_train_smote, self.y_test

        except Exception as e:
            raise CustomException(e, sys)