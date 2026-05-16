import os
import sys
import pandas as pd # type: ignore
import spacy  # type: ignore
from sqlalchemy import text  # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from  SMS_Spam_Classifier.database.sql_client import SQLClient
from  SMS_Spam_Classifier.exceptions.exception import CustomException
from  SMS_Spam_Classifier.logger.logging import logging
from  SMS_Spam_Classifier.constant import TABLE_NAME


class DataIngestion:
        

    def __init__(self):
        try:
            # SQLClient is a singleton — no new connection opens here
            # if it was already created
            self.sql_client = SQLClient()
            self.engine = self.sql_client.get_engine()

            self.nlp = spacy.load('en_core_web_sm')
            logging.info("DataIngestion component initialized.")
        except Exception as e:
            raise CustomException(e, sys)


    def fetch_data_from_SQL(self) -> pd.DataFrame:
        """
        Pulls the raw dataset from PostgreSQL into a DataFrame.

        pd.read_sql() uses the SQLAlchemy engine directly.
        This is equivalent to pulling from a MongoDB collection
        using self.collection.find({}) in your NYC Taxi project.
        """
        try:
            query = f"SELECT * FROM {TABLE_NAME}"

            df = pd.read_sql(
                sql=text(query),   # text() wraps raw SQL — best practice with SQLAlchemy 2.x
                con=self.engine.connect() # type: ignore
            )

            logging.info(f"Data fetched from PostgreSQL. Shape: {df.shape}")
            return df

        except Exception as e:
            raise CustomException(e, sys)


    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cleans the raw DataFrame:
        - Removes null rows
        - Removes duplicate SMS messages
        - Encodes label column: ham → 0, spam → 1
        """
        try:
            before = len(df)

            df.dropna(inplace=True)
            df.drop_duplicates(keep="first", inplace=True)

            after = len(df)
            logging.info(
                f"Cleaning complete. "
                f"Removed {before - after} rows. "
                f"Remaining: {after}"
            )


            # Sanity check — if any label failed to map, it becomes NaN
            if df['label'].isnull().sum() > 0:
                raise ValueError("Label encoding failed — unexpected values in 'label' column.")

            return df

        except Exception as e:
            raise CustomException(e, sys)
        
    
    def preprocess_text(self, text: str) -> str:
        try:
            doc = self.nlp(text)   # ✅ Uses already-loaded model
            tokens = []
            for token in doc:
                if token.is_stop or token.is_punct or token.is_space:
                    continue
                if not token.text.isalnum():
                    continue
            tokens.append(token.lemma_.lower())
            return " ".join(tokens)
        except Exception as e:
            raise CustomException(e, sys)


    def split_data(self, df: pd.DataFrame):
        """
        Splits cleaned data into train and test sets.

        stratify=y → Ensures both splits have same ham/spam ratio.
        Critical for imbalanced datasets like this one (~87% ham, ~13% spam).
        """
        try:
            X = df['message']   # Features — raw SMS text
            y = df['label']     # Target — 0 (ham) or 1 (spam)

            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=0.2,
                random_state=42,
                stratify=y       # Preserve class distribution
            )

            logging.info(
                f"Train size: {len(X_train)} | "
                f"Test size: {len(X_test)}"
            )
            return X_train, X_test, y_train, y_test

        except Exception as e:
            raise CustomException(e, sys)


    def initiate_data_ingestion(self):
        try:
            logging.info("Data Ingestion started.")

            df = self.fetch_data_from_SQL()
            df = self.clean_data(df)

            logging.info("Text preprocessing started. This may take a minute...")

            # ✅ nlp.pipe() processes all messages in batches — far faster than .apply()
            messages = df["message"].tolist()
            processed = []

            for doc in self.nlp.pipe(messages, batch_size=500, disable=["ner", "parser"]):
                tokens = []
                for token in doc:
                    if token.is_stop or token.is_punct or token.is_space:
                        continue
                    if not token.text.isalnum():
                        continue
                    tokens.append(token.lemma_.lower())
                processed.append(" ".join(tokens))

            df["transformed_texts"] = processed
            logging.info("Text preprocessing completed.")

            X_train, X_test, y_train, y_test = self.split_data(df)

            logging.info("Data Ingestion completed successfully.")
            return X_train, X_test, y_train, y_test

        except Exception as e:
            raise CustomException(e, sys)