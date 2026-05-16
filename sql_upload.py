import os
import sys
import pandas as pd # type: ignore
from sqlalchemy import create_engine, text   # type: ignore
from dotenv import load_dotenv     # type: ignore       
from SMS_Spam_Classifier.exceptions.exception import CustomException
from SMS_Spam_Classifier.logger.logging import logging
from SMS_Spam_Classifier.constant import TABLE_NAME, SQL_DB_URL_KEY



load_dotenv()

class SQLDataUpload:

    def __init__(self):
        try:
            #Read the PostgreSQL URL from system env variable 
            self.db_url = os.getenv(SQL_DB_URL_KEY)

            if self.db_url is None:
                raise Exception(f"Environment key '{SQL_DB_URL_KEY}' is not set.")

            # create_engine() sets up the connection pool 
            # It does NOT open a connection yet. Connection opens only when
            # you actually execute a query (lazy connection).
            self.engine = create_engine(
                self.db_url,
                echo=False   # Set echo=True during debugging to see raw SQL logs
            )

            #Test the connection immediately to see if it works
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))  # Lightweight ping query
            
            logging.info("PostgreSQL Engine created and connection verified.")

        except Exception as e:
            raise CustomException(e, sys)


    def load_csv(self, file_path: str) -> pd.DataFrame:
        """
        Reads the CSV file and returns a clean DataFrame.
        The SMS Spam dataset from Kaggle has 5 columns but only
        the first two matter: v1 (label) and v2 (message).
        """
        try:
            # encoding='latin-1' is necessary because spam.csv has special characters
            df = pd.read_csv(file_path, encoding='latin-1')

            # Keeping only the two useful columns and rename them to clean names
            df = df.drop(["Unnamed: 2","Unnamed: 3","Unnamed: 4"],axis=1)
            df = df.rename(columns={"v1":"label","v2":"message"})

            #Mapping teh values of label to Binary.
            df["label"] = df["label"].map({"ham":0 , "spam" : 1})

            df.reset_index(drop=True, inplace=True)

            logging.info(f"CSV loaded successfully. Total records: {len(df)}")
            return df

        except Exception as e:
            raise CustomException(e, sys)


    def insert_data_to_SQL(self, dataframe: pd.DataFrame, table_name: str = TABLE_NAME) -> int:
        """
        Pushes the DataFrame into a PostgreSQL table.

        if_exists='replace' → Drops and recreates the table every time.
        Good for development. In production use 'append' or handle migrations.

        index=False → Don't write the DataFrame index as a column in the DB.
        """
        try:
            dataframe.to_sql(
                name=table_name,       # Table name in PostgreSQL
                con=self.engine,       # SQLAlchemy engine (PostgreSQL connection)
                if_exists='replace',   # Drop table if exists, then recreate
                index=False,           # Don't push DataFrame index as a column
                method='multi'         # Batch insert — much faster than row-by-row
            )

            record_count = len(dataframe)
            logging.info(
                f"Data pushed to PostgreSQL table '{table_name}'. "
                f"Records inserted: {record_count}"
            )
            return record_count

        except Exception as e:
            raise CustomException(e, sys)


# ─── Run this file directly to upload your dataset ────────────────────────────
if __name__ == '__main__':
    try:
        FILE_PATH = "Data/spam.csv"

        uploader = SQLDataUpload()

        # Step 1: Load CSV into DataFrame
        df = uploader.load_csv(file_path=FILE_PATH)
        print(f"Preview of loaded data:\n{df.head()}\n")

        # Step 2: Push DataFrame into PostgreSQL
        count = uploader.insert_data_to_SQL(dataframe=df)
        print(f"Total records inserted into PostgreSQL: {count}")

    except Exception as e:
        raise CustomException(e, sys)