import os
import sys
from sqlalchemy import create_engine, text   # type: ignore
from dotenv import load_dotenv               # type: ignore
from SMS_Spam_Classifier.exceptions.exception import CustomException
from SMS_Spam_Classifier.logger.logging import logging
from SMS_Spam_Classifier.constant import SQL_DB_URL_KEY, DATABASE_NAME

load_dotenv()


class SQLClient:
    """
    Singleton PostgreSQL client.

    WHY SINGLETON?
    If DataIngestion, ModelTrainer, and Evaluator all create their own
    SQLClient, you'd open 3 separate connection pools to the DB.
    Singleton ensures only ONE engine is created for the entire app lifecycle.

    This is the exact same pattern you used in your MongoDBClient.
    """

    engine = None  # Class-level variable — shared across all instances

    def __init__(self, database_name: str = DATABASE_NAME) -> None:
        try:
            # Only create the engine ONCE (singleton check)
            if SQLClient.engine is None:

                db_url = os.getenv(SQL_DB_URL_KEY)

                if db_url is None:
                    raise Exception(
                        f"Environment key '{SQL_DB_URL_KEY}' is not set."
                    )

                # pool_size     → number of persistent connections kept open
                # max_overflow  → extra connections allowed beyond pool_size
                # pool_pre_ping → tests connection health before using it
                #                 (prevents "connection closed" errors)
                SQLClient.engine = create_engine(
                    db_url,
                    pool_size=5,
                    max_overflow=10,
                    pool_pre_ping=True,
                    echo=False
                )

                # Verify connection works right away
                with SQLClient.engine.connect() as conn:
                    conn.execute(text("SELECT 1"))

                logging.info(
                    f"PostgreSQL connection established. "
                    f"Database: {database_name}"
                )

            # Instance-level reference to the class-level engine
            self.engine = SQLClient.engine
            self.database_name = database_name

        except Exception as e:
            raise CustomException(e, sys)

    def get_engine(self):
        """Returns the SQLAlchemy engine for use in other components."""
        return self.engine