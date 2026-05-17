from SMS_Spam_Classifier.components.data_ingestion import DataIngestion
from SMS_Spam_Classifier.components.data_validation import DataValidation
from SMS_Spam_Classifier.components.data_transformation import DataTransformation
from SMS_Spam_Classifier.components.model_trainer import ModelTrainer
from SMS_Spam_Classifier.logger.logging import logging

if __name__ == "__main__":

    #Stage 1: Data Ingestion
    logging.info(" STAGE 1: DATA INGESTION ")
    ingestion = DataIngestion()
    X_train, X_test, y_train, y_test = ingestion.initiate_data_ingestion()
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape:  {X_test.shape}")
    print("="*50)
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape:  {y_test.shape}")

    #Stage 2: Data Validation
    logging.info(" STAGE 2: DATA VALIDATION ")
    validation = DataValidation(X_train, X_test, y_train, y_test)
    is_valid = validation.initiate_data_validation()

    if not is_valid:
        raise Exception("Data Validation FAILED. Fix data issues before proceeding.")
    
    #Stage 3: Data Transformation
    logging.info(" STAGE 3: DATA TRANSFORMATION ")
    transformation = DataTransformation(X_train, X_test, y_train, y_test)
    X_train_smote, X_test_tfidf, y_train_smote, y_test = transformation.transform()

      #Stage 4: Model Training
    logging.info("="*20 + " STAGE 4: MODEL TRAINING " + "="*20)
    trainer = ModelTrainer(X_train_smote, X_test_tfidf, y_train_smote, y_test)
    best_model, best_metrics = trainer.train_and_select_best()

    print("\nPipeline completed successfully.")
    print(f"Best model : {best_model}")
    print(f"Best model metrics: {best_metrics}")