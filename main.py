from SMS_Spam_Classifier.components.data_ingestion import DataIngestion

if __name__ == "__main__":
    obj = DataIngestion()
    X_train, X_test, y_train, y_test = obj.initiate_data_ingestion()
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape:  {X_test.shape}")
    print("---------------------------------")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape:  {y_test.shape}")