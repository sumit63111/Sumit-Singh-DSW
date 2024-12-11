import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder

class ModelPipeline:
    def __init__(self):
        self.ab_model = None
        self.lr_model = None
        self.selected_model = None
        self.data = None
        self.test_data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.label_encoders = {}

    def load(self, file_path):
        """Loads data from an Excel file."""
        self.data = pd.read_excel(file_path)
        print("Data loaded successfully.")

    def load_test_data(self, file_path):
        """Loads external test data from an Excel file."""
        self.test_data = pd.read_excel(file_path)
        print("Test data loaded successfully.")

    def preprocess(self):
        """Preprocesses the data for training."""
        if self.data is None:
            raise ValueError("Data not loaded. Please load data first.")

        # Handle datetime columns by converting them to numerical format
        for col in self.data.select_dtypes(include=['datetime64', 'datetime']):
            self.data[col] = self.data[col].apply(lambda x: x.timestamp() if pd.notnull(x) else 0)

        # Handle categorical columns by encoding them
        for col in self.data.select_dtypes(include=['object', 'category']):
            le = LabelEncoder()
            self.data[col] = le.fit_transform(self.data[col].astype(str))
            self.label_encoders[col] = le
e
        X = self.data.iloc[:, :-1]
        y = self.data.iloc[:, -1]

        # Train-test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        print("Data preprocessing complete.")

    def preprocess_test_data(self):
        """Preprocesses the external test data."""
        if self.test_data is None:
            raise ValueError("External test data not loaded. Please load test data first.")

        # Handle datetime columns by converting them to numerical format
        for col in self.test_data.select_dtypes(include=['datetime64', 'datetime']):
            self.test_data[col] = self.test_data[col].apply(lambda x: x.timestamp() if pd.notnull(x) else 0)

        # Handle categorical columns by encoding them using the same label encoders as training data
        for col in self.test_data.select_dtypes(include=['object', 'category']):
            if col in self.label_encoders:
                le = self.label_encoders[col]
                self.test_data[col] = le.transform(self.test_data[col].astype(str).fillna("unknown"))
            else:
                raise ValueError(f"Column '{col}' in test data was not present in training data.")
        print("External test data preprocessing complete.")

    def train(self):
        """Trains both AdaBoostClassifier and LogisticRegression models."""
        if self.X_train is None or self.y_train is None:
            raise ValueError("Data not preprocessed. Please preprocess data first.")

        # Train AdaBoostClassifier
        self.ab_model = AdaBoostClassifier(random_state=42)
        self.ab_model.fit(self.X_train, self.y_train)
        print("AdaBoost training complete.")

        # Train LogisticRegression
        self.lr_model = LogisticRegression(random_state=42, max_iter=1000)
        self.lr_model.fit(self.X_train, self.y_train)
        print("Logistic Regression training complete.")

    def test(self):
        """Tests both models and selects the one with higher accuracy."""
        if self.ab_model is None or self.lr_model is None:
            raise ValueError("Models not trained. Please train the models first.")
        ab_predictions = self.ab_model.predict(self.X_test)
        ab_accuracy = accuracy_score(self.y_test, ab_predictions)
        print("AdaBoost Testing Results:")
        print(f"Accuracy: {ab_accuracy:.2f}")
        lr_predictions = self.lr_model.predict(self.X_test)
        lr_accuracy = accuracy_score(self.y_test, lr_predictions)
        print("Logistic Regression Testing Results:")
        print(f"Accuracy: {lr_accuracy:.2f}")
        if ab_accuracy >= lr_accuracy:
            self.selected_model = self.ab_model
            print("AdaBoost selected as the final model.")
        else:
            self.selected_model = self.lr_model
            print("Logistic Regression selected as the final model.")

    def test_external(self):
        """Tests the selected model on external test data if provided."""
        if self.selected_model is None:
            raise ValueError("No model selected. Please run the test method first.")

        self.preprocess_test_data()
        X_external = self.test_data.iloc[:, :-1]
        y_external = self.test_data.iloc[:, -1]
        predictions = self.selected_model.predict(X_external)
        accuracy = accuracy_score(y_external, predictions)
        report = classification_report(y_external, predictions)
        print("Selected Model External Testing Results:")
        print(f"Accuracy: {accuracy:.2f}")
        print("Classification Report:")
        print(report)

    def predict(self, input_data):
        """Generates predictions for the provided input data using the selected model."""
        if self.selected_model is None:
            raise ValueError("No model selected. Please run the test method first.")

        predictions = self.selected_model.predict(input_data)
        return predictions

if __name__ == "__main__":
    pipeline = ModelPipeline()
    train_file_path = "/content/train_data.xlsx"
    test_file_path = "/content/test_data.xlsx"
    pipeline.load(train_file_path)
    pipeline.preprocess()
    pipeline.train()
    pipeline.test()
    pipeline.load_test_data(test_file_path)
    pipeline.test_external()
    example_data = pipeline.X_test.iloc[:5]  # Example input data
    predictions = pipeline.predict(example_data)
    print("Predictions:", predictions)
