import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, accuracy_score, roc_auc_score
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek, SMOTEENN
from scipy.stats import randint

class ClassificationPipeline:
    def __init__(self):
        """Initialize the pipeline."""
        self.dataset_path = input("Enter the path to your dataset: ")
        self.df = pd.read_csv(self.dataset_path)
        self.target_column = input("Enter the name of the target column: ")
        
        """ADD ADDITIONAL LOGIC IF NECESSARY"""

    # Preprocess the dataset
        self.preprocess_data()

        # Placeholder for results
        self.results = {}
        self.best_method = None

    def preprocess_data(self):
        """Preprocesses the dataset by handling duplicates, outliers, encoding categorical variables, and splitting the data."""
        #  Drop duplicate rows
        initial_shape = self.df.shape
        self.df = self.df.drop_duplicates()
        print(f"Removed {initial_shape[0] - self.df.shape[0]} duplicate rows.")
        print(f"Current shape: {self.df.shape}")

        # Handling outliers by clipping values at the 1st and 99th percentiles
        numerical_columns = self.df.select_dtypes(include='number').columns
        for col in numerical_columns:
            lower_bound = self.df[col].quantile(0.01)
            upper_bound = self.df[col].quantile(0.99)
            self.df.loc[:, col] = np.clip(self.df[col], lower_bound, upper_bound)
            print(f"{col} outliers removed.")
        print("Outliers capped at 1st and 99th percentiles.")

        # Encoding categorical variables using LabelEncoder
        categorical_columns = self.df.select_dtypes(include=['object']).columns
        le = LabelEncoder()
        for col in categorical_columns:
            self.df.loc[:, col] = le.fit_transform(self.df[col].astype(str))
            print(f"{col} encoded.")
        print("Categorical data encoded.")

        #  Spliting the dataset into features (X) and target (y)
        X = self.df.drop(columns=[self.target_column])
        y = self.df[self.target_column]
        print("Features and target split.")

        # Check class distribution and split into training and testing sets
        print("Original Class Distribution:")
        print(y.value_counts())

        # Stratified train-test split to preserve class distribution
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, stratify=y, random_state=42
        )

        print("\nTraining Set Distribution:")
        print(y_train.value_counts())
        print("\nTest Set Distribution:")
        print(y_test.value_counts())

        print("\nClass Distribution Percentages:")
        print("Original Dataset:")
        print((y.value_counts(normalize=True) * 100).round(2))
        print("Training Set:")
        print((y_train.value_counts(normalize=True) * 100).round(2))
        print("Test Set:")
        print((y_test.value_counts(normalize=True) * 100).round(2))

        # Storing the splits as attributes 
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
    

    def train_model(self, model_name, model, resample=None, use_important_features=False):
        """Train a model and compute metrics."""
        if resample == "SMOTE":
            smote = SMOTE(random_state=42)
            X_resampled, y_resampled = smote.fit_resample(self.X_train, self.y_train)
        elif resample == "SMOTETomek":
            smt = SMOTETomek(random_state=42)
            X_resampled, y_resampled = smt.fit_resample(self.X_train, self.y_train)
        elif resample == "SMOTEENN":
            sme = SMOTEENN(random_state=42)
            X_resampled, y_resampled = sme.fit_resample(self.X_train, self.y_train)
        else:
            X_resampled, y_resampled = self.X_train, self.y_train

        if use_important_features:
            # Training a Random Forest to extract feature importance
            rf = RandomForestClassifier(random_state=42)
            rf.fit(X_resampled, y_resampled)
            important_features = X_resampled.columns[rf.feature_importances_ > 0.05]

            # Subset both training and testing data to include only important features
            X_resampled = X_resampled[important_features]
            X_test_subset = self.X_test[important_features]  
        else:
            X_test_subset = self.X_test  #  full test dataset

        # Train the model
        model.fit(X_resampled, y_resampled)

        # Predicting and computing metrics
        y_pred = model.predict(X_test_subset)  # consistent test subset
        y_prob = model.predict_proba(X_test_subset)[:, 1]
        accuracy = accuracy_score(self.y_test, y_pred)
        auc_score = roc_auc_score(self.y_test, y_prob)

        # to store results
        key = f"{model_name}_{resample}" if resample else model_name
        key += "_important_features" if use_important_features else ""
        self.results[key] = {"accuracy": accuracy, "auc": auc_score}

        # best method
        if not self.best_method or (accuracy + auc_score) > (
            self.results[self.best_method]["accuracy"] + self.results[self.best_method]["auc"]
        ):
            self.best_method = key

    def tune_hyperparameters(self, model_name, model, param_grid, use_randomized_search=False):
        """Tune hyperparameters using GridSearchCV or RandomizedSearchCV."""
        if use_randomized_search:
            search = RandomizedSearchCV(model, param_grid, n_iter=10, cv=3, scoring="accuracy", random_state=42)
        else:
            search = GridSearchCV(model, param_grid, cv=3, scoring="accuracy")

        search.fit(self.X_train, self.y_train)
        best_model = search.best_estimator_

        # Evaluate the best model
        self.train_model(f"{model_name}_tuned", best_model)

    #     self.DecisionTree(X_train, X_test, y_train, y_test)


    # def DecisionTree(self,X_train, X_test, y_train, y_test):
    #     # Train Decision Tree
    #     dt = DecisionTreeClassifier(random_state=42)
    #     dt.fit(X_train, y_train)
    #     # Predict
    #     y_pred = dt.predict(X_test)
    #     # Evaluate model
    #     print("Classification Report:\n")
    #     print(classification_report(y_test, y_pred)) # 'classification_report' is a nifty little function that gives us a lot of classification metrics

    

    """ADD AS MANY FUNCTIONS AS NECESSARY."""

    def run_pipeline(self):
        """ADD NECESSARY LOGIC TO COMPLEETE THIS FUNCTION."""

        # Training Decision Tree and Random Forest models
        dt = DecisionTreeClassifier(random_state=42)
        rf = RandomForestClassifier(random_state=42)
        self.train_model("DecisionTree", dt)
        self.train_model("RandomForest", rf)

        # Applying resampling techniques
        for resample in ["SMOTE", "SMOTETomek", "SMOTEENN"]:
            self.train_model("DecisionTree", DecisionTreeClassifier(random_state=42), resample=resample)
            self.train_model("RandomForest", RandomForestClassifier(random_state=42), resample=resample)

        # Training on important features
        self.train_model("DecisionTree", DecisionTreeClassifier(random_state=42), use_important_features=True)
        self.train_model("RandomForest", RandomForestClassifier(random_state=42), use_important_features=True)

        # Hyperparameter tuning
        dt_param_grid = {"max_depth": [None, 10, 20], "min_samples_split": [2, 5, 10]}
        rf_param_grid = {"n_estimators": [100, 200], "max_depth": [None, 10, 20]}
        self.tune_hyperparameters("DecisionTree", DecisionTreeClassifier(random_state=42), dt_param_grid)
        self.tune_hyperparameters("RandomForest", RandomForestClassifier(random_state=42), rf_param_grid)
        self.tune_hyperparameters(
            "RandomForest", RandomForestClassifier(random_state=42), rf_param_grid, use_randomized_search=True
        )

        
        
        """DO NOT CHANGE THE FOLLOWING TWO LINES OF CODE. THEY ARE NEEDED TO TEST YOUR MODEL PERFORMANCE BY THE TEST SUITE."""
        print(f"Best Accuracy Score: {self.results[self.best_method]['accuracy']:.4f}")
        print(f"Best AUC Score: {self.results[self.best_method]['auc']:.4f}")


if __name__ == "__main__":
    pipeline = ClassificationPipeline()
    pipeline.run_pipeline()
   
