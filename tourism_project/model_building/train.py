# for data manipulation
import pandas as pd
# for model training
import xgboost as xgb
# for data preprocessing and pipeline creation
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
# for one-hot encoding
from sklearn.preprocessing import OneHotEncoder, StandardScaler
# for hyperparameter tuning
from sklearn.model_selection import GridSearchCV
# for model serialization
import joblib
import mlflow
# for generating classification report
from sklearn.metrics import classification_report

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("Tourism_Poject_Production")

# Xtrain/Xtest/ytrain/ytest are downloaded from the previous job's artifact
X_train = pd.read_csv("/content/tourism_project/model_building/Xtrain.csv")
X_test  = pd.read_csv("/content/tourism_project/model_building/Xtest.csv")
y_train = pd.read_csv("/content/tourism_project/model_building/ytrain.csv").squeeze()
y_test  = pd.read_csv("/content/tourism_project/model_building/ytest.csv").squeeze()

# Define feature types
numeric_features = [
    "Age",
    "CityTier",
    "DurationOfPitch",
    "NumberOfPersonVisiting",
    "NumberOfFollowups",
    "PreferredPropertyStar",
    "NumberOfTrips",
    "Passport",
    "PitchSatisfactionScore",
    "OwnCar",
    "NumberOfChildrenVisiting",
    "MonthlyIncome"
]

categorical_features = [
    "TypeofContact",
    "Occupation",
    "Gender",
    "ProductPitched",
    "MaritalStatus",
    "Designation"
]

# Define the preprocessing steps
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown="ignore"), categorical_features)
)

# Define base XGBoost model
xgb_model = xgb.XGBClassifier(random_state=42)

# Define hyperparameter grid
param_grid = {
    "xgbclassifier__n_estimators": [50, 100],
    "xgbclassifier__max_depth": [2, 3],
    "xgbclassifier__learning_rate": [0.05, 0.1],
}

# Model pipeline
model_pipeline = make_pipeline(
    preprocessor,
    xgb_model
)

with mlflow.start_run():
    # Hyperparameter tuning
    grid_search = GridSearchCV(model_pipeline, param_grid, cv=5, scoring="f1", n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # Log all parameter combinations and their mean test scores
    results = grid_search.cv_results_
    for i in range(len(results["params"])):
        param_set = results["params"][i]
        mean_score = results["mean_test_score"][i]
        std_score = results["std_test_score"][i]

        # Log each combination as a separate MLflow run
        with mlflow.start_run(nested=True):
            mlflow.log_params(param_set)
            mlflow.log_metric("mean_test_score", mean_score)
            mlflow.log_metric("std_test_score", std_score)

    # Log best parameters separately in main run
    mlflow.log_params(grid_search.best_params_)

    # Store and evaluate the best model
    best_model = grid_search.best_estimator_

    classification_threshold = 0.45

    y_pred_train_proba = best_model.predict_proba(X_train)[:, 1]
    y_pred_train = (y_pred_train_proba >= classification_threshold).astype(int)

    y_pred_test_proba = best_model.predict_proba(X_test)[:, 1]
    y_pred_test = (y_pred_test_proba >= classification_threshold).astype(int)

    train_report = classification_report(y_train, y_pred_train, output_dict=True)
    test_report = classification_report(y_test, y_pred_test, output_dict=True)

    # Log the metrics for the best model
    mlflow.log_metrics({
        "train_accuracy": train_report["accuracy"],
        "train_precision": train_report["1"]["precision"],
        "train_recall": train_report["1"]["recall"],
        "train_f1-score": train_report["1"]["f1-score"],
        "test_accuracy": test_report["accuracy"],
        "test_precision": test_report["1"]["precision"],
        "test_recall": test_report["1"]["recall"],
        "test_f1-score": test_report["1"]["f1-score"]
    })

    # Save next to app.py so the Streamlit app can load it directly, and log
    # it as an MLflow artifact for traceability
    model_path = "/content/tourism_project/deployment/best_tourism_package_purchase_model_v1.joblib"
    joblib.dump(best_model, model_path)
    mlflow.log_artifact(model_path, artifact_path="model")
    print(f"Model saved to {model_path}")
