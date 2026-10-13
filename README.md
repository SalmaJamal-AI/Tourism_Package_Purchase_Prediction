# Tourism Package Purchase Prediction — End-to-End MLOps

**Project Overview**

This project develops an end-to-end Machine Learning and MLOps pipeline to predict whether a customer will purchase a newly introduced Wellness Tourism Package offered by a travel company, Visit with Us.

The solution uses customer demographics, travel preferences, and sales interaction data to identify potential buyers before marketing outreach. It integrates data preprocessing, model training, hyperparameter tuning, experiment tracking, model evaluation, and a Streamlit-based prediction interface into a reproducible workflow.

The project also incorporates GitHub Actions for CI/CD automation and MLflow for experiment tracking, demonstrating how machine learning workflows can be organized for repeatable model development and delivery.

**Key Features**

- **Data preprocessing:** Numerical feature scaling and categorical feature encoding using scikit-learn pipelines.

- **Machine Learning**: XGBoost classifier for binary purchase prediction.

- **Hyperparameter optimization:** GridSearchCV with cross-validation to identify suitable model configurations.

- **Experiment tracking:** MLflow to record model parameters, evaluation metrics, and model artifacts.

- **Model evaluation:** Accuracy, precision, recall, and F1-score, with F1-score used as the primary metric for hyperparameter selection.

- **CI/CD automation:** GitHub Actions workflow to orchestrate data preparation, model training, evaluation, and model artifact handling.

- **Prediction interface:** Streamlit application that accepts customer information and predicts purchase or no purchase.

**Technology Stack**

Python | Pandas | scikit-learn | XGBoost | MLflow | Streamlit | Joblib | Git | GitHub Actions

**Business Value**

The project demonstrates how predictive analytics can support customer segmentation and targeted marketing by helping businesses prioritize prospective buyers. The automated workflow provides a foundation for reproducible model training, experiment tracking, and streamlined model updates.

**Repository**

[View the source code and project files](https://github.com/SalmaJamal-AI/Tourism_Package_Purchase_Prediction)

## Project Notebook

Explore the complete Jupyter notebook for data analysis, preprocessing, model development, hyperparameter tuning, and evaluation.

[View the Jupyter Notebook](notebooks/TourismPackage_Advanced_MLOps_Project.ipynb)
