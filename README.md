🛡️ Fraud Detection System Using Machine Learning
Overview

The Fraud Detection System is a machine learning project designed to identify fraudulent financial transactions. It uses multiple machine learning algorithms to detect suspicious activities and helps prevent financial losses.

The system preprocesses the dataset, handles missing values, balances the data using SMOTE, trains multiple models, and evaluates them using accuracy, AUC score, confusion matrix, and ROC curves. Among all models, Random Forest delivers the best performance and identifies the key features contributing to fraud detection.

📂 Dataset

The dataset contains financial transaction records with features like transaction amount, device type, location, etc.

The target column is Fraudulent (renamed to is_fraud in the project).

Dataset preprocessing includes:

Handling missing values

Label encoding categorical features

Balancing classes using SMOTE

⚙️ Features

Data Cleaning & Preprocessing

Label Encoding for categorical variables

SMOTE for class imbalance handling

Feature Scaling using StandardScaler

Exploratory Data Analysis (EDA) with visualizations

Model Training & Evaluation for:

Logistic Regression

Decision Tree

K-Nearest Neighbors (KNN)

Naive Bayes

Support Vector Machine (SVM)

Random Forest

Model Evaluation Metrics: Accuracy, AUC Score, Confusion Matrix, ROC Curve

Feature Importance Analysis using Random Forest
