# FRAUD DETECTION SYSTEM PROJECT

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import roc_auc_score, roc_curve

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from imblearn.over_sampling import SMOTE

# LOAD DATASET

df = pd.read_csv(r"C:\Users\yadav\Desktop\Fraud Detection Dataset.csv")

print("Dataset Shape:", df.shape)
print(df.head())
print(df.info())

print("\nMissing Values:\n", df.isnull().sum())
print("\nDuplicate Rows:", df.duplicated().sum())

df.rename(columns={'Fraudulent': 'is_fraud'}, inplace=True)

for col in df.columns:
    if df[col].dtype == 'object':
        df[col].fillna(df[col].mode()[0], inplace=True)
    else:
        df[col].fillna(df[col].median(), inplace=True)
        
if 'Transaction_ID' in df.columns:
    df.drop(['Transaction_ID'], axis=1, inplace=True)

for col in df.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

plt.figure(figsize=(6,4))
sns.countplot(x='is_fraud', data=df)
plt.title("Fraud vs Non-Fraud Distribution")
plt.show()

plt.figure(figsize=(6,4))
sns.boxplot(x='is_fraud', y='Transaction_Amount', data=df)
plt.title("Transaction Amount vs Fraud")
plt.show()

plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()


X = df.drop('is_fraud', axis=1)
y = df['is_fraud']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

smote = SMOTE(random_state=42)

X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

print("Before SMOTE:", y_train.value_counts())
print("After SMOTE:", y_train_res.value_counts())


scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train_res)
X_test_scaled = scaler.transform(X_test)


models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=7),
    "Naive Bayes": GaussianNB(),
    "SVM": SVC(kernel='rbf', probability=True),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42)
}

results = []


for name, model in models.items():

    if name in ["Decision Tree", "Random Forest"]:
        model.fit(X_train_res, y_train_res)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:,1]
    else:
        model.fit(X_train_scaled, y_train_res)
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:,1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    results.append([name, acc, auc])

    print("\n===================================")
    print(name)
    print("Accuracy:", acc)
    print("AUC Score:", auc)

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

summary = pd.DataFrame(results, columns=["Model", "Accuracy", "AUC Score"])

print("\nFINAL MODEL COMPARISON")
print(summary)

summary.set_index("Model").plot(kind='bar', figsize=(8,4))
plt.title("Fraud Detection Model Comparison")
plt.ylabel("Score")
plt.show()

plt.figure(figsize=(7,5))

for name, model in models.items():

    if name in ["Decision Tree", "Random Forest"]:
        y_prob = model.predict_proba(X_test)[:,1]
    else:
        y_prob = model.predict_proba(X_test_scaled)[:,1]

    fpr, tpr, _ = roc_curve(y_test, y_prob)

    plt.plot(fpr, tpr, label=name)

plt.plot([0,1],[0,1],'--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.show()


rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train_res, y_train_res)

importances = pd.Series(rf.feature_importances_, index=X.columns)

importances.sort_values().plot(kind='barh', figsize=(8,5))
plt.title("Feature Importance (Random Forest)")
plt.show()

print("\nFraud Detection System Completed Successfully")
