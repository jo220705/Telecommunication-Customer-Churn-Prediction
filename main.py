import pandas as pd
print("Program Started")
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import joblib
import matplotlib.pyplot as plt


df = pd.read_csv("dataset/WA_Fn-UseC_-Telco-Customer-Churn.csv")

print(df.head())

print("\n------------------------")
print("Dataset Shape")
print(df.shape)

print("\n------------------------")
print("Column Names")
print(df.columns)

print("\n------------------------")
print("Dataset Information")
print(df.info())

print("\n------------------------")
print("Missing Values")

print(df.isnull().sum())

print("\n------------------------")
print("Before Conversion")

print(df["TotalCharges"].dtype)

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

print("\n------------------------")
print("After Conversion")

print(df["TotalCharges"].dtype)

print("\nMissing Values in TotalCharges")

print(df["TotalCharges"].isnull().sum())

print("\nRows Before Cleaning:")
print(df.shape)

df = df.dropna()
print("\nRows After Cleaning:")
print(df.shape)
print("\nMissing Values After Cleaning")

print(df.isnull().sum())
print("\nText Columns")

print(df.select_dtypes(include="object").columns)
df = df.drop("customerID", axis=1)
print("\nColumns After Removing customerID")

print(df.columns)
df = pd.get_dummies(df, drop_first=True)
print("\nShape After Encoding")
print(df.shape)

print("\nFirst 5 Rows After Encoding")
print(df.head())
# Features and Target

X = df[[
    "tenure",
    "MonthlyCharges",
    "TotalCharges"
]]

y = df["Churn_Yes"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = DecisionTreeClassifier(random_state=42)

model.fit(X_train, y_train)

prediction = model.predict(X_test)

accuracy = accuracy_score(y_test, prediction)

print("Model Accuracy:", round(accuracy * 100, 2), "%")
joblib.dump(model, "customer_churn_model.pkl")

print("Model Saved Successfully")

cm = confusion_matrix(y_test, prediction)

print("\nConfusion Matrix")
print(cm)

print("\nClassification Report")

print(classification_report(y_test, prediction))

joblib.dump(model, "customer_churn_model.pkl")

print("\nModel Saved Successfully")

importance = model.feature_importances_

feature_names = X.columns

plt.figure(figsize=(10,6))

plt.barh(feature_names, importance)

plt.title("Feature Importance")

plt.xlabel("Importance")

plt.show()