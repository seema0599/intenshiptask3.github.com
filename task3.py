#IRIS FLOWER CLASSIFICATION

# Importing required libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Step 1: Load the Dataset
df = pd.read_csv('iris_is.csv', encoding='ISO-8859-1')  # Load the dataset
print("Dataset loaded successfully!:",df)
print("\nFirst 5 rows of the dataset:\n", df.head())  # Display the first 5 rows
print("\nInfo about the dataset:\n", df.info())  # Dataset structure and types
print("\nStatistical description of the dataset:\n", df.describe())  # Summary statistics

# Checking for missing values
print("\nMissing values in each column:\n", df.isnull().sum())  # Identify missing values
print("\nUnique species in the dataset:", df['species'].unique())  # Unique values in the target column

# Step 2: Splitting Features and Target Variables
X = df.drop(columns=['species'])  # Features (independent variables)
y = df['species']  # Target variable (dependent variable)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("\nData split successfully!")
print("Training features shape:", X_train.shape)  # Dimensions of training data
print("Testing features shape:", X_test.shape)  # Dimensions of testing data

# Step 3: Visualizing Target Distribution in Test Set
sns.countplot(x=y_test, hue=y_test, palette="viridis", legend=False)
plt.title("Distribution of Iris Species in Test Set")
plt.show()

# Step 4: Logistic Regression Model
log_reg_model = LogisticRegression(max_iter=200)  # Initialize Logistic Regression
log_reg_model.fit(X_train, y_train)  # Train the model
y_pred_lr = log_reg_model.predict(X_test)  # Make predictions on the test set

# Evaluation of Logistic Regression Model
print("\nLogistic Regression Model:")
conf_matrix_lr = confusion_matrix(y_test, y_pred_lr)
print("Confusion Matrix:\n", conf_matrix_lr)
print("Classification Report:\n", classification_report(y_test, y_pred_lr))
accuracy_lr = accuracy_score(y_test, y_pred_lr)
print(f"Logistic Regression Model Accuracy: {accuracy_lr * 100:.2f}%")

# Step 5: Random Forest Model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)  # Initialize Random Forest
rf_model.fit(X_train, y_train)  # Train the model
y_pred_rf = rf_model.predict(X_test)  # Make predictions on the test set

# Evaluation of Random Forest Model
print("\nRandom Forest Model:")
conf_matrix_rf = confusion_matrix(y_test, y_pred_rf)
print("Confusion Matrix:\n", conf_matrix_rf)
print("Classification Report:\n", classification_report(y_test, y_pred_rf))
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f"Random Forest Model Accuracy: {accuracy_rf * 100:.2f}%")

# Step 6: Correlation Analysis
numeric_df = df.select_dtypes(include=[np.number])  # Select numeric columns
correlation_matrix = numeric_df.corr()  # Compute correlations

# Plot the correlation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='Reds', fmt='.2f', cbar=True)
plt.title("Correlation Heatmap for Numeric Columns")
plt.show()

# Step 7: Predicting on New Data
data = {
    'sepal_length': [5],
    'sepal_width': [3],
    'petal_length': [1.5],
    'petal_width': [0.2],
}

input_data = pd.DataFrame(data)  # Convert the dictionary into a DataFrame
flower_type = rf_model.predict(input_data)[0]  # Predict the flower type using Random Forest
print(f"\nFlower Classified as: {flower_type}")