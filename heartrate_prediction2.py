# -*- coding: utf-8 -*-
#HEARTRATE_PREDICTION.ipynb



#Original file is located at(https://colab.research.google.com/drive/1uK255DwmxbqeOuufmSazmvBgp6ZsLDcO)

#HERE I HAVE USED AHEART RATE CSV FILE(https://raw.githubusercontent.com/abilbiju/DATASETS/main/heartrate_prediction.csv) FOR PERFORMING THE EDA ANALYSIS AND PREDICTION.

#STEP1 :LOADING THE DATASET

import pandas as pd

# Load the dataset
url = 'https://raw.githubusercontent.com/abilbiju/DATASETS/main/heartrate_prediction.csv'
data = pd.read_csv(url)

# Display the first few rows of the dataset
data.head()

#STEP 2: EDA ANALYSIS

# Import necessary libraries for EDA
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Check the dimensions of the dataset
print("Dataset Dimensions: ", data.shape)

# Get an overview of the dataset
print("\nDataset Information:")
data.info()

# Check for missing values
print("\nMissing Values:")
print(data.isnull().sum())

# Summary statistics
print("\nSummary Statistics:")
print(data.describe())

# Correlation matrix
correlation_matrix = data.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="YlGnBu")
plt.title("Correlation Matrix")
plt.show()

# Distribution of the target variable
plt.figure(figsize=(8, 6))
sns.histplot(data['target'], kde=True)
plt.title("Distribution of Target Variable")
plt.xlabel("Target Variable")
plt.ylabel("Count")
plt.show()

#Step 3: Preparing the Data for Modeling

from sklearn.model_selection import train_test_split

# Splitting the data into features and target variable
X = data.drop('target', axis=1)
y = data['target']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Step 4: Model Selection, Training, and Evaluation

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Create the classifier
classifier = RandomForestClassifier(random_state=42)

# Train the model
classifier.fit(X_train, y_train)

# Predict on the test set
y_pred = classifier.predict(X_test)

# Model evaluation
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)

#An accuracy score of 1.0 indicates that the model achieved a perfect prediction on the test set.
