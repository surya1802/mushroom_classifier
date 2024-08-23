Mushroom Classification with Random Forest

Description
This project involves classifying mushrooms into two categories: edible (0) and poisonous (1) using a Random Forest classifier. The dataset used is mushroom_cleaned.csv, which contains features related to the mushrooms' physical characteristics.

Prerequisites
Ensure you have the following Python libraries installed:

pandas
numpy
scikit-learn
You can install these libraries using pip if they are not already installed:

pip install pandas numpy scikit-learn
Setup and Installation
Clone the Repository:

git clone https://github.com/yourusername/yourrepository.git
cd yourrepository
Download the Dataset:

Ensure the mushroom_cleaned.csv file is placed in the root directory of the repository.

Usage
Run the following script to perform the classification:

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

np.random.seed(0)

# Load the dataset
df = pd.read_csv('mushroom_cleaned.csv')

# Display the first few rows of the dataframe
print(df.head())

# Check the column names and data types
print(df.info())

# Check for missing values
print(df.isnull().sum())

# Encoding categorical columns
df_encoded = pd.get_dummies(df)

# Define features and target
X = df_encoded.drop('class', axis=1)
Y = df_encoded['class']

# Split the data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Train the Random Forest classifier
classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(X_train, Y_train)

# Make predictions
Y_pred = classifier.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(Y_test, Y_pred)
report = classification_report(Y_test, Y_pred)

print(f"Accuracy: {accuracy:.2f}")
print("\nClassification Report:\n", report)
Results
The Random Forest classifier achieved an accuracy of 0.99 on the test set. The classification report provides high precision, recall, and F1-score for both classes:


Accuracy: 0.99

Classification Report:
               precision    recall  f1-score   support

           0       0.99      0.99      0.99      4909
           1       0.99      0.99      0.99      5898

    accuracy                           0.99     10807
   macro avg       0.99      0.99      0.99     10807
weighted avg       0.99      0.99      0.99     10807

Additional Notes
The dataset mushroom_cleaned.csv should be placed in the root directory of the repository.
This project demonstrates the effectiveness of Random Forest in classifying mushrooms based on their physical characteristics.
For any questions or contributions, please contact me or open an issue on GitHub.

Feel free to adjust the text according to your specific project details and preferences. To add this file to your GitHub repository, you can follow these steps:

Create the File Locally:

Save the above content into a file named README.txt using any text editor.

Add the File to Your Repository:

git add README.txt
git commit -m "Add README.txt with project details"
git push origin main
Replace main with the appropriate branch name if your default branch is different. This will upload the README.txt to your GitHub repository.
