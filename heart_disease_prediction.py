#step 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

#step 1 load the data
df = pd.read_csv(r"C:\Users\Shivam Kaushik\Desktop\Heart Disease Project\Data\heart.csv")

#step 2: Basic checks
print("Shape:, df.shape")
print("Columns:", df.columns)
print(df.head())

#step3: Convert categorical data(if any)
df = pd.get_dummies(df, drop_first=True)

#step 4: split the data
x = df.drop("HeartDisease", axis=1)
y = df['HeartDisease']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

#step 5: Train model
model = RandomForestClassifier(random_state=42)
model.fit(x_train, y_train)

#step 6: Predict and evaluate
y_pred = model.predict(x_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

from sklearn.metrics import confusion_matrix
#step7: Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

#Visualize the Confusion matrix
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Disease','Disease'])
plt.xlabel('Predicated')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()

import os
save_path = r"C:\Users\Shivam Kaushik\Desktop\Heart Disease Project\Outputs"
if not os.path.exists(save_path):
    os.makedirs(save_path)

# Standardize features before saving scaler
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Retrain model with scaled  data
model.fit(x_train_scaled, y_train)

#save model and scaler
joblib.dump(model, os.path.join(save_path, "heart_disease_model.pkl"))
joblib.dump(scaler, os.path.join(save_path, "scaler.pkl"))
print("Model and Scaler saved successfully.")

from sklearn.metrics import classification_report, accuracy_score

y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
# path to save folder
save_folder = r"C:\Users\Shivam Kaushik\Desktop\Heart Disease Project\Outputs"
report_file = os.path.join(save_folder, "model_report.txt")

if not os.path.exists(save_folder):
    os.makedir(save_folder)

with open(report_file, "w") as f:
    f.write(f"Accuracy: {accuracy * 100:2f}%\n\n")
    f.write("Classification Report:\n")
    f.write(report)

print("Model report at:", report_file)