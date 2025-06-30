Heart Disease Prediction using Machine Learning
This project predicts whether a person has heart disease using clinical features like age, cholesterol, resting blood pressure, and chest pain type. 
I used a Random Forest Classifier to train the model and evaluated it using accuracy, classification report, and confusion matrix.


Why This Project Matters
- Heart disease is one of the leading causes of death worldwide.
- Early prediction using machine learning can help in timely diagnosis and treatment.
- This project simulates how real-world healthcare AI systems work by analyzing patient data and predicting disease risk.

What I Did
- Cleaned and preprocessed the heart disease dataset
- Encoded categorical variables using 'get_dummies'
- Trained a **Random Forest model**
- Evaluated model using accuracy, precision, recall, F1-score
- Saved:
  - Trained model ('heart_disease_model.pkl')
  - Scaler ('scaler.pkl')
  - Model report ('model_report.txt')
  - Confusion matrix heatmap ('confusion_matrix.png')

Example Output
Accuracy: 86.89%

Classification Report:
              precision    recall  f1-score   support

           0       0.88      0.85      0.86        54
           1       0.86      0.89      0.87        56

    accuracy                           0.87       110
   macro avg       0.87      0.87      0.87       110
weighted avg       0.87      0.87      0.87       110

Tools and Libraries Used
- Python
- Pandas, NumPy
- Scikit-learn
- Seaborn, Matplotlib
- Joblib

What This Project Covers
- Heart disease prediction using machine learning
- Data preprocessing and feature encoding
- Model training with Random Forest
- Model evaluation (accuracy, precision, recall, F1-score)
- Confusion matrix visualization
- Saving model, scaler, and evaluation report
- Clean and organized ML workflow

How to Run
Install required libraries: pandas, numpy, scikit-learn, matplotlib, seaborn, joblib
- Place 'heart.csv' inside the 'Data/' folder
- Run the script:
  python heart_disease_prediction.py
It will train the model, evaluate it, and save all outputs in the 'Outputs/' folder

Output Files

- heart_disease_model.pkl        → Trained model
- scaler.pkl                     → Scaler used for input normalization
- model_report.txt               → Accuracy and classification report
- confusion_matrix.png           → Confusion matrix heatmap

About Me
I’m Jaya Kaushik, exploring machine learning and its applications in healthcare. 
I enjoy building real-world ML projects to improve my skills and help others learn too.
- Passionate about AI, data science & healthcare tech
- Always learning and sharing projects

Lets Connect:
GitHub: (https://github.com/jayakaushik2102)  
LinkedIn: (https://www.linkedin.com/in/jaya-kaushik-b51978237)

