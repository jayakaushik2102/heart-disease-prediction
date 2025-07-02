Heart Disease Prediction using Machine Learning
This project predicts whether a person has heart disease using clinical features like age, cholesterol, resting blood pressure, and chest pain type.  
A Random Forest Classifier is used for training, and the model is evaluated using accuracy, classification report, and confusion matrix.

Why This Project Matters
- Heart disease is one of the leading causes of death worldwide.
- Early prediction using machine learning can help in timely diagnosis and treatment.
- Simulates how real-world healthcare AI systems analyze patient data and predict disease risk.

What I Did
- Cleaned and preprocessed the heart disease dataset
- Encoded categorical variables using 'get_dummies'
- Trained a Random Forest model
- Evaluated model using accuracy, precision, recall, F1-score
- Saved:
  - Trained model             -                        'heart_disease_model.pkl'
  - Scaler                    -                        'scaler.pkl'
  - Model report              -                        'model_report.pkl'
  - Confusion matrix heatmap  -                        'confusion_matrix.png'

Example Output
Accuracy: 75.543478% 

Tools and Libraries Used
- Python
- Pandas, NumPy
- Scikit-learn
- Seaborn, Matplotlib
- Joblib

What This Project Covers
- Data preprocessing & feature encoding
- ML model training using Random Forest
- Model evaluation metrics
- Visualization with confusion matrix
- Saving model, scaler & reports
- End-to-end machine learning workflow

How to Run
Install required libraries: pandas, numpy, scikit-learn, matplotlib, seaborn, joblib
- Place 'heart.csv' inside the 'Data/' folder
- Run the script:
  python heart_disease_prediction.py
It will train the model, evaluate it, and save all outputs in the 'Outputs/' folder

Output Files

- heart_disease_model.pkl        → Trained model
- scaler.pkl                     → Scaler used for input normalization
- model_report.pkl              → Accuracy and classification report
- confusion_matrix.png           → Confusion matrix heatmap

About Me
I’m Jaya Kaushik, exploring machine learning and its applications in healthcare. 
I enjoy building real-world ML projects to improve my skills and help others learn too.
- Passionate about AI, data science & healthcare tech
- Always learning and sharing projects

Lets Connect:
GitHub: (https://github.com/jayakaushik2102)  
LinkedIn: (https://www.linkedin.com/in/jaya-kaushik-b51978237)

