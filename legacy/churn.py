import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report

data = pd.read_csv('Churn_Modelling.csv')
X = data[['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']]
y = data['Exited']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
with open('churn_prediction_results.txt', 'w') as f:
    
   #Logistic Regression
    f.write('Logistic Regression:\n')
    model_lr = LogisticRegression()
    model_lr.fit(X_train_scaled, y_train)
    y_pred_lr = model_lr.predict(X_test_scaled)
    f.write('Accuracy: {}\n'.format(accuracy_score(y_test, y_pred_lr)))
    f.write('Classification Report:\n{}\n'.format(classification_report(y_test, y_pred_lr)))
    f.write('Churned (1) vs Not Churned (0) in Test Data:\n')
    f.write(pd.DataFrame({'Exited': y_test}).value_counts().to_string() + '\n')
    # Random Forest
    f.write('\nRandom Forest:\n')
    model_rf = RandomForestClassifier()
    model_rf.fit(X_train_scaled, y_train)
    y_pred_rf = model_rf.predict(X_test_scaled)
    f.write('Accuracy: {}\n'.format(accuracy_score(y_test, y_pred_rf)))
    f.write('Classification Report:\n{}\n'.format(classification_report(y_test, y_pred_rf)))
    f.write('Churned (1) vs Not Churned (0) in Test Data:\n')
    f.write(pd.DataFrame({'Exited': y_test}).value_counts().to_string() + '\n')
    # Gradient Boosting
    f.write('\nGradient Boosting:\n')
    model_gb = GradientBoostingClassifier()
    model_gb.fit(X_train_scaled, y_train)
    y_pred_gb = model_gb.predict(X_test_scaled)
    f.write('Accuracy: {}\n'.format(accuracy_score(y_test, y_pred_gb)))
    f.write('Classification Report:\n{}\n'.format(classification_report(y_test, y_pred_gb)))
    f.write('Churned (1) vs Not Churned (0) in Test Data:\n')
    f.write(pd.DataFrame({'Exited': y_test}).value_counts().to_string() + '\n')

#new data
prompts = ['Credit Score', 
            'Age', 
            'Tenure', 
            'Balance', 
            'Number of Products', 
            'Has Credit Card (1 for Yes, 0 for No)', 
            'Is Active Member (1 for Yes, 0 for No)', 
            'Estimated Salary']

user_input = []
for prompt in prompts:
    user_input.append(float(input(f'Enter {prompt}: ')))
sample_data = np.array([user_input])

sample_data_scaled = scaler.transform(sample_data)
lr_prediction = model_lr.predict(sample_data_scaled)
rf_prediction = model_rf.predict(sample_data_scaled)
gb_prediction = model_gb.predict(sample_data_scaled)
for i in range(len(sample_data)):
    print(f'Customer {i+1}:')
    print('Logistic Regression Prediction:', 'Exited' if lr_prediction[i] == 1 else 'Not Exited')
    print('Random Forest Prediction:', 'Exited' if rf_prediction[i] == 1 else 'Not Exited')
    print('Gradient Boosting Prediction:', 'Exited' if gb_prediction[i] == 1 else 'Not Exited')
    print()
