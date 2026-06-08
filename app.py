from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd

app = Flask(__name__)

with open('logistic_regression_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    
    input_data = pd.DataFrame([{
        'Age': data['Age'],
        'Income': data['Income'],
        'LoanAmount': data['LoanAmount'],
        'CreditScore': data['CreditScore'],
        'MonthsEmployed': data['MonthsEmployed'],
        'NumCreditLines': data['NumCreditLines'],
        'InterestRate': data['InterestRate'],
        'LoanTerm': data['LoanTerm'],
        'DTIRatio': data['DTIRatio'],
        'Education': data['Education'],
        'HasMortgage': data['HasMortgage'],
        'HasDependents': data['HasDependents'],
        'HasCoSigner': data['HasCoSigner'],
        'EmploymentType_Full-time': data['EmploymentType_Full-time'],
        'EmploymentType_Part-time': data['EmploymentType_Part-time'],
        'EmploymentType_Self-employed': data['EmploymentType_Self-employed'],
        'EmploymentType_Unemployed': data['EmploymentType_Unemployed'],
        'MaritalStatus_Divorced': data['MaritalStatus_Divorced'],
        'MaritalStatus_Married': data['MaritalStatus_Married'],
        'MaritalStatus_Single': data['MaritalStatus_Single'],
        'LoanPurpose_Auto': data['LoanPurpose_Auto'],
        'LoanPurpose_Business': data['LoanPurpose_Business'],
        'LoanPurpose_Education': data['LoanPurpose_Education'],
        'LoanPurpose_Home': data['LoanPurpose_Home'],
        'LoanPurpose_Other': data['LoanPurpose_Other']
    }])
    
    # Scale and predict
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]
    
    return jsonify({
        'prediction': 'DEFAULT' if prediction == 1 else 'NO DEFAULT',
        'default_probability': round(float(probability) * 100, 2)
    })

if __name__ == '__main__':
    app.run(debug=True)