# Loan Default Prediction

Flask web app and machine learning project for predicting whether a borrower is likely to default on a loan. The repository includes the training notebook, dataset, evaluation visuals, and the Flask frontend/backend used for inference.

## Project Overview

This project uses borrower, employment, and loan-related attributes to predict:

- `DEFAULT`
- `NO DEFAULT`

The current Flask app serves a browser UI and exposes a JSON prediction endpoint.

## Repository Structure

```text
Loan_default_ML/
|-- app.py
|-- templates/
|   `-- index.html
|-- Loan_default.csv
|-- Loan_default.ipynb
|-- confusion_matrix_comparison.png
|-- feature_importance.png
|-- Report.png
|-- Screen Record.mp4
|-- caption.srt
|-- requirements.txt
|-- .gitignore
|-- README.md
```

Notes:

- The repository ignores `*.pkl` files.
- `app.py` expects `logistic_regression_model.pkl` and `scaler.pkl` to be available in the project root at runtime.

## Files

- `app.py`: Flask application with `/` and `/predict` routes
- `templates/index.html`: Browser UI for entering borrower details and viewing predictions
- `Loan_default.csv`: Source dataset used for analysis and modeling
- `Loan_default.ipynb`: Notebook for data exploration, preprocessing, training, and evaluation
- `confusion_matrix_comparison.png`: Model comparison visual
- `feature_importance.png`: Feature importance visual
- `Report.png`: Summary report image
- `Screen Record.mp4`: Demo recording
- `caption.srt`: Subtitle file for the demo video
- `requirements.txt`: Python dependencies

## Application Behavior

### Home Page

Open `/` to load the HTML form in `templates/index.html`.

### Prediction Endpoint

`POST /predict`

The request body must be JSON with the fields expected by `app.py`, including:

- `Age`
- `Income`
- `LoanAmount`
- `CreditScore`
- `MonthsEmployed`
- `NumCreditLines`
- `InterestRate`
- `LoanTerm`
- `DTIRatio`
- `Education`
- `HasMortgage`
- `HasDependents`
- `HasCoSigner`
- `EmploymentType_Full-time`
- `EmploymentType_Part-time`
- `EmploymentType_Self-employed`
- `EmploymentType_Unemployed`
- `MaritalStatus_Divorced`
- `MaritalStatus_Married`
- `MaritalStatus_Single`
- `LoanPurpose_Auto`
- `LoanPurpose_Business`
- `LoanPurpose_Education`
- `LoanPurpose_Home`
- `LoanPurpose_Other`

Example response:

```json
{
  "prediction": "DEFAULT",
  "default_probability": 73.42
}
```

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Flask app:

```bash
python app.py
```

Then open the local server shown in the terminal.

## Required Model Artifacts

To run inference successfully, place these files in the repository root:

- `logistic_regression_model.pkl`
- `scaler.pkl`

These files are not committed to git because `*.pkl` is ignored.

## Tech Stack

- Python
- Flask
- pandas
- scikit-learn
- XGBoost
- NumPy
- Jupyter Notebook

## Demo Assets

The repository includes:

- `Screen Record.mp4` for the walkthrough
- `caption.srt` for subtitles
- `Report.png` for the summary visual

## Training Workflow

The notebook covers:

- data loading and inspection
- preprocessing and feature engineering
- model training
- evaluation and comparison
- selection of the final model used by the Flask app

## License

No license file is currently included in the repository.
