from pathlib import Path
import sys
import joblib

project_root = Path(__file__).resolve().parent.parent

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

model_path = project_root / "models" / "crediscope_random_forest.joblib"

model = joblib.load(model_path)

print("CrediScope model loaded successfully.")
print(type(model))

sample_applicant = {
    "num_dependents": 1,
    "region": "East of England",
    "employment_status": "Employed",
    "months_in_current_job": 48,
    "annual_income_gbp": 42000,
    "monthly_expenses_gbp": 1800,
    "years_at_current_address": 5,
    "existing_debt_gbp": 8000,
    "num_existing_credit_lines": 2,
    "num_late_payments_24m": 0,
    "credit_history": "All paid on time",
    "has_savings_account": "Yes",
    "savings_balance_gbp": 5000,
    "checking_account_status": "0-500",
    "loan_amount_gbp": 10000,
    "loan_term_months": 36,
    "loan_purpose": "Car",
    "has_guarantor": "No",
    "previous_application_declined": "No",
    "application_channel": "Online"
}

import pandas as pd

applicant_df = pd.DataFrame([sample_applicant])

prediction = model.predict(applicant_df)[0]
probability = model.predict_proba(applicant_df)[0][1]

print("Prediction:", prediction)
print("Default probability:", probability)