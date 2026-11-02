from pathlib import Path
import sys
import joblib
import shap

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

decision = "Higher Risk" if prediction == 1 else "Lower Risk"

print("Decision:", decision)
print(f"Predicted default probability: {probability:.1%}")

preprocessor = model.named_steps["preprocessor"]
classifier = model.named_steps["classifier"]

transformed_applicant = preprocessor.transform(applicant_df)

explainer = shap.TreeExplainer(classifier)
shap_values = explainer.shap_values(transformed_applicant)

feature_names = (
    preprocessor
    .named_steps["column_transformer"]
    .get_feature_names_out()
)

default_shap_values = shap_values[0, :, 1]

explanation = pd.DataFrame({
    "feature": feature_names,
    "shap_value": default_shap_values
})

explanation["absolute_shap"] = explanation["shap_value"].abs()

top_factors = explanation.sort_values(
    "absolute_shap",
    ascending=False
).head(5)

print("\nTop prediction factors:")
print(top_factors[["feature", "shap_value"]])

friendly_labels = {
    "numeric__expense_to_income_ratio": "Expense-to-income ratio",
    "numeric__annual_income_gbp": "Annual income",
    "numeric__debt_to_income_ratio": "Debt-to-income ratio",
    "numeric__num_late_payments_24m": "Late payments in the last 24 months",
    "categorical__credit_history_All paid on time": "Credit history: all paid on time"
}

top_factors["display_feature"] = top_factors["feature"].replace(
    friendly_labels
)

top_factors["direction"] = top_factors["shap_value"].apply(
    lambda x: "higher" if x > 0 else "lower"
)

print("\nPlain-English factors:")

for _, row in top_factors.iterrows():
    print(
        f"- {row['display_feature']} pushed the predicted default risk "
        f"{row['direction']} for this applicant."
    )

def predict_risk(
    num_dependents,
    region,
    employment_status,
    months_in_current_job,
    annual_income_gbp,
    monthly_expenses_gbp,
    years_at_current_address,
    existing_debt_gbp,
    num_existing_credit_lines,
    num_late_payments_24m,
    credit_history,
    has_savings_account,
    savings_balance_gbp,
    checking_account_status,
    loan_amount_gbp,
    loan_term_months,
    loan_purpose,
    has_guarantor,
    previous_application_declined,
    application_channel
):
    applicant = pd.DataFrame([{
        "num_dependents": num_dependents,
        "region": region,
        "employment_status": employment_status,
        "months_in_current_job": months_in_current_job,
        "annual_income_gbp": annual_income_gbp,
        "monthly_expenses_gbp": monthly_expenses_gbp,
        "years_at_current_address": years_at_current_address,
        "existing_debt_gbp": existing_debt_gbp,
        "num_existing_credit_lines": num_existing_credit_lines,
        "num_late_payments_24m": num_late_payments_24m,
        "credit_history": credit_history,
        "has_savings_account": has_savings_account,
        "savings_balance_gbp": savings_balance_gbp,
        "checking_account_status": checking_account_status,
        "loan_amount_gbp": loan_amount_gbp,
        "loan_term_months": loan_term_months,
        "loan_purpose": loan_purpose,
        "has_guarantor": has_guarantor,
        "previous_application_declined": previous_application_declined,
        "application_channel": application_channel
    }])

    prediction = model.predict(applicant)[0]
    probability = model.predict_proba(applicant)[0][1]

    decision = "Higher Risk" if prediction == 1 else "Lower Risk"

    return decision, f"{probability:.1%}"

import gradio as gr

with gr.Blocks(title="CrediScope") as demo:

    gr.Markdown("# CrediScope")
    gr.Markdown("### Clear decisions. Clear reasons.")

    # -------------------------
    # Applicant Details
    # -------------------------

    gr.Markdown("## Applicant Details")

    num_dependents = gr.Number(
        label="Number of Dependants",
        value=0,
        minimum=0
    )

    region = gr.Dropdown(
        label="Region",
        choices=[
            "East of England",
            "East Midlands",
            "London",
            "North East",
            "North West",
            "Scotland",
            "South East",
            "South West",
            "Wales",
            "West Midlands",
            "Yorkshire and The Humber"
        ]
    )

    employment_status = gr.Dropdown(
        label="Employment Status",
        choices=[
            "Employed",
            "Self-employed",
            "Unemployed",
            "Student",
            "Retired"
        ]
    )

    months_in_current_job = gr.Number(
        label="Months in Current Job",
        value=0,
        minimum=0
    )

    # -------------------------
    # Financial Details
    # -------------------------

    gr.Markdown("## Financial Details")

    annual_income_gbp = gr.Number(
        label="Annual Income (£)",
        value=30000,
        minimum=0
    )

    monthly_expenses_gbp = gr.Number(
        label="Monthly Expenses (£)",
        value=1500,
        minimum=0
    )

    existing_debt_gbp = gr.Number(
        label="Existing Debt (£)",
        value=0,
        minimum=0
    )

    savings_balance_gbp = gr.Number(
        label="Savings Balance (£)",
        value=0,
        minimum=0
    )

    # -------------------------
    # Credit Details
    # -------------------------

    gr.Markdown("## Credit Details")

    years_at_current_address = gr.Number(
        label="Years at Current Address",
        value=1,
        minimum=0
    )

    num_existing_credit_lines = gr.Number(
        label="Number of Existing Credit Lines",
        value=0,
        minimum=0
    )

    num_late_payments_24m = gr.Number(
        label="Late Payments in Last 24 Months",
        value=0,
        minimum=0
    )

    credit_history = gr.Dropdown(
        label="Credit History",
        choices=[
            "All paid on time",
            "Existing credits paid",
            "No credits taken",
            "Past delays",
            "Critical account"
        ]
    )

    # -------------------------
    # Account Details
    # -------------------------

    gr.Markdown("## Account Details")

    has_savings_account = gr.Dropdown(
        label="Has Savings Account",
        choices=["Yes", "No"]
    )

    checking_account_status = gr.Dropdown(
        label="Checking Account Status",
        choices=[
            "Overdrawn",
            "0-500",
            ">=500",
            "No account"
        ]
    )

    # -------------------------
    # Loan Details
    # -------------------------

    gr.Markdown("## Loan Details")

    loan_amount_gbp = gr.Number(
        label="Loan Amount (£)",
        value=10000,
        minimum=0
    )

    loan_term_months = gr.Number(
        label="Loan Term (Months)",
        value=36,
        minimum=1
    )

    loan_purpose = gr.Dropdown(
        label="Loan Purpose",
        choices=[
            "Car",
            "Home improvements",
            "Debt consolidation",
            "Education",
            "Medical",
            "Other"
        ]
    )

    has_guarantor = gr.Dropdown(
        label="Has Guarantor",
        choices=["Yes", "No"]
    )

    # -------------------------
    # Application Details
    # -------------------------

    gr.Markdown("## Application Details")

    previous_application_declined = gr.Dropdown(
        label="Previous Application Declined",
        choices=["Yes", "No"]
    )

    application_channel = gr.Dropdown(
        label="Application Channel",
        choices=[
            "Online",
            "Branch",
            "Telephone"
        ]
    )


demo.launch()