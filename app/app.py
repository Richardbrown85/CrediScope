from pathlib import Path
import sys

import gradio as gr
import joblib
import numpy as np
import pandas as pd
import shap


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

project_root = Path(__file__).resolve().parent.parent

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

model_path = (
    project_root
    / "models"
    / "crediscope_random_forest.joblib"
)


# ---------------------------------------------------------
# Load trained CrediScope model
# ---------------------------------------------------------

model = joblib.load(model_path)

preprocessor = model.named_steps["preprocessor"]
classifier = model.named_steps["classifier"]


# ---------------------------------------------------------
# Read categorical values directly from trained model
# ---------------------------------------------------------

column_transformer = (
    preprocessor
    .named_steps["column_transformer"]
)

categorical_pipeline = (
    column_transformer
    .named_transformers_["categorical"]
)

encoder = categorical_pipeline.named_steps["encoder"]

categorical_features = [
    "region",
    "employment_status",
    "credit_history",
    "has_savings_account",
    "checking_account_status",
    "loan_purpose",
    "has_guarantor",
    "previous_application_declined",
    "application_channel"
]

category_choices = {
    feature: [str(value) for value in categories]
    for feature, categories in zip(
        categorical_features,
        encoder.categories_
    )
}


# ---------------------------------------------------------
# SHAP explainer
# ---------------------------------------------------------

explainer = shap.TreeExplainer(classifier)

feature_names = (
    column_transformer
    .get_feature_names_out()
)


# ---------------------------------------------------------
# Friendly feature labels
# ---------------------------------------------------------

friendly_labels = {
    "num_dependents": "Number of dependants",
    "months_in_current_job": "Time in current job",
    "annual_income_gbp": "Annual income",
    "monthly_expenses_gbp": "Monthly expenses",
    "years_at_current_address": "Time at current address",
    "existing_debt_gbp": "Existing debt",
    "num_existing_credit_lines": "Existing credit lines",
    "num_late_payments_24m": "Late payments in the last 24 months",
    "savings_balance_gbp": "Savings balance",
    "loan_amount_gbp": "Requested loan amount",
    "loan_term_months": "Loan term",
    "debt_to_income_ratio": "Debt-to-income ratio",
    "expense_to_income_ratio": "Expense-to-income ratio"
}


# ---------------------------------------------------------
# Convert transformed feature name into readable text
# ---------------------------------------------------------

def make_feature_readable(feature_name):

    clean_name = (
        feature_name
        .replace("numeric__", "")
        .replace("categorical__", "")
    )

    if clean_name in friendly_labels:
        return friendly_labels[clean_name]

    for feature in categorical_features:

        prefix = f"{feature}_"

        if clean_name.startswith(prefix):

            category = clean_name.replace(
                prefix,
                "",
                1
            )

            feature_label = (
                feature
                .replace("_", " ")
                .title()
            )

            return f"{feature_label}: {category}"

    return (
        clean_name
        .replace("_", " ")
        .title()
    )


# ---------------------------------------------------------
# Produce applicant-level SHAP explanation
# ---------------------------------------------------------

def explain_applicant(applicant):

    transformed = preprocessor.transform(applicant)

    shap_values = explainer.shap_values(
        transformed
    )

    # SHAP output format can vary slightly between versions.
    if isinstance(shap_values, list):

        default_values = np.asarray(
            shap_values[1]
        )[0]

    else:

        shap_array = np.asarray(shap_values)

        if shap_array.ndim == 3:

            default_values = (
                shap_array[0, :, 1]
            )

        elif shap_array.ndim == 2:

            default_values = (
                shap_array[0]
            )

        else:

            raise ValueError(
                "Unexpected SHAP output format."
            )

    explanation = pd.DataFrame({
        "feature": feature_names,
        "shap_value": default_values
    })

    explanation["absolute_shap"] = (
        explanation["shap_value"].abs()
    )

    top_factors = (
        explanation
        .sort_values(
            "absolute_shap",
            ascending=False
        )
        .head(5)
    )

    reasons = []

    for _, row in top_factors.iterrows():

        display_feature = make_feature_readable(
            row["feature"]
        )

        if row["shap_value"] > 0:

            direction = (
                "pushed the predicted "
                "default risk higher"
            )

        else:

            direction = (
                "pushed the predicted "
                "default risk lower"
            )

        reasons.append(
            f"- **{display_feature}** "
            f"{direction} for this applicant."
        )

    return "\n".join(reasons)


# ---------------------------------------------------------
# Prediction function
# ---------------------------------------------------------

def predict_risk(
    num_dependents,
    region,
    employment_status,
    years_in_current_job,
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

    months_in_current_job_total = (
        years_in_current_job * 12
        + months_in_current_job
    )

    applicant = pd.DataFrame([{
        "num_dependents": num_dependents,
        "region": region,
        "employment_status": employment_status,
        "months_in_current_job": months_in_current_job_total,
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
        "previous_application_declined":
            previous_application_declined,
        "application_channel": application_channel
    }])

    prediction = model.predict(applicant)[0]

    probability = model.predict_proba(
        applicant
    )[0][1]

    decision = (
        "Higher Risk"
        if prediction == 1
        else "Lower Risk"
    )

    explanation = explain_applicant(
        applicant
    )

    return (
        decision,
        f"{probability:.1%}",
        explanation
    )


# ---------------------------------------------------------
# Gradio interface
# ---------------------------------------------------------

with gr.Blocks(title="CrediScope") as demo:

    gr.Markdown("# CrediScope")
    gr.Markdown(
        "### Clear decisions. Clear reasons."
    )

    # -----------------------------------------------------
    # 1. Loan Details
    # -----------------------------------------------------

    gr.Markdown("## 1. Loan Details")

    with gr.Row():

        loan_amount_gbp = gr.Number(
            label="Loan Amount (£)",
            value=10000,
            minimum=0
        )

        loan_term_months = gr.Dropdown(
            label="Loan Term",
            choices=[
                12,
                24,
                36,
                48,
                60,
                72
            ],
            value=36
        )

    with gr.Row():

        loan_purpose = gr.Dropdown(
            label="Loan Purpose",
            choices=category_choices[
                "loan_purpose"
            ],
            value=category_choices[
                "loan_purpose"
            ][0]
        )

        application_channel = gr.Dropdown(
            label="Application Channel",
            choices=category_choices[
                "application_channel"
            ],
            value=category_choices[
                "application_channel"
            ][0]
        )

    # -----------------------------------------------------
    # 2. Applicant Details
    # -----------------------------------------------------

    gr.Markdown("## 2. Applicant Details")

    with gr.Row():

        region = gr.Dropdown(
            label="Region",
            choices=category_choices[
                "region"
            ],
            value=category_choices[
                "region"
            ][0]
        )

        num_dependents = gr.Number(
            label="Number of Dependants",
            value=0,
            minimum=0,
            precision=0,
            step=1
        )

    # -----------------------------------------------------
    # 3. Employment & Residence
    # -----------------------------------------------------

    gr.Markdown(
        "## 3. Employment & Residence"
    )

    employment_status = gr.Dropdown(
        label="Employment Status",
        choices=category_choices[
            "employment_status"
        ],
        value=category_choices[
            "employment_status"
        ][0]
    )

    with gr.Row():

        years_in_current_job = gr.Number(
            label="Years in Current Job",
            value=0,
            minimum=0,
            precision=0,
            step=1
        )

        months_in_current_job = gr.Number(
            label="Additional Months",
            value=0,
            minimum=0,
            maximum=11,
            precision=0,
            step=1
        )

        years_at_current_address = gr.Number(
            label="Years at Current Address",
            value=1,
            minimum=0
        )

    # -----------------------------------------------------
    # 4. Financial Position
    # -----------------------------------------------------

    gr.Markdown("## 4. Financial Position")

    with gr.Row():

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

    with gr.Row():

        has_savings_account = gr.Dropdown(
            label="Has Savings Account",
            choices=category_choices[
                "has_savings_account"
            ],
            value=category_choices[
                "has_savings_account"
            ][0]
        )

        savings_balance_gbp = gr.Number(
            label="Savings Balance (£)",
            value=0,
            minimum=0
        )

        checking_account_status = gr.Dropdown(
            label="Checking Account Status",
            choices=category_choices[
                "checking_account_status"
            ],
            value=category_choices[
                "checking_account_status"
            ][0]
        )

    # -----------------------------------------------------
    # 5. Credit History
    # -----------------------------------------------------

    gr.Markdown("## 5. Credit History")

    with gr.Row():

        num_existing_credit_lines = gr.Number(
            label="Existing Credit Lines",
            value=0,
            minimum=0,
            precision=0,
            step=1
        )

        num_late_payments_24m = gr.Number(
            label="Late Payments in Last 24 Months",
            value=0,
            minimum=0,
            precision=0,
            step=1
        )

    credit_history = gr.Dropdown(
        label="Credit History",
        choices=category_choices[
            "credit_history"
        ],
        value=category_choices[
            "credit_history"
        ][0]
    )

    with gr.Row():

        has_guarantor = gr.Dropdown(
            label="Has Guarantor",
            choices=category_choices[
                "has_guarantor"
            ],
            value=category_choices[
                "has_guarantor"
            ][0]
        )

        previous_application_declined = (
            gr.Dropdown(
                label=(
                    "Previous Application "
                    "Declined"
                ),
                choices=category_choices[
                    "previous_application_declined"
                ],
                value=category_choices[
                    "previous_application_declined"
                ][0]
            )
        )

    # -----------------------------------------------------
    # 6. Review
    # -----------------------------------------------------

    gr.Markdown(
        "## 6. Review Application"
    )

    gr.Markdown(
        "Check the applicant information above "
        "before running the risk assessment."
    )

    # -----------------------------------------------------
    # 7. Risk Assessment
    # -----------------------------------------------------

    gr.Markdown(
        "## 7. Risk Assessment"
    )

    predict_button = gr.Button(
        "Predict Risk",
        variant="primary"
    )

    with gr.Row():

        decision_output = gr.Textbox(
            label="Risk Assessment",
            interactive=False
        )

        probability_output = gr.Textbox(
            label=(
                "Predicted Default Probability"
            ),
            interactive=False
        )

    gr.Markdown(
        "### Main Factors Influencing "
        "This Assessment"
    )

    explanation_output = gr.Markdown(
        "Run the risk assessment to see "
        "the main prediction factors."
    )

    gr.Markdown(
        "> **Important:** These factors explain "
        "how the model reached this prediction. "
        "They do not prove that any factor caused "
        "the applicant's risk."
    )

    predict_button.click(
        fn=predict_risk,
        inputs=[
            num_dependents,
            region,
            employment_status,
            years_in_current_job,
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
        ],
        outputs=[
            decision_output,
            probability_output,
            explanation_output
        ]
    )


# ---------------------------------------------------------
# Launch application
# ---------------------------------------------------------

if __name__ == "__main__":
    demo.launch()