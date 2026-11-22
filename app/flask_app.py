"""CrediScope Flask UI:
Start from the repository root: python app/flask_app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))  # Needed to load src.preprocessing from joblib

import joblib
import numpy as np
import pandas as pd
import shap
from flask import Flask, jsonify, render_template, request

MODEL_PATH = ROOT / "models" / "crediscope_random_forest.joblib"

# Keep the feature list and ordering aligned with the trained model.
CATEGORICAL_FEATURES = [
    "region", "employment_status", "credit_history", "has_savings_account",
    "checking_account_status", "loan_purpose", "has_guarantor",
    "previous_application_declined", "application_channel",
]

NUMERICAL_FIELDS = {
    "num_dependents": (0, 30, True),
    "years_in_current_job": (0, 80, True),
    "months_in_current_job": (0, 11, True),
    "annual_income_gbp": (0, 100_000_000, False),
    "monthly_expenses_gbp": (0, 100_000_000, False),
    "years_at_current_address": (0, 120, False),
    "existing_debt_gbp": (0, 100_000_000, False),
    "num_existing_credit_lines": (0, 1000, True),
    "num_late_payments_24m": (0, 1000, True),
    "savings_balance_gbp": (0, 100_000_000, False),
    "loan_amount_gbp": (0, 100_000_000, False),
    "loan_term_months": (1, 360, True),
}

# Only request a positive-class probability, not an automatic lending decision.
FEATURE_LABELS = {
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
    "expense_to_income_ratio": "Expense-to-income ratio",
}

app = Flask(__name__, template_folder="templates", static_folder="static")
app.json.sort_keys = False

# Fail visibly at application startup if the trained pipeline is missing or invalid.
model = joblib.load(MODEL_PATH)
preprocessor = model.named_steps["preprocessor"]
classifier = model.named_steps["classifier"]
column_transformer = preprocessor.named_steps["column_transformer"]
categorical_pipeline = column_transformer.named_transformers_["categorical"]
encoder = categorical_pipeline.named_steps["encoder"]

if len(CATEGORICAL_FEATURES) != len(encoder.categories_):
    raise RuntimeError("Categorical feature list differs from the saved model.")

category_choices = {
    feature: [str(value) for value in values]
    for feature, values in zip(CATEGORICAL_FEATURES, encoder.categories_)
}
feature_names = list(column_transformer.get_feature_names_out())
explainer = shap.TreeExplainer(classifier)


def readable_feature(feature_name: str) -> str:
    cleaned = feature_name.replace("numeric__", "").replace("categorical__", "")
    if cleaned in FEATURE_LABELS:
        return FEATURE_LABELS[cleaned]
    # Check longer prefixes first, avoiding ambiguous partial matches.
    for feature in sorted(CATEGORICAL_FEATURES, key=len, reverse=True):
        prefix = f"{feature}_"
        if cleaned.startswith(prefix):
            category = cleaned[len(prefix):]
            return f"{feature.replace('_', ' ').title()}: {category}"
    return cleaned.replace("_", " ").title()


def explain_applicant(applicant: pd.DataFrame) -> list[dict]:
    transformed = preprocessor.transform(applicant)
    shap_values = explainer.shap_values(transformed)
    if isinstance(shap_values, list):
        contributions = np.asarray(shap_values[1])[0]
    else:
        values = np.asarray(shap_values)
        if values.ndim == 3:
            contributions = values[0, :, 1]
        elif values.ndim == 2:
            contributions = values[0]
        else:
            raise ValueError("Unexpected SHAP output shape")

    contributions = np.asarray(contributions, dtype=float)
    if len(contributions) != len(feature_names):
        raise ValueError("SHAP feature count did not match the trained model")

    order = np.argsort(np.abs(contributions))[::-1][:5]
    return [
        {
            "feature": readable_feature(feature_names[i]),
            "direction": "higher" if contributions[i] > 0 else "lower",
            "contribution": float(contributions[i]),
        }
        for i in order
    ]


def applicant_notice(higher_risk: bool, probability: float) -> str:
    pct = f"{probability:.1%}"
    if higher_risk:
        return (
            f"Your application has been assessed as presenting a higher predicted risk "
            f"of default ({pct}). The assessment reflects the financial and credit "
            "information provided, and the main model factors are shown above. "
            "This is not a final lending decision. An authorised human reviewer "
            "should consider the application before any outcome is determined."
        )
    return (
        f"Your application has been assessed as presenting a lower predicted risk "
        f"of default ({pct}). The assessment reflects the financial and credit "
        "information provided. This does not guarantee loan approval; "
        "the result should form only part of a human-reviewed lending decision."
    )


def validate_payload(data: object) -> dict:
    if not isinstance(data, dict):
        raise ValueError("Expected an applicant record")
    clean = {}
    for field, (minimum, maximum, integer_only) in NUMERICAL_FIELDS.items():
        value = data.get(field)
        if value is None or value == "" or isinstance(value, bool):
            raise ValueError(f"{field.replace('_', ' ').title()} is required")
        try:
            numeric = float(value)
        except (ValueError, TypeError):
            raise ValueError(f"{field.replace('_', ' ').title()} must be a number") from None
        if not np.isfinite(numeric) or not minimum <= numeric <= maximum:
            raise ValueError(f"{field.replace('_', ' ').title()} must be between {minimum} and {maximum}")
        if integer_only and not numeric.is_integer():
            raise ValueError(f"{field.replace('_', ' ').title()} must be a whole number")
        clean[field] = int(numeric) if integer_only else numeric
    for field, choices in category_choices.items():
        value = data.get(field)
        if not isinstance(value, str) or value not in choices:
            raise ValueError(f"Choose a valid {field.replace('_', ' ')}")
        clean[field] = value
    return clean


@app.get("/")
def index():
    return render_template("crediscope.html", categories=category_choices)


@app.post("/api/predict")
def predict():
    try:
        data = validate_payload(request.get_json(silent=True))
        applicant = pd.DataFrame([{
            "num_dependents": data["num_dependents"],
            "region": data["region"],
            "employment_status": data["employment_status"],
            "months_in_current_job": data["years_in_current_job"] * 12 + data["months_in_current_job"],
            "annual_income_gbp": data["annual_income_gbp"],
            "monthly_expenses_gbp": data["monthly_expenses_gbp"],
            "years_at_current_address": data["years_at_current_address"],
            "existing_debt_gbp": data["existing_debt_gbp"],
            "num_existing_credit_lines": data["num_existing_credit_lines"],
            "num_late_payments_24m": data["num_late_payments_24m"],
            "credit_history": data["credit_history"],
            "has_savings_account": data["has_savings_account"],
            "savings_balance_gbp": data["savings_balance_gbp"],
            "checking_account_status": data["checking_account_status"],
            "loan_amount_gbp": data["loan_amount_gbp"],
            "loan_term_months": data["loan_term_months"],
            "loan_purpose": data["loan_purpose"],
            "has_guarantor": data["has_guarantor"],
            "previous_application_declined": data["previous_application_declined"],
            "application_channel": data["application_channel"],
        }])

        probability = float(model.predict_proba(applicant)[0, 1])
        higher_risk = int(model.predict(applicant)[0]) == 1
        factors = explain_applicant(applicant)
        return jsonify({
            "risk": "Higher Risk" if higher_risk else "Lower Risk",
            "higher_risk": higher_risk,
            "probability": probability,
            "factors": factors,
            "notice": applicant_notice(higher_risk, probability),
            "threshold": 0.5,
        })
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception:
        app.logger.exception("CrediScope prediction failed")
        return jsonify({"error": "Assessment could not be completed. Check the server terminal."}), 500


if __name__ == "__main__":
    # Local demonstration only: do not expose the development server publicly.
    app.run(host="127.0.0.1", port=5000, debug=False)
