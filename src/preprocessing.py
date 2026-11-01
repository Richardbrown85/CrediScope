def add_financial_ratios(data):
    data = data.copy()

    data["debt_to_income_ratio"] = (
        data["existing_debt_gbp"] / data["annual_income_gbp"]
    )

    data["expense_to_income_ratio"] = (
        data["monthly_expenses_gbp"] * 12 / data["annual_income_gbp"]
    )

    return data