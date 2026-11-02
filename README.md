# CrediScope

**Clear decisions. Clear reasons.**

CrediScope is an explainable credit-risk assessment prototype developed for the COM300 Applied Artificial Intelligence module.

The system predicts whether a loan applicant is at higher or lower risk of defaulting within 24 months. It combines a trained machine-learning model with applicant-level explanations and a plain-English notice designed to make the model output understandable to both technical users and applicants.

---

## Project Objectives

CrediScope was developed to demonstrate how machine learning could support credit-risk assessment while maintaining transparency and human oversight.

The system is designed to:

- predict `default_within_24m` from an applicant record
- provide a predicted probability of default
- explain individual predictions using SHAP
- convert technical model explanations into readable language
- generate an applicant-facing notice
- support, rather than replace, human lending decisions
- document model limitations, fairness concerns and responsible-AI considerations

---

## Dataset

The project uses a synthetic UK credit application dataset containing:

- 6,000 loan applications
- 25 predictive features
- 1 application identifier
- 1 binary target: `default_within_24m`

The target distribution is imbalanced:

- approximately 74% non-default
- approximately 26% default

The dataset also contains a small amount of missing data in:

- `savings_balance_gbp`
- `years_at_current_address`

The application identifier is excluded from model training.

Protected attributes are retained for fairness analysis but excluded from the production model.

---

## Machine-Learning Approach

Two main model families were evaluated:

- Logistic Regression
- Random Forest

Logistic Regression was used as an interpretable baseline.

Random Forest was selected as the final production model because it provided stronger detection of default cases while maintaining good overall discrimination.

The final Random Forest uses:

- 200 trees
- `random_state=42`
- `class_weight="balanced"`

The final model operates at the default classification threshold of `0.50`.

---

## Final Model Performance

Performance on the held-out test set:

| Metric | Result |
|---|---:|
| Accuracy | 0.787 |
| Precision | 0.581 |
| Recall | 0.593 |
| F1 Score | 0.587 |
| ROC-AUC | 0.820 |

Confusion matrix:

```text
[[762 131]
 [125 182]]
```

The model correctly identifies more default cases than the Logistic Regression baseline, but this comes with an increased number of false positives.

This trade-off is discussed in the accompanying analysis and Responsible AI evaluation.

---

## Feature Engineering

Two additional financial ratios are calculated during preprocessing:

### Debt-to-income ratio

```text
existing debt / annual income
```

### Expense-to-income ratio

```text
monthly expenses × 12 / annual income
```

These ratios provide additional information about affordability and existing financial commitments.

Feature engineering is implemented in:

```text
src/preprocessing.py
```

---

## Production Preprocessing Pipeline

The final saved model includes the complete preprocessing workflow.

### Numeric features

Numeric values are processed using median imputation where required.

### Categorical features

Categorical values use:

- most-frequent imputation
- one-hot encoding
- `handle_unknown="ignore"`

The Gradio interface reads the permitted categorical values directly from the fitted encoder so that the interface remains aligned with the trained model.

---

## Explainability

CrediScope uses SHAP to produce applicant-level explanations.

For each prediction, the system identifies the most influential transformed features and determines whether each factor pushed the predicted default risk:

- higher
- lower

The application translates these into readable statements such as:

```text
Expense-to-income ratio pushed the predicted default risk higher for this applicant.
```

SHAP explanations describe how the model reached a prediction. They should not be interpreted as proving that a feature caused the applicant's risk.

---

## Applicant-Facing Explanation

The Gradio application also produces a plain-English applicant notice.

This separates two audiences:

### Technical explanation

Provides SHAP-based information about the factors influencing the prediction.

### Applicant notice

Explains the result in accessible language without exposing raw model internals or SHAP values.

The application also makes clear that the result is not a final lending decision.

---

## Responsible AI

CrediScope is designed as a decision-support prototype rather than an autonomous lending system.

The project considers:

- false positives
- false negatives
- subgroup performance
- fairness across protected characteristics
- small subgroup uncertainty
- proxy-risk features such as region
- explainability limitations
- synthetic-data limitations
- preprocessing leakage
- model monitoring
- human oversight

Protected attributes are excluded from the production model but retained separately for fairness evaluation.

The interface includes a human-review notice stating that model output should not be used as the sole basis for a lending decision.

---

## Project Structure

```text
CrediScope/
│
├── app/
│   └── app.py
│
├── data/
│
├── models/
│   └── crediscope_random_forest.joblib
│
├── notebooks/
│   └── CrediScope_Analysis_v2.ipynb
│
├── src/
│   └── preprocessing.py
│
├── .gitignore
├── README.md
└── requirements.txt
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/Richardbrown85/CrediScope.git
cd CrediScope
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate the virtual environment.

### Windows PowerShell

```powershell
.venv\Scripts\Activate.ps1
```

Install the project dependencies:

```bash
pip install -r requirements.txt
```

---

## Run the Analysis Notebook

Start Jupyter or open the notebook directly in VS Code:

```text
notebooks/CrediScope_Analysis_v2.ipynb
```

The notebook has been tested using:

```text
Restart Kernel → Run All
```

and executes successfully from top to bottom without relying on hidden notebook state.

---

## Run the CrediScope Application

From the project root:

```bash
python app/app.py
```

Gradio will start the local application and provide a browser address.

The interface allows the user to:

1. enter loan details
2. enter applicant details
3. enter employment and residence information
4. enter financial information
5. enter credit-history information
6. review the application
7. run the risk assessment

The output includes:

- risk assessment
- predicted default probability
- main SHAP-based prediction factors
- applicant-facing notice
- Responsible AI / human-review warning

---

## Reproducibility

The project includes:

- a saved production model
- reproducible preprocessing
- feature engineering in `src/`
- a pinned Python dependency list in `requirements.txt`
- a notebook that runs successfully from a fresh kernel
- version control through Git and GitHub

The production model can be loaded directly using:

```python
import joblib

model = joblib.load(
    "models/crediscope_random_forest.joblib"
)
```

---

## Key Limitations

CrediScope is a prototype and should not be treated as a production lending system.

Important limitations include:

- the dataset is synthetic
- real-world credit behaviour may differ from the training data
- subgroup sizes vary
- fairness metrics can be unstable for small groups
- SHAP explains model behaviour rather than causation
- false positives may incorrectly identify a lower-risk applicant as higher risk
- false negatives may fail to identify an applicant who later defaults
- model performance may deteriorate over time if the applicant population changes

A real deployment would require further validation, governance, monitoring and regulatory review.

---

## Development Environment

The project was developed using:

- Python 3.14
- VS Code
- Jupyter
- pandas
- NumPy
- scikit-learn
- matplotlib
- SHAP
- Gradio
- joblib
- Git
- GitHub

Exact installed package versions are recorded in:

```text
requirements.txt
```

---

## Status

Current project status:

- model development complete
- model evaluation complete
- fairness analysis complete
- Responsible AI analysis complete
- SHAP explainability complete
- production model saved and validated
- reproducible notebook complete
- Gradio application complete
- applicant-facing explanation implemented
- human-review warning implemented

The remaining work relates primarily to final documentation, technical reporting, presentation preparation and submission packaging.