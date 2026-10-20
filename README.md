# CrediScope

**Clear decisions. Clear reasons.**

CrediScope is an explainable credit risk assessment prototype developed for the COM300 Applied Artificial Intelligence module.

## Development Environment

- Python 3.14.8
- Visual Studio Code
- Virtual environment: `venv`
- Jupyter notebooks for exploratory analysis and model development

## Initial Dependencies

- `pandas` — dataset loading and manipulation
- `numpy` — numerical operations
- `scikit-learn` — preprocessing, modelling and evaluation
- `matplotlib` — visualisation
- `jupyter` — exploratory analysis and model development

Additional dependencies will be added only when required and documented as the project develops.

## Project Structure

- `data/` — project datasets
- `notebooks/` — exploratory analysis and model development
- `src/` — reusable Python source code
- `models/` — trained model artefacts
- `app/` — application-related code

## Dataset

The project currently uses:

- `creditsense_applications.csv`
- `creditsense_data_dictionary.csv`

The main dataset contains 6,000 synthetic credit applications and 27 columns.

## Reproducibility

Dependencies are recorded in `requirements.txt`.

To create the virtual environment:

```powershell
python -m venv .venv