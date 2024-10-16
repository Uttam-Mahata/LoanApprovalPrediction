
# Loan Status Prediction with Machine Learning Models

## Overview

This project uses machine learning models to predict loan status based on a provided dataset. It applies three models—RandomForest, XGBoost, and LightGBM—and compares their performance using Stratified K-Fold Cross-Validation and ROC-AUC scores. The final trained models are used to generate predictions on the test dataset.

## Dataset

- `train.csv`: Training dataset, which includes features and the target variable (`loan_status`).
- `test.csv`: Test dataset, used for generating predictions. It contains the same features as the training set, excluding the target variable.

### Key Features in the Dataset
- `person_home_ownership`
- `loan_intent`
- `loan_grade`
- `cb_person_default_on_file`
- Other numerical features (e.g., `loan_amnt`, `person_income`)

## Project Structure

- `loan_status_prediction.py`: The main Python script that processes the data, trains the models, evaluates them, and generates predictions.
- `submission.csv`: Output file containing the final predictions for the RandomForest model on the test data.

## Requirements

The following Python libraries are required to run this project:

```bash
pandas==1.x.x
numpy==1.x.x
matplotlib==3.x.x
scikit-learn==1.x.x
xgboost==1.x.x
lightgbm==3.x.x
```

To install all dependencies, run:

```bash
pip install pandas numpy matplotlib scikit-learn xgboost lightgbm
```

## Steps to Run

1. **Preprocessing**:
    - Categorical features (`person_home_ownership`, `loan_intent`, `loan_grade`, `cb_person_default_on_file`) are encoded as numerical values for model compatibility.
    - The dataset is divided into features (`X`) and target (`y`), with ID columns excluded.

2. **Model Training and Cross-Validation**:
    - Three machine learning models are initialized: RandomForest, XGBoost, and LightGBM.
    - Stratified K-Fold Cross-Validation with 5 splits is applied to ensure a balanced representation of classes during training and testing phases.
    - ROC curves are plotted for each model, and the average AUC score is calculated.

3. **ROC Curve Plot**:
    - A plot is generated to visualize the ROC curves for RandomForest, XGBoost, and LightGBM models, helping compare their performance.

4. **Final Model Training and Prediction**:
    - Each model is retrained on the entire training dataset.
    - Predictions (probabilities) are made on the test dataset, and these are saved in `final_predictions` DataFrame.
    - The final submission file (`submission.csv`) contains predictions from the RandomForest model.

5. **Output**:
    - `submission.csv`: The submission file containing the predicted loan status probabilities for the test data using the RandomForest model.

## Usage

1. Place the `train.csv` and `test.csv` files in the project directory.
2. Run the script:

    ```bash
    python loan_status_prediction.py
    ```

3. After successful execution, a file named `submission.csv` will be created, which contains the predicted loan status for each entry in the test dataset using the RandomForest model.

## Results

- The AUC (Area Under the Curve) scores for RandomForest, XGBoost, and LightGBM models are displayed in the console after cross-validation.
- A ROC curve plot is generated to visualize the performance of each model.

## Customizations

- You can easily modify the script to use a different model (XGBoost or LightGBM) for the final submission by updating the last section where predictions are saved.


This README file provides instructions on how to use the project, its dependencies, and details on how the predictions are generated.
