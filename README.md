# CISB60_Final
Predicting loan defaults using financial and personal data
# Loan Default Prediction

This project aims to predict whether a borrower will default on their loan using financial and personal data. The predictions can help financial institutions reduce losses and improve lending strategies.

## Objective

The objective of this project is to predict loan default status based on a variety of financial features, including loan amount, interest rate, debt-to-income ratio, and credit history metrics. A binary target variable (`loan_status_binary`) is used, where:
- `0` indicates "Fully Paid"
- `1` indicates "Defaulted"

## Dataset

The dataset is sourced from LendingClub's public data and contains the following key columns:
- **Financial Information**: `loan_amnt`, `funded_amnt`, `int_rate`, `installment`, `annual_inc`, `dti`
- **Credit History**: `revol_bal`, `revol_util`, `inq_last_6mths`, `open_acc`, `total_acc`
- **Loan Status**: `loan_status` (mapped to `loan_status_binary`)

### Preprocessing
- **Missing Values**: Handled using median imputation for numeric columns.
- **Feature Transformation**: The `term` column was converted to numeric format.
- **Target Mapping**: The `loan_status` column was mapped to a binary target variable, excluding loans marked as "Current."

## Project Workflow

### Exploratory Data Analysis (EDA)
- Checked for missing values and handled them appropriately.
- Visualized relationships and patterns using histograms, correlation heatmaps, and boxplots.
- Excluded rows with non-relevant loan statuses (e.g., "Current").

### Machine Learning Model
- Implemented a **Random Forest Classifier** for loan default prediction.
- Conducted hyperparameter tuning using **GridSearchCV** with a reduced parameter grid for efficiency:
  - `n_estimators`: Number of trees in the forest.
  - `max_depth`: Maximum depth of the tree.
  - `min_samples_split`: Minimum number of samples required to split an internal node.
  - `min_samples_leaf`: Minimum number of samples required to be a leaf node.
- Evaluated the model using accuracy, precision, recall, and a confusion matrix.
- Visualized feature importance to interpret the Random Forest model.

### Deep Learning Model
- Built a simple **Neural Network** using TensorFlow/Keras with the following architecture:
  - Input layer with 64 neurons (ReLU activation)
  - Hidden layer with 32 neurons (ReLU activation)
  - Output layer with 1 neuron (Sigmoid activation for binary classification)
- Compiled the model with the Adam optimizer and binary cross-entropy loss.
- Trained for 20 epochs with a batch size of 32.
- Visualized training and validation accuracy/loss.

### Hyperparameter Optimization
- Used **GridSearchCV** for optimizing the Random Forest Classifier.
- Reduced computation time by:
  - Simplifying the parameter grid.
  - Reducing cross-validation folds to `cv=2`.

---

## Results

### Machine Learning Model
- Best Parameters: `{'n_estimators': 100, 'max_depth': 20, 'min_samples_split': 5, 'min_samples_leaf': 2}`
- Test Accuracy: *Displayed in the classification report*
- Feature importance highlights the most significant predictors of loan default.

### Deep Learning Model
- Training Accuracy: *Displayed in the training curve*
- Validation Accuracy: *Displayed in the training curve*

---

