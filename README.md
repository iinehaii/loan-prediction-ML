# Bank Loan Approval Prediction using Machine Learning

This repository contains the code and resources for a project aimed at predicting loan eligibility using machine learning (ML) algorithms. The project compares the performance of different ML algorithms in predicting whether a loan application will be approved, based on historical data.

## Project Overview

Bank loan approval is a critical process for financial institutions to assess the repayment ability of borrowers. Traditionally, manual underwriting and credit scoring models are used, but they have limitations. Machine learning (ML) has emerged as a powerful tool to improve the accuracy of these predictions. This project explores four ML algorithms and an ensemble learning method for predicting loan eligibility:

- Logistic Regression
- Decision Trees
- Random Forest
- XGBoost
- **Voting Algorithm** (Ensemble Learning)

### Dataset

The dataset was sourced from Analytics Vidhya and contains information on 614 loan applications. It includes 12 dependent variables and one independent variable, as described below:

- **Dependent Variables**: `Loan_ID`, `Gender`, `Married`, `Dependents`, `Education`, `Self_Employed`, `ApplicantIncome`, `CoapplicantIncome`, `LoanAmount`, `Loan_Amount_Term`, `Credit_History`, and `Property_Area`.
- **Independent Variable**: `Loan_Status` (whether the loan was approved or not).

### Problem Statement

The primary goal of this project is to predict the eligibility of a loan application based on the applicant's details. The model's prediction will help banks and financial institutions make more informed lending decisions and reduce the risk of default.

### Machine Learning Models Used

1. **Logistic Regression**: A simple and efficient algorithm for binary classification tasks.
2. **Decision Trees**: A non-linear classification model that splits the data into subsets based on feature values.
3. **Random Forest**: An ensemble of decision trees that improves prediction accuracy by averaging the results of multiple trees.
4. **XGBoost**: A powerful, efficient gradient boosting algorithm that performs well in many machine learning tasks.
5. **Voting Algorithm (Ensemble Learning)**: Combines predictions from multiple models to improve accuracy.

### Performance Evaluation

Each algorithm's performance was evaluated using the following metrics:

- **Accuracy**
- **Precision**
- **Recall**
- **F1 Score**

#### Results

- **Random Forest** was the most accurate model, with an accuracy score of **79.47%**.
- Using **Ensemble Learning (Voting Algorithm)**, the overall accuracy improved to **80.1%**.

## Project Structure

The repository includes the following files:

- `data/`: Contains the loan dataset used for training and testing.
- `notebooks/`: Google Colab notebooks detailing the data preprocessing, model training, evaluation, and results.
- `models/`: Serialized model files for logistic regression, decision trees, random forest, and XGBoost.
- `README.md`: This file.

## How to Run

1. Clone the repository:

    ```bash
    git clone https://github.com/your-username/loan-approval-prediction.git
    cd loan-approval-prediction
    ```

2. Open the Google Colab notebook:

    - You can access the notebook through Google Colab [here](https://colab.research.google.com/).
    
    - Upload the notebook from the `notebooks/` directory in the repository.

3. Upload the dataset (`data/`) to Colab or use the provided code to load it directly from the repository.

4. Run the cells in the notebook to preprocess the data, train the models, and evaluate the performance.

## Dependencies

- Google Colab (with in-built libraries such as Scikit-learn, Pandas, NumPy, etc.)
- XGBoost

To install additional dependencies, you can run:

```bash
!pip install xgboost
```

## Conclusion

This project demonstrates the potential of machine learning algorithms for predicting loan eligibility. The most effective models were Random Forest and Ensemble Learning (Voting Algorithm), achieving accuracy scores of **79.47%** and **80.1%**, respectively. While machine learning offers a significant improvement in loan approval predictions, it's important to remember that no model is foolproof, and external factors like macroeconomic conditions can still impact loan repayment.

## Future Work

- Experimenting with other machine learning algorithms.
- Incorporating additional features such as borrower credit scores and economic indicators.
- Exploring deep learning models for more complex prediction tasks.

## Acknowledgements

- Dataset provided by [Analytics Vidhya](https://www.analyticsvidhya.com/).


