# Credit Default Prediction

## Project Overview
This project aims to predict credit card default payments using machine learning techniques. The dataset contains information about credit card clients in Taiwan, including their payment history, bill amounts, and demographic information.

## Dataset Description
The dataset contains 30,000 instances with 25 features including:
- Demographic information (sex, education, marriage, age)
- Credit amount and payment history
- Bill statements for 6 months
- Previous payments for 6 months
- Target variable: default payment next month (Yes = 1, No = 0)

## Project Structure
```
.
├── data/
│   ├── UCI_Credit_Card.csv          # Original dataset
│   ├── X_train.csv                  # Training features
│   ├── X_val.csv                    # Validation features
│   ├── X_test.csv                   # Test features
│   ├── y_train.csv                  # Training labels
│   ├── y_val.csv                    # Validation labels
│   ├── y_test.csv                   # Test labels
│   └── pca.csv                      # PCA transformed data
├── notebooks/
│   ├── preprocessing.ipynb          # Data preprocessing and cleaning
│   ├── EDA.ipynb                    # Exploratory Data Analysis
│   ├── Baseline and Penalized Regression.ipynb  # Baseline models
│   ├── dimension_reduction.ipynb    # Dimension reduction techniques
│   ├── Tree_Based_Models.ipynb      # Tree-based models
│   └── evaluation.ipynb             # Model evaluation and comparison
├── fig/                            # Generated figures from analysis
│   ├── ROC.png                     # ROC curves
│   ├── ROC_test.png                # ROC curves on test set
│   ├── corrmap.png                 # Correlation heatmap
│   └── dt_cost tuned.png           # Decision tree cost tuning results
├── results/                        # Model results and visualizations
│   ├── lr.csv                      # Logistic Regression predictions
│   ├── xgb.csv                     # XGBoost predictions
│   ├── pcr.csv                     # PCR predictions
│   ├── lift_curves_comparison.png  # Comparison of model lift curves
comparison
│   └── feature_importance_*.png    # Various feature importance plots
├── compare_lift_curves.py          # Script for comparing model performance
└── README.md                       # Project documentation
```

## Methodology

### 1. Data Preprocessing
- Removed ID column (not useful for modeling)
- Cleaned education and marriage variables
  - Education: Mapped values 0, 5, 6 to 4 (Others)
  - Marriage: Mapped value 0 to 3 (Others)
- Split data into training (60%), validation (20%), and test (20%) sets
- Standardized numerical features

### 2. Exploratory Data Analysis
- Performed descriptive statistics analysis
- Analyzed correlations between features
- Created correlation heatmaps
- Conducted collinearity analysis using Variance Inflation Factor (VIF)

### 3. Feature Engineering and Dimension Reduction
- Identified highly correlated features
- Applied Principal Component Analysis (PCA)
- Used feature importance analysis from tree-based models

### 4. Modeling Approaches
- Baseline Models:
  - Logistic Regression
  - Penalized Regression (Lasso, Ridge, Elastic Net)
- Tree-based Models:
  - Decision Trees
  - Random Forest
  - Gradient Boosting
  - XGBoost
- Principal Component Regression (PCR)

### 5. Model Evaluation and Comparison
- Implemented comprehensive model evaluation using:
  - ROC-AUC scores
  - Lift curves analysis
  - Population-wise lift comparison
- Key metrics for model comparison:
  - Maximum lift achieved
  - Lift at different population percentiles (10%, 20%, 50%)
  - Overall ROC-AUC performance

## Key Findings
- The dataset shows significant correlations between bill amounts across different months
- Payment history (PAY_0 to PAY_6) shows moderate correlations with default probability
- Demographic features (age, education, marriage) have relatively weak correlations with default probability
- Tree-based models generally performed better than linear models
- Model Performance Comparison:
  - XGBoost achieved the highest ROC-AUC score and maximum lift
  - Logistic Regression and PCR showed competitive performance at different population percentiles
  - All models significantly outperformed random prediction (lift > 1)

## Dependencies
- Python 3.9+
- pandas
- numpy
- scikit-learn
- xgboost
- matplotlib
- seaborn
- statsmodels

## Usage
1. Clone the repository
2. Install the required dependencies
3. Run the notebooks in the following order:
   - preprocessing.ipynb
   - EDA.ipynb
   - Baseline and Penalized Regression.ipynb
   - dimension_reduction.ipynb
   - Tree_Based_Models.ipynb
   - evaluation.ipynb
4. To compare model performance:
   - Ensure model predictions are saved in the results directory
   - Run `python compare_lift_curves.py` to generate lift curve comparisons
   - View the results in `results/lift_curves_comparison.png`

## Future Improvements
- Feature engineering to create more predictive variables
- Experiment with other advanced models (Neural Networks, etc.)
- Implement cross-validation for more robust model evaluation
- Analyze feature interactions
- Deploy model as a web service



