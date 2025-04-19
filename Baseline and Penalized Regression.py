# Credit Card Default Prediction Analysis - Improved Version
print("running...")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, roc_curve, precision_recall_curve, average_precision_score
from sklearn.feature_selection import SequentialFeatureSelector, SelectFromModel, RFE
import joblib
import os
import time
from datetime import datetime
import warnings
from sklearn.base import clone
from sklearn.utils import resample
from scipy import stats
warnings.filterwarnings('ignore')

# Create a directory for results if it doesn't exist
results_dir = f"results"
os.makedirs(results_dir, exist_ok=True)

# Set random seed for reproducibility
np.random.seed(42)

# Load the data
print("Loading data...")
credit_data = pd.read_csv("UCI_Credit_Card.csv")

# Basic information about the dataset
print("\nBasic Information:")
print(f"Shape of dataset: {credit_data.shape}")
print("\nFirst few rows:")
print(credit_data.head())

# Check for missing values
print("\nMissing values in each column:")
print(credit_data.isnull().sum())

# ==== Data Preprocessing ====
print("\n==== Data Preprocessing ====")

# Drop ID column (not useful for modeling)
credit_data = credit_data.drop('ID', axis=1)

# Check education and marriage for unusual values
print("\nUnique values in EDUCATION:", credit_data['EDUCATION'].unique())
print("Unique values in MARRIAGE:", credit_data['MARRIAGE'].unique())

# Fix education and marriage variables
# For education: 0, 5, 6 are mapped to 4 (Others)
# For marriage: 0 is mapped to 3 (Others)
credit_data['EDUCATION'] = credit_data['EDUCATION'].map(lambda x: 4 if x in [0, 5, 6] else x)
credit_data['MARRIAGE'] = credit_data['MARRIAGE'].map(lambda x: 3 if x == 0 else x)

print("\nAfter cleaning:")
print("Unique values in EDUCATION:", credit_data['EDUCATION'].unique())
print("Unique values in MARRIAGE:", credit_data['MARRIAGE'].unique())

# Feature and target variables
X = credit_data.drop('default.payment.next.month', axis=1)
y = credit_data['default.payment.next.month']

# Split data into training, validation and test sets (60%, 20%, 20%)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

print(f"Training set shape: {X_train.shape}")
print(f"Validation set shape: {X_val.shape}")
print(f"Test set shape: {X_test.shape}")

# Save the cleaned and split datasets
print("\nSaving cleaned and split datasets...")
datasets = {
    'train': (X_train, y_train),
    'validation': (X_val, y_val),
    'test': (X_test, y_test)
}

for split_name, (X, y) in datasets.items():
    # Combine features and target for saving
    combined_data = pd.concat([X, y.rename('target')], axis=1)
    # Save to CSV
    output_path = os.path.join(results_dir, f'{split_name}_set.csv')
    combined_data.to_csv(output_path, index=False)
    print(f"Saved {split_name} set to: {output_path}")

# Standardize numerical features
scaler = StandardScaler()
numerical_cols = ['LIMIT_BAL', 'AGE'] + [col for col in X.columns if col.startswith('BILL_') or col.startswith('PAY_AMT')]
X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
X_val[numerical_cols] = scaler.transform(X_val[numerical_cols])
X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

# Define the CV strategy - use stratified k-fold to handle class imbalance
cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Function to evaluate and report model performance
def evaluate_model(model, X_train, y_train, X_val, y_val, model_name, selected_features=None):
    """Evaluate model performance with multiple metrics and visualizations"""
    start_time = time.time()
    
    # Use selected features if provided
    X_train_use = X_train[selected_features] if selected_features is not None else X_train
    X_val_use = X_val[selected_features] if selected_features is not None else X_val
    
    # Cross-validation on training data
    cv_scores = cross_val_score(model, X_train_use, y_train, cv=cv_strategy, scoring='roc_auc')
    
    # Fit the model on training data
    model.fit(X_train_use, y_train)
    
    # Predictions on validation data
    y_pred_proba = model.predict_proba(X_val_use)[:, 1]
    y_pred = model.predict(X_val_use)
    
    # Compute various metrics
    roc_auc = roc_auc_score(y_val, y_pred_proba)
    avg_precision = average_precision_score(y_val, y_pred_proba)
    
    # Get ROC curve data
    fpr, tpr, _ = roc_curve(y_val, y_pred_proba)
    
    # Classification report
    report = classification_report(y_val, y_pred)
    
    # Confusion matrix
    cm = confusion_matrix(y_val, y_pred)
    
    # Training time
    training_time = time.time() - start_time
    
    print(f"\n{model_name} Results:")
    print(f"Training time: {training_time:.2f} seconds")
    print(f"CV ROC-AUC Scores: {cv_scores}")
    print(f"Mean CV ROC-AUC: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
    print(f"Validation ROC-AUC: {roc_auc:.4f}")
    print(f"Validation Avg Precision: {avg_precision:.4f}")
    
    print("\nClassification Report:")
    print(report)
    
    print("\nConfusion Matrix:")
    print(cm)
    
    # Plot individual ROC curve
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend()
    plt.savefig(os.path.join(results_dir, f'roc_curve_{model_name.replace(" ", "_").lower()}.png'))
    plt.close()
    
    # For models with coefficients, plot feature importance
    if hasattr(model, 'coef_'):
        features = selected_features if selected_features is not None else X_train.columns
        coef = model.coef_[0]
        
        # Sort coefficients and feature names
        indices = np.argsort(np.abs(coef))
        sorted_features = np.array(features)[indices]
        sorted_coef = coef[indices]
        
        plt.figure(figsize=(12, 10))
        plt.barh(range(len(sorted_coef)), sorted_coef)
        plt.yticks(range(len(sorted_coef)), sorted_features)
        plt.xlabel('Coefficient magnitude')
        plt.title(f'Feature Importance - {model_name}')
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f'feature_importance_{model_name.replace(" ", "_").lower()}.png'))
        plt.close()
    
    # Return key metrics, model, and ROC curve data
    return {'avg_precision': avg_precision, 'roc_auc': roc_auc, 'model': model, 'roc_data': (fpr, tpr)}

# Dictionary to store models and their performance
models = {}

# Function to tune hyperparameters
def tune_hyperparameters(model, param_grid, X_train, y_train, cv=None, scoring='roc_auc'):
    """Tune hyperparameters using grid search and cross-validation"""
    if cv is None:
        cv = cv_strategy
        
    grid_search = GridSearchCV(
        model, param_grid, cv=cv, scoring=scoring, n_jobs=-1, verbose=1
    )
    grid_search.fit(X_train, y_train)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best CV score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_

print("\n==== Model Building and Evaluation ====")

# 1. Baseline Logistic Regression
print("\nTraining baseline Logistic Regression...")
baseline_model = LogisticRegression(random_state=42, max_iter=2000)
models['Baseline Logistic Regression'] = evaluate_model(baseline_model, X_train, y_train, X_val, y_val, "Baseline Logistic Regression")

# 2. Ridge Logistic Regression (L2 regularization) with hyperparameter tuning
print("\nTraining Ridge Logistic Regression with Hyperparameter Tuning...")
ridge_param_grid = {
    'C': [0.001, 0.01, 0.1, 1.0, 10.0],
    'class_weight': [None, 'balanced', {0: 1, 1: 3}, {0: 1, 1: 4}]
}
ridge_base = LogisticRegression(penalty='l2', random_state=42, max_iter=2000)
ridge_tuned = tune_hyperparameters(ridge_base, ridge_param_grid, X_train, y_train)
models['Ridge Logistic Regression (Tuned)'] = evaluate_model(ridge_tuned, X_train, y_train, X_val, y_val, "Ridge Logistic Regression (Tuned)")

# 3. Lasso Logistic Regression (L1 regularization) with hyperparameter tuning
print("\nTraining Lasso Logistic Regression with Hyperparameter Tuning...")
lasso_param_grid = {
    'C': [0.001, 0.01, 0.1, 1.0, 10.0],
    'class_weight': [None, 'balanced', {0: 1, 1: 3}, {0: 1, 1: 4}]
}
lasso_base = LogisticRegression(penalty='l1', solver='liblinear', random_state=42, max_iter=2000)
lasso_tuned = tune_hyperparameters(lasso_base, lasso_param_grid, X_train, y_train)
models['Lasso Logistic Regression (Tuned)'] = evaluate_model(lasso_tuned, X_train, y_train, X_val, y_val, "Lasso Logistic Regression (Tuned)")

# 4. Elastic Net Logistic Regression (L1 + L2 regularization) with hyperparameter tuning
print("\nTraining Elastic Net Logistic Regression with Hyperparameter Tuning...")
elastic_net_param_grid = {
    'C': [0.001, 0.01, 0.1, 1.0],
    'l1_ratio': [0.2, 0.5, 0.8],
    'class_weight': [None, 'balanced', {0: 1, 1: 3}, {0: 1, 1: 4}]
}
elastic_net_base = LogisticRegression(penalty='elasticnet', solver='saga', random_state=42, max_iter=2000)
elastic_net_tuned = tune_hyperparameters(elastic_net_base, elastic_net_param_grid, X_train, y_train)
models['Elastic Net Logistic Regression (Tuned)'] = evaluate_model(elastic_net_tuned, X_train, y_train, X_val, y_val, "Elastic Net Logistic Regression (Tuned)")

# 5. Forward Selection with different class weights
print("\nTraining Forward Selection with different class weights...")
class_weights = [None, 'balanced', {0: 1, 1: 3}, {0: 1, 1: 4}]
best_forward_auc = 0
best_forward_model = None
best_forward_weight = None

for weight in class_weights:
    print(f"\nTrying Forward Selection with class_weight: {weight}")
    forward_selector = SequentialFeatureSelector(
        LogisticRegression(random_state=42, max_iter=2000, class_weight=weight),
        n_features_to_select=10,
        direction='forward',
        scoring='roc_auc',
        cv=3
    )
    forward_selector.fit(X_train, y_train)
    forward_features = X_train.columns[forward_selector.get_support()]
    print(f"Selected features (Forward): {forward_features.tolist()}")
    
    # Train model with selected features
    forward_model = LogisticRegression(random_state=42, max_iter=2000, class_weight=weight)
    results = evaluate_model(forward_model, X_train, y_train, X_val, y_val, f"Forward Selection (weight={weight})", forward_features)
    
    if results['roc_auc'] > best_forward_auc:
        best_forward_auc = results['roc_auc']
        best_forward_model = forward_model
        best_forward_weight = weight
        best_forward_features = forward_features

models['Forward Selection'] = {'model': best_forward_model, 'roc_auc': best_forward_auc, 'features': best_forward_features}
print(f"\nBest Forward Selection class weight: {best_forward_weight} with ROC-AUC: {best_forward_auc:.4f}")

# 6. Backward Selection with different class weights
print("\nTraining Backward Selection with different class weights...")
class_weights = [None, 'balanced', {0: 1, 1: 3}, {0: 1, 1: 4}]
best_backward_auc = 0
best_backward_model = None
best_backward_weight = None

for weight in class_weights:
    print(f"\nTrying Backward Selection with class_weight: {weight}")
    
    # Initialize variables to store models and their metrics
    models_by_k = {}
    features_remaining = X_train.columns.tolist()
    n_features = len(features_remaining)
    
    # For each k from p down to 1
    for k in range(n_features, 0, -1):
        # Fit model with current features
        model = LogisticRegression(random_state=42, max_iter=2000, class_weight=weight)
        model.fit(X_train[features_remaining], y_train)
        
        # Calculate t-statistics for each feature
        y_pred = model.predict(X_train[features_remaining])
        residuals = y_train - y_pred
        
        # Calculate standard errors using bootstrap
        n_bootstrap = 100
        coef_samples = np.zeros((n_bootstrap, len(features_remaining)))
        
        for i in range(n_bootstrap):
            # Bootstrap sample indices
            indices = np.random.choice(len(y_train), len(y_train), replace=True)
            # Fit model on bootstrap sample
            model_boot = LogisticRegression(random_state=42, max_iter=2000, class_weight=weight)
            model_boot.fit(X_train[features_remaining].iloc[indices], y_train.iloc[indices])
            coef_samples[i, :] = model_boot.coef_[0]
        
        # Calculate standard errors
        std_errors = np.std(coef_samples, axis=0)
        t_stats = np.abs(model.coef_[0] / std_errors)
        
        # Store current model and its cross-validated metrics
        cv_scores = cross_val_score(model, X_train[features_remaining], y_train, 
                                  cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
                                  scoring='roc_auc')
        
        models_by_k[k] = {
            'features': features_remaining.copy(),
            'model': clone(model),
            'cv_roc_auc_mean': cv_scores.mean(),
            'cv_roc_auc_std': cv_scores.std()
        }
        
        # If we need to remove more features
        if k > 1:
            # Find feature with smallest absolute t-statistic
            min_t_idx = np.argmin(t_stats)
            # Remove that feature
            removed_feature = features_remaining.pop(min_t_idx)
            print(f"Dropped feature '{removed_feature}' (t-stat: {t_stats[min_t_idx]:.4f})")
    
    # Select best model using cross-validated ROC-AUC
    best_k = max(models_by_k.keys(), key=lambda k: models_by_k[k]['cv_roc_auc_mean'])
    best_model = models_by_k[best_k]['model']
    selected_features = models_by_k[best_k]['features']
    
    print(f"\nBest model for this weight has {best_k} features:")
    print(f"Selected features: {selected_features}")
    print(f"Mean CV ROC-AUC: {models_by_k[best_k]['cv_roc_auc_mean']:.4f} (±{models_by_k[best_k]['cv_roc_auc_std']:.4f})")
    
    # Evaluate on validation set
    results = evaluate_model(best_model, X_train, y_train, X_val, y_val, 
                           f"Backward Selection (weight={weight})", selected_features)
    
    if results['roc_auc'] > best_backward_auc:
        best_backward_auc = results['roc_auc']
        best_backward_model = best_model
        best_backward_weight = weight
        best_backward_features = selected_features

models['Backward Selection'] = {'model': best_backward_model, 'roc_auc': best_backward_auc, 'features': best_backward_features}
print(f"\nBest Backward Selection class weight: {best_backward_weight} with ROC-AUC: {best_backward_auc:.4f}")

# After all models are evaluated and before finding the best model, add:
print("\n==== Creating Combined ROC Curve Plot ====")
plt.figure(figsize=(12, 10))
colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']

# Store best models and their metrics for comparison
best_models = {
    'Baseline LR': baseline_model,
    'Ridge (w={0: 1, 1: 4})': ridge_tuned,
    'Lasso (w={0: 1, 1: 4})': lasso_tuned,
    'Elastic Net (w={0: 1, 1: 4})': elastic_net_tuned,
    'Forward Selection (no weights)': best_forward_model,
    'Backward Selection (w={0: 1, 1: 3})': best_backward_model
}

# Plot ROC curves for best models
for (model_name, model), color in zip(best_models.items(), colors):
    if model_name.startswith('Forward Selection'):
        y_pred_proba = model.predict_proba(X_val[best_forward_features])[:, 1]
    elif model_name.startswith('Backward Selection'):
        y_pred_proba = model.predict_proba(X_val[best_backward_features])[:, 1]
    else:
        y_pred_proba = model.predict_proba(X_val)[:, 1]
    
    fpr, tpr, _ = roc_curve(y_val, y_pred_proba)
    roc_auc = roc_auc_score(y_val, y_pred_proba)
    plt.plot(fpr, tpr, color=color, label=f'{model_name} (AUC = {roc_auc:.4f})')

plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Combined ROC Curves for Best Models')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'combined_roc_curves_best_models.png'), bbox_inches='tight', dpi=300)
plt.close()

# Create comparison chart for best models
print("\n==== Model Comparison for Best Models ====")
model_metrics = []

for model_name, model in best_models.items():
    if model_name.startswith('Forward Selection'):
        y_pred_proba = model.predict_proba(X_val[best_forward_features])[:, 1]
    elif model_name.startswith('Backward Selection'):
        y_pred_proba = model.predict_proba(X_val[best_backward_features])[:, 1]
    else:
        y_pred_proba = model.predict_proba(X_val)[:, 1]
    
    roc_auc = roc_auc_score(y_val, y_pred_proba)
    avg_precision = average_precision_score(y_val, y_pred_proba)
    model_metrics.append({
        'Model': model_name,
        'ROC-AUC': roc_auc,
        'Avg Precision': avg_precision
    })

# Sort models by ROC-AUC performance
model_metrics = sorted(model_metrics, key=lambda x: x['ROC-AUC'], reverse=True)

# Print metrics
print("\nModels ranked by ROC-AUC performance:")
for i, metrics in enumerate(model_metrics, 1):
    print(f"{i}. {metrics['Model']}: ROC-AUC = {metrics['ROC-AUC']:.4f}, Avg Precision = {metrics['Avg Precision']:.4f}")

# Create bar plot comparing models
plt.figure(figsize=(14, 8))
models_names = [m['Model'] for m in model_metrics]
roc_auc_scores = [m['ROC-AUC'] for m in model_metrics]

bars = plt.bar(models_names, roc_auc_scores)
plt.title('Model Comparison - ROC-AUC Scores (Best Models)')
plt.xlabel('Models')
plt.ylabel('ROC-AUC Score')
plt.ylim(0.702, 0.708)  # Adjusted y-axis limits to zoom in on the differences
plt.xticks(rotation=45, ha='right')
bars[0].set_color('red')  # Highlight the best model
plt.grid(axis='y', linestyle='--', alpha=0.7)  # Added more visible grid
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'model_comparison_best_models.png'))
plt.close()

# Find the overall best model
best_model_metrics = max(model_metrics, key=lambda x: x['ROC-AUC'])
best_model_name = best_model_metrics['Model']
best_auc = best_model_metrics['ROC-AUC']

print(f"\n==== Best Model: {best_model_name} with ROC-AUC {best_auc:.4f} ====")

# ==== Final Evaluation on Test Set ====
print("\n==== Final Evaluation on Test Set ====")

# Get the best model and its features
best_model = best_models[best_model_name]
selected_features = None
if best_model_name.startswith('Forward Selection'):
    selected_features = best_forward_features
elif best_model_name.startswith('Backward Selection'):
    selected_features = best_backward_features

# For the best model, evaluate on the test set
if selected_features is not None:
    print(f"\nUsing {len(selected_features)} selected features for final evaluation")
    results = evaluate_model(best_model, X_train, y_train, X_test, y_test, f"{best_model_name} (Test)", selected_features)
else:
    results = evaluate_model(best_model, X_train, y_train, X_test, y_test, f"{best_model_name} (Test)")

# Save the best model
joblib.dump(best_model, os.path.join(results_dir, 'best_model.pkl'))

print(f"\nAnalysis complete. Results and visualizations have been saved to the '{results_dir}' directory.") 