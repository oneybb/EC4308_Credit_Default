import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

def calculate_lift_curve(y_true, y_prob):
    """
    Calculate lift curve coordinates
    Returns percentiles and lift values
    """
    # Sort by probability in descending order
    df = pd.DataFrame({'true': y_true, 'prob': y_prob})
    df = df.sort_values('prob', ascending=False)
    
    # Calculate cumulative response rate
    total_positives = df['true'].sum()
    baseline_rate = total_positives / len(df)
    
    # Calculate lift for different percentiles
    percentiles = np.arange(0.1, 1.1, 0.1)
    lift_values = []
    
    for p in percentiles:
        cutoff = int(len(df) * p)
        subset = df.iloc[:cutoff]
        response_rate = subset['true'].mean()
        lift = response_rate / baseline_rate
        lift_values.append(lift)
    
    return percentiles, lift_values

def plot_lift_curves(models_data, save_path=None):
    """
    Plot lift curves for multiple models
    models_data: dictionary with model names and their dataframes
    """
    plt.figure(figsize=(12, 8))
    colors = ['blue', 'red', 'green']
    
    # Calculate and plot lift curves for each model
    for (model_name, df), color in zip(models_data.items(), colors):
        percentiles, lift_values = calculate_lift_curve(df['actual'], df['probability'])
        plt.plot(percentiles, lift_values, marker='o', color=color, label=f'{model_name} (AUC={roc_auc_score(df["actual"], df["probability"]):.3f})')
    
    # Add reference line (lift = 1)
    plt.axhline(y=1, color='gray', linestyle='--', label='Random Model')
    
    plt.xlabel('Proportion of Population')
    plt.ylabel('Lift')
    plt.title('Lift Curve Comparison')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # Load the data
    try:
        logistic_df = pd.read_csv('results/lr.csv')
        xgb_df = pd.read_csv('results/xgb.csv')
        pcr_df = pd.read_csv('results/pcr.csv')
        
        # Create dictionary of models and their data
        models_data = {
            'Logistic Regression': logistic_df,
            'XGBoost': xgb_df,
            'PCR': pcr_df
        }
        
        # Create results directory if it doesn't exist
        import os
        os.makedirs('results', exist_ok=True)
        
        # Plot lift curves
        plot_lift_curves(models_data, save_path='results/lift_curves_comparison.png')
        
        # Print model performance metrics
        print("\nModel Performance Metrics:")
        print("-" * 50)
        for model_name, df in models_data.items():
            auc = roc_auc_score(df['actual'], df['probability'])
            percentiles, lift_values = calculate_lift_curve(df['actual'], df['probability'])
            max_lift = max(lift_values)
            print(f"\n{model_name}:")
            print(f"ROC-AUC Score: {auc:.4f}")
            print(f"Maximum Lift: {max_lift:.4f}")
            print(f"Lift at 10% population: {lift_values[0]:.4f}")
            print(f"Lift at 20% population: {lift_values[1]:.4f}")
            print(f"Lift at 50% population: {lift_values[4]:.4f}")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure all prediction files are in the correct location.")

if __name__ == "__main__":
    main() 