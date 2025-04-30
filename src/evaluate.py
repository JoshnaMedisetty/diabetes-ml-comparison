from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, roc_curve)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.config import RESULTS_DIR

def evaluate_model(model, X_test, y_test, model_name):
    """Returns metrics and ROC data for a model"""
    y_pred = model.predict(X_test)
    
    # Get predicted probabilities
    if hasattr(model, 'predict_proba'):  # For non-DNN models
        y_proba = model.predict_proba(X_test)[:,1]
    else:  # For DNN
        y_proba = model.predict(X_test).flatten()
    
    # Calculate metrics
    metrics = {
        'Model': model_name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1': f1_score(y_test, y_pred),
        'ROC AUC': roc_auc_score(y_test, y_proba)
    }
    
    # Calculate ROC curve data
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_data = (fpr, tpr, metrics['ROC AUC'])
    
    return metrics, roc_data

def plot_comparisons(metrics_df, roc_data):
    """Generate and save comparison graphs"""
    # Metrics bar chart
    plt.figure(figsize=(12, 6))
    metrics_df.set_index('Model').plot(kind='bar', rot=45)
    plt.title('Model Performance Comparison')
    plt.ylabel('Score')
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(f'{RESULTS_DIR}/metric_comparison.png')
    plt.close()
    
    # Combined ROC curves
    plt.figure(figsize=(10, 8))
    for model_name, (fpr, tpr, auc) in roc_data.items():
        plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Combined ROC Curves')
    plt.legend()
    plt.savefig(f'{RESULTS_DIR}/combined_roc.png')
    plt.close()

def save_metrics(metrics_list):
    """Save metrics summary to CSV"""
    metrics_df = pd.DataFrame(metrics_list)
    metrics_df.to_csv(f'{RESULTS_DIR}/metrics_summary.csv', index=False)
    return metrics_df

def plot_comparison_bars(metrics_csv_path):
    df = pd.read_csv(metrics_csv_path)
    metrics_to_plot = ['Accuracy', 'F1', 'ROC AUC']  # Choose your key metrics

    for metric in metrics_to_plot:
        plt.figure(figsize=(8, 5))
        sns.barplot(x='Model', y=metric, data=df, palette='viridis')
        plt.ylabel(metric)
        plt.title(f'Model {metric} Comparison')
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.savefig(f'{RESULTS_DIR}/model_{metric.lower()}_comparison.png')
        plt.close()
