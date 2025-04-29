
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report, roc_curve)
import pandas as pd
import matplotlib.pyplot as plt
import shap
from config import RESULTS_DIR, FEATURE_NAMES

def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    if hasattr(model, 'predict_proba'):  # For non-DNN models
        y_proba = model.predict_proba(X_test)[:,1]
    else:  # For DNN
        y_proba = model.predict(X_test)
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_proba)
    }
    
    # Save reports
    save_reports(y_test, y_pred, y_proba, model_name)
    return metrics

def save_reports(y_test, y_pred, y_proba, model_name):
    # Classification report
    report = classification_report(y_test, y_pred, output_dict=True)
    pd.DataFrame(report).transpose().to_csv(f'{RESULTS_DIR}/{model_name}_report.csv')
    
    # Confusion matrix
    plt.figure(figsize=(8,6))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
    plt.title(f'{model_name} Confusion Matrix')
    plt.savefig(f'{RESULTS_DIR}/{model_name}_confusion.png')
    plt.close()
    
    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.figure()
    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc_score(y_test, y_proba):.2f})')
    plt.plot([0,1], [0,1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.savefig(f'{RESULTS_DIR}/{model_name}_roc.png')
    plt.close()

