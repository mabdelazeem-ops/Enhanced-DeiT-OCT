import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

def evaluate_model_performance(y_true, y_pred_probs, class_names):
    """Calculates evaluation metrics: Accuracy, Precision, Recall, F1-Score, and AUC."""
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true_labels = np.argmax(y_true, axis=1)
    
    print("\n--- Quantitative Classification Report ---")
    print(classification_report(y_true_labels, y_pred, target_names=class_names, digits=4))
    
    auc = roc_auc_score(y_true, y_pred_probs, multi_class="ovr")
    print(f"Mean Macro Area Under ROC (AUC): {auc:.4f}")
    
    cm = confusion_matrix(y_true_labels, y_pred)
    return cm, auc
