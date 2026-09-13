from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix

def calculate_metrics(y_true, y_probs):
    try:
        y_preds = [1 if p > 0.5 else 0 for p in y_probs]
        if len(set(y_true)) < 2: auc = 0.5
        else: auc = roc_auc_score(y_true, y_probs)
        tn, fp, fn, tp = confusion_matrix(y_true, y_preds, labels=[0, 1]).ravel()
        acc = accuracy_score(y_true, y_preds)
        sen = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spe = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        return acc, auc, spe, sen
    except: return 0.0, 0.5, 0.0, 0.0