# utils/plotting.py
import matplotlib.pyplot as plt


def plot_training_curves(history, save_path):
    if len(history['train_loss']) == 0:
        return

    epochs = range(1, len(history['train_loss']) + 1)
    plt.figure(figsize=(24, 4))

    metrics_config = [
        ('Loss', 'train_loss', None, 'Loss'),
        ('AUC', 'train_auc', 'val_auc', 'AUC'),
        ('Accuracy', 'train_acc', 'val_acc', 'Accuracy'),
        ('Sensitivity', 'train_sen', 'val_sen', 'Sensitivity'),
        ('Specificity', 'train_spe', 'val_spe', 'Specificity'),
        ('Gap (Train - Val)', 'auc_gap', None, 'Gap (Lower is Better)')
    ]

    for idx, (title, train_key, val_key, ylabel) in enumerate(metrics_config):
        plt.subplot(1, 6, idx + 1)

        if train_key in history and len(history[train_key]) == len(epochs):
            color = 'purple' if train_key == 'auc_gap' else 'b'
            marker = 'x' if train_key == 'auc_gap' else '^'
            label = 'AUC Gap' if train_key == 'auc_gap' else f'Train {ylabel}'
            plt.plot(epochs, history[train_key], color=color, linestyle='--', marker=marker, label=label, alpha=0.7)

            if train_key == 'auc_gap':
                plt.axhline(y=0, color='green', linestyle=':', alpha=0.5, label="Ideal (0)")

        if val_key and val_key in history and len(history[val_key]) == len(epochs):
            plt.plot(epochs, history[val_key], 'r-s', label=f'Val {ylabel}', alpha=0.9)

        plt.title(title, fontsize=11, fontweight='bold')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)