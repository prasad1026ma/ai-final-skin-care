from Classification.constants import CONDITIONS
from Classification.classification_pipeline import load_model
from Classification.data_cleaning import process_scin_dataset, load_dataset
from Classification.modeling.training import split_train_test
from Classification.skin_dataset import SkinDataset

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
from sklearn.preprocessing import label_binarize
class ModelEvaluator:
    """
    Evaluator Class for the Model
    """
    def __init__(self, model, device, class_names):
        self.model = model
        self.device = device
        self.class_names = class_names
        self.model.eval()

    def get_predictions(self, test_loader):
        all_preds = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                probabilities = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probabilities.cpu().numpy())

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)

        return all_labels, all_preds, all_probs

    def calculate_metrics(self, y_true, y_pred,y_probs):
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
            'precision_weighted': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
            'recall_weighted': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
            'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        }

        # Calculate AUC-ROC
        n_classes = len(np.unique(y_true))
        y_true_bin = label_binarize(y_true, classes=range(n_classes))
        metrics['auc_roc_macro'] = roc_auc_score(y_true_bin, y_probs, average='macro', multi_class='ovr')
        metrics['auc_roc_weighted'] = roc_auc_score(y_true_bin, y_probs, average='weighted', multi_class='ovr')
        return metrics

    def print_metrics(self, metrics):
        print("OVERALL PERFORMANCE METRICS")
        for metric_name, value in metrics.items():
            print(f"{metric_name:.<30} {value:.4f}")

    def plot_confusion_matrix(self, y_true, y_pred, save_path):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(12, 10))
        labels = self.class_names if self.class_names else range(len(cm))

        sns.heatmap(cm, annot=True, fmt='d',cmap='Blues', xticklabels=labels,
            yticklabels=labels, cbar_kws={'label': 'Count'})

        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved to {save_path}")

    def analyze_per_class_performance(self, y_true,y_pred):
        report = classification_report(y_true, y_pred, target_names=self.class_names, output_dict=True, zero_division=0)
        df = pd.DataFrame(report).transpose()

        df_classes = df.iloc[:-3]
        df_classes = df_classes.sort_values('f1-score')

        print("PER-CLASS PERFORMANCE")
        print(df_classes.to_string())

        weak_classes = df_classes[df_classes['f1-score'] < 0.7]
        if len(weak_classes) > 0:
            print("WEAK PERFORMING CLASSES:")
            print(weak_classes[['precision', 'recall', 'f1-score']].to_string())

        return df

    def plot_per_class_metrics(self,performance_df,save_path):
        df_classes = performance_df.iloc[:-3]

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        metrics = ['precision', 'recall', 'f1-score']

        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            df_sorted = df_classes.sort_values(metric, ascending=True)

            colors = ['red' if x < 0.7 else 'green' for x in df_sorted[metric]]

            ax.barh(range(len(df_sorted)), df_sorted[metric], color=colors, alpha=0.6)
            ax.set_yticks(range(len(df_sorted)))
            ax.set_yticklabels(df_sorted.index)
            ax.set_xlabel(metric.capitalize())
            ax.set_title(f'{metric.capitalize()} by Class')
            ax.axvline(x=0.7, color='red', linestyle='--', alpha=0.5, label='Threshold (0.7)')
            ax.legend()
            ax.set_xlim([0, 1])

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Per-class metrics plot saved to {save_path}")

    def plot_roc_curves(self, y_true, y_probs, save_path):
        """Plot ROC curves for each class."""
        n_classes = len(np.unique(y_true))
        y_true_bin = label_binarize(y_true, classes=range(n_classes))
        fig, ax = plt.subplots(figsize=(10, 8))

        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
            auc = roc_auc_score(y_true_bin[:, i], y_probs[:, i])
            label = self.class_names[i] if self.class_names else f'Class {i}'
            ax.plot(fpr, tpr, label=f'{label} (AUC = {auc:.2f})')

        ax.plot([0, 1], [0, 1], 'k--', label='Random')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(alpha=0.3)

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"ROC curves saved to {save_path}")


def evaluate_model(model_path, test_dataset, test_loader, num_classes, class_names= CONDITIONS,
                   output_dir='./evaluation_results'):

    os.makedirs(output_dir, exist_ok=True)
    model, device = load_model(
        model_path=model_path,
        num_classes=num_classes,
        device='mps'
    )
    print(f"Model loaded successfully on device: {device}")

    print(f"Test dataset: {len(test_dataset)} images")
    evaluator = ModelEvaluator(model, device, class_names=class_names)

    print("Generating predictions on test set:")
    y_true, y_pred, y_probs = evaluator.get_predictions(test_loader)

    metrics = evaluator.calculate_metrics(y_true, y_pred, y_probs)
    evaluator.print_metrics(metrics)

    evaluator.plot_confusion_matrix(
        y_true, y_pred, save_path=os.path.join(output_dir, 'confusion_matrix.png')
    )
    evaluator.plot_confusion_matrix(
        y_true, y_pred, save_path=os.path.join(output_dir, 'confusion_matrix_normalized.png')
    )

    print("Analyzing per-class performance...")
    performance_df = evaluator.analyze_per_class_performance(y_true, y_pred)
    evaluator.plot_per_class_metrics(
        performance_df,
        save_path=os.path.join(output_dir, 'per_class_metrics.png')
    )

    print("Plotting ROC curves...")
    evaluator.plot_roc_curves(
        y_true, y_probs,
        save_path=os.path.join(output_dir, 'roc_curves.png')
    )

    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(os.path.join(output_dir, 'metrics.csv'), index=False)
    performance_df.to_csv(os.path.join(output_dir, 'per_class_performance.csv'))

    return metrics, performance_df


# Example usage
if __name__ == "__main__":
    # Assuming you have these from your train/test split function
    dataset_path = process_scin_dataset(
        cases_csv_path='data/scin_cases.csv',
        labels_csv_path='data/scin_labels.csv',
        images_base_dir='data/images/',
        output_csv='data/dataset.csv'
    )
    image_paths, labels = load_dataset('data/dataset.csv')
    _,(test_paths, test_labels) = split_train_test(image_paths, labels)
    test_transform = SkinDataset.get_transforms(train=False, input_size=224)
    test_dataset = SkinDataset(
        image_paths=test_paths,
        labels=test_labels,
        transform=test_transform
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4
    )

    metrics, performance_df = evaluate_model(
        model_path='modeling/best_model.pth',
        test_dataset=test_dataset,
        test_loader=test_loader,
        num_classes=5,
        class_names=CONDITIONS,
        output_dir='./evaluation_results'
    )