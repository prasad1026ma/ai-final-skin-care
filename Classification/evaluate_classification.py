from Classification.utilities.constants import CONDITIONS
from Classification.classification_pipeline import load_model
from Classification.utilities.data_cleaning import process_scin_dataset, load_dataset
from Classification.modeling.training import split_train_val_test
from Classification.utilities.skin_dataset import SkinDataset

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import DataLoader
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
        """
        Extract out model predictions and probabilities
        """
        all_preds = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            # iterate over all images and labels in the test loader
            for images, labels in test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                # extract the predicted label and the probability for that label
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
        """
        Calculate the model specific metrics based on test set
        """
        # calculate the evaluation metrics using the sklearn metric functionalities
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
            'precision_weighted': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
            'recall_weighted': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
            'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        }

        # calculate AUC-ROC values
        y_true_bin = label_binarize(y_true, classes=range(len(np.unique(y_true))))
        metrics['auc_roc_macro'] = roc_auc_score(y_true_bin, y_probs, average='macro', multi_class='ovr')
        metrics['auc_roc_weighted'] = roc_auc_score(y_true_bin, y_probs, average='weighted', multi_class='ovr')
        return metrics

    def plot_confusion_matrix(self, y_true, y_pred, save_path):
        """
        Plot the confusion matrix based on y true vs y predicted
        """
        # create the confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(12, 10))

        # plot the confusion matrix using seaborn
        sns.heatmap(cm, annot=True, fmt='d',cmap='Blues', xticklabels=self.class_names,
            yticklabels=self.class_names, cbar_kws={'label': 'Count'})

        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")

    def analyze_per_class_performance(self, y_true,y_pred):
        """
        Examine per class performance based on sklearns classification method
        """
        # create a report (returns a dict) of the models performance
        report = classification_report(y_true, y_pred, target_names=self.class_names,
                                       output_dict=True, zero_division=0)
        df = pd.DataFrame(report).transpose()

        # extract out the classes names from the dataframe and sort by f1-score
        df_classes = df.iloc[:-3]
        df_classes = df_classes.sort_values('f1-score')

        print("Per-class Performance")
        print(df_classes.to_string())

        return df

    def plot_per_class_metrics(self, performance_df, save_path):
        """
        Plot out the per class metrics into bar charts
        """
        df_classes = performance_df.iloc[:-3]
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        metrics = ['precision', 'recall', 'f1-score']

        # iterate over every metric and create a plot to to compare classes
        # red bars means metric is under a threshold and green means over
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

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Per-class metrics plot saved to {save_path}")

    def plot_roc_curves(self, y_true, y_probs, save_path):
        """Plot ROC curves for each class."""
        # Get number of unique classes
        n_classes = len(np.unique(y_true))

        # Binarize the labels for one-vs-rest ROC calculation
        y_true_bin = label_binarize(y_true, classes=range(n_classes))

        # Handle edge case where there's only one class in y_true
        if y_true_bin.shape[1] == 1:
            print("Warning: Only one class present in y_true. Cannot generate ROC curves.")
            return

        fig, ax = plt.subplots(figsize=(10, 8))

        # Plot ROC curve for each class
        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
            auc = roc_auc_score(y_true_bin[:, i], y_probs[:, i])
            ax.plot(fpr, tpr, label=f'{self.class_names[i]} (AUC = {auc:.2f})', linewidth=2)

        # Plot random classifier line
        ax.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=2)

        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title('ROC Curves - Multi-Class Classification', fontsize=14, fontweight='bold')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        ax.grid(alpha=0.3)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"ROC curves saved to {save_path}")
        plt.close()


def evaluate_model(model_path, test_dataset, test_loader, num_classes, class_names= CONDITIONS,
                   output_dir='./evaluation_results'):
    """
    Runs the model evaluation pipeline and saves graphs to evaluation_results
    """
    # create a new directory to save the outputted results
    os.makedirs(output_dir, exist_ok=True)

    # load the model saved in the inputted model path
    try:
        model, device = load_model(
            model_path=model_path,
            num_classes=num_classes,
            device='mps'
        )
        print(f"Model loaded successfully on device: {device}")
    except Exception as e:
        print(f"Error in loading the model due to {e}")
        model, device = None, None

    print(f"Test dataset: {len(test_dataset)} images")

    evaluator = ModelEvaluator(model, device, class_names=class_names)

    # generate the prediction on the test set
    y_true, y_pred, y_probs = evaluator.get_predictions(test_loader)

    # calculate the metrics based on the predictions from the test set
    metrics = evaluator.calculate_metrics(y_true, y_pred, y_probs)

    print("Performance Metrics")
    for metric_name, value in metrics.items():
        print(f"{metric_name:.<30} {value:.4f}")

    # plot the confusion matrix
    evaluator.plot_confusion_matrix(
        y_true, y_pred, save_path=os.path.join(output_dir, 'confusion_matrix.png')
    )

    # analyze and plot the per class performance metrics
    performance_df = evaluator.analyze_per_class_performance(y_true, y_pred)
    evaluator.plot_per_class_metrics(
        performance_df,
        save_path=os.path.join(output_dir, 'per_class_metrics.png')
    )

    # plot the ROC curve
    evaluator.plot_roc_curves(
        y_true, y_probs,
        save_path=os.path.join(output_dir, 'roc_curves.png')
    )

    # save the metrics to a csv
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(os.path.join(output_dir, 'metrics.csv'), index=False)
    performance_df.to_csv(os.path.join(output_dir, 'per_class_performance.csv'))

    return metrics, performance_df


# Example usage
if __name__ == "__main__":
    dataset_path = process_scin_dataset(
        cases_csv_path='data/scin_cases.csv',
        labels_csv_path='data/scin_labels.csv',
        images_base_dir='data/images/',
        output_csv='data/dataset.csv'
    )
    image_paths, labels = load_dataset('data/dataset.csv')
    _,_,(test_paths, test_labels) = split_train_val_test(image_paths, labels)
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