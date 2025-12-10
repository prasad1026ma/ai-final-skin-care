import matplotlib.pyplot as plt
import json


def save_history(history, filepath):
    """Save training history to JSON file"""
    with open(filepath, 'w') as f:
        json.dump(history, f, indent=4)
    print(f"Saved to {filepath}")


def load_history(filepath):
    """Load training history from JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)


def plot_training_curves(history, model_name="Model", save_path=None):
    """
    Plot accuracy and loss curves for one model

    Args:
        history: dict with keys 'train_acc', 'val_acc', 'train_loss', 'val_loss'
        model_name: Name for the title
        save_path: Where to save (optional)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(history['train_acc']) + 1)

    # Accuracy plot
    ax1.plot(epochs, history['train_acc'], 'b-', label='Train', linewidth=2)
    ax1.plot(epochs, history['val_acc'], 'r-', label='Validation', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title(f'{model_name} - Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Loss plot
    ax2.plot(epochs, history['train_loss'], 'b-', label='Train', linewidth=2)
    ax2.plot(epochs, history['val_loss'], 'r-', label='Validation', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title(f'{model_name} - Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')


def plot_comparison(cnn_history, resnet_history, save_path=None):
    """
    Compare two models side by side

    Args:
        cnn_history: Custom CNN history dict
        resnet_history: ResNet history dict
        save_path: Where to save (optional)
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    cnn_epochs = range(1, len(cnn_history['train_acc']) + 1)
    resnet_epochs = range(1, len(resnet_history['train_acc']) + 1)

    # Training Accuracy
    axes[0, 0].plot(cnn_epochs, cnn_history['train_acc'], 'b-', label='Custom CNN', linewidth=2)
    axes[0, 0].plot(resnet_epochs, resnet_history['train_acc'], 'r-', label='ResNet-18', linewidth=2)
    axes[0, 0].set_title('Training Accuracy')
    axes[0, 0].set_ylabel('Accuracy (%)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Validation Accuracy
    axes[0, 1].plot(cnn_epochs, cnn_history['val_acc'], 'b-', label='Custom CNN', linewidth=2)
    axes[0, 1].plot(resnet_epochs, resnet_history['val_acc'], 'r-', label='ResNet-18', linewidth=2)
    axes[0, 1].set_title('Validation Accuracy')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Training Loss
    axes[1, 0].plot(cnn_epochs, cnn_history['train_loss'], 'b-', label='Custom CNN', linewidth=2)
    axes[1, 0].plot(resnet_epochs, resnet_history['train_loss'], 'r-', label='ResNet-18', linewidth=2)
    axes[1, 0].set_title('Training Loss')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Validation Loss
    axes[1, 1].plot(cnn_epochs, cnn_history['val_loss'], 'b-', label='Custom CNN', linewidth=2)
    axes[1, 1].plot(resnet_epochs, resnet_history['val_loss'], 'r-', label='ResNet-18', linewidth=2)
    axes[1, 1].set_title('Validation Loss')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')


if __name__ == "__main__":
    history = load_history('resnet_history.json')
    plot_training_curves(history, model_name="ResNet-18", save_path="resnet_curves.png")
    cnn_history = load_history('cnn_history.json')
    resnet_history = load_history('resnet_history.json')
    plot_comparison(cnn_history, resnet_history, save_path="comparison.png")