import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from skin_dataset import SkinDataset
from data_cleaning import  process_scin_dataset,load_dataset
from sklearn.model_selection import train_test_split
from cnn import Cnn
from res_net import ResNetClassifier
from constants import LABELS_CSV_PATH, IMAGES_BASE_DIR, CASES_CSV_PATH, OUTPUT_CSV, NUM_CLASSES

def split_train_test(image_paths, labels, test_split=0.2):
    train_paths, test_paths, train_labels, test_labels = train_test_split(
        image_paths,
        labels,
        test_size=test_split,
        stratify=labels,
        random_state=42
    )
    return (train_paths, train_labels), (test_paths, test_labels)

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return running_loss / len(dataloader), 100 * correct / total

def test_model(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return running_loss / len(dataloader), 100 * correct / total
def training_pipeline(image_paths, labels, num_classes, epochs=10, batch_size=64, input_size=224):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    (train_paths, train_labels), (test_paths, test_labels) = split_train_test(image_paths, labels)

    # Create datasets
    train_dataset = SkinDataset(train_paths, train_labels, transform=SkinDataset.get_transforms(train=True))
    test_dataset = SkinDataset(test_paths,  test_labels,  transform=SkinDataset.get_transforms(train=False))

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=4)

    # CNN model
    model = Cnn(num_classes=num_classes, input_size=224)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2, verbose=True
    )

    print(f"Beginning Training Process")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Number of classes: {num_classes}\n")

    # Train for N epochs
    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        scheduler.step(train_loss)


    # Final test evaluation
    test_loss, test_acc = test_model(model, test_loader, criterion, device)
    torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, 'best_model.pth')
    print(f"\nFINAL TEST ACCURACY: {test_acc:.2f}%")
    return model


if __name__ == "__main__":
    dataset_path = process_scin_dataset(
        cases_csv_path=CASES_CSV_PATH,
        labels_csv_path=LABELS_CSV_PATH,
        images_base_dir=IMAGES_BASE_DIR,
        output_csv=OUTPUT_CSV
    )
    image_paths, labels = load_dataset(OUTPUT_CSV)
    dataset = SkinDataset(
        image_paths,
        labels,
        transform=SkinDataset.get_transforms(train=True)
    )

    model = training_pipeline(image_paths, labels, num_classes=10)







