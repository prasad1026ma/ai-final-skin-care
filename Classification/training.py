import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data_cleaning import SkinDataset, get_transforms
from sklearn.model_selection import train_test_split
from cnn import Cnn

def split_test_train(image_paths, labels, test_split, val_split):
    train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
        image_paths, labels, test_size=test_split, stratify=labels, random_state=42
    )

    val_size_adjusted = val_split / (1 - test_split)
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_val_paths, train_val_labels, test_size=val_size_adjusted,
        stratify=train_val_labels, random_state=42
    )

    return (train_paths, train_labels), (val_paths, val_labels), (test_paths, test_labels)


def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
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
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    return running_loss / len(dataloader), 100 * correct / total

def validate(model, dataloader, criterion):
    """ checks to ensure that the data isnt overfitting"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images, labels
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return running_loss / len(dataloader), 100 * correct / total

def training_pipline(image_paths, labels, num_classes, epochs=10, batch_size=32,
                input_size=224):
    (train_paths, train_labels), (val_paths, val_labels), (test_paths, test_labels) = (
        split_test_train(image_paths, labels))
    train_dataset = SkinDataset(train_paths, train_labels, transform=get_transforms(train=True))
    val_dataset = SkinDataset(val_paths, val_labels, transform=get_transforms(train=False))
    test_dataset = SkinDataset(test_paths, test_labels, transform=get_transforms(train=False))

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    model = Cnn(num_classes=num_classes, input_size=input_size)
    # loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    best_val_acc = 0.0
    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = validate(model, val_loader, criterion)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_skin_classifier.pth')
            print(f'Model saved with validation accuracy: {val_acc:.2f}%')

        model.load_state_dict(torch.load('best_skin_classifier.pth'))
        test_loss, test_acc = validate(model, test_loader, criterion)
        print(f'Final Model Test Accuracy: {test_acc}')

        return model






