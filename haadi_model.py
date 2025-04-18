import os
import random
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt

# Configuration
DATA_DIR = Path('project_data')
SPLIT_DIR = Path('splits')
MODEL_DIR = Path('models')
BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 1e-3
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

# Ensure directories exist
SPLIT_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

def get_datasets():
    """
    Load full dataset and split into train/val/test. Save/load split indices to avoid re-splitting on every run.
    """
    # One-time global transforms
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    full_dataset = datasets.ImageFolder(root=DATA_DIR, transform=transform)
    num_samples = len(full_dataset)

    # Load or create split indices
    indices_file = SPLIT_DIR / 'indices.pt'
    if indices_file.exists():
        splits = torch.load(indices_file)
        train_idx, val_idx, test_idx = splits['train'], splits['val'], splits['test']
        print(f"Loaded existing splits: {len(train_idx)} train, {len(val_idx)} val, {len(test_idx)} test")
    else:
        indices = list(range(num_samples))
        random.shuffle(indices)
        train_end = int(TRAIN_RATIO * num_samples)
        val_end = train_end + int(VAL_RATIO * num_samples)
        train_idx = indices[:train_end]
        val_idx = indices[train_end:val_end]
        test_idx = indices[val_end:]
        torch.save({'train': train_idx, 'val': val_idx, 'test': test_idx}, indices_file)
        print(f"Created new splits: {len(train_idx)} train, {len(val_idx)} val, {len(test_idx)} test")

    train_set = Subset(full_dataset, train_idx)
    val_set = Subset(full_dataset, val_idx)
    test_set = Subset(full_dataset, test_idx)

    return train_set, val_set, test_set, full_dataset.classes


def get_loaders(train_set, val_set, test_set):
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    return train_loader, val_loader, test_loader


def build_model(num_classes):
    """
    Use a pretrained ResNet18, replace the final layer.
    """
    model = models.resnet18(pretrained=False)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    return model


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, running_corrects = 0.0, 0
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        preds = torch.argmax(outputs, dim=1)
        running_corrects += torch.sum(preds == labels.data)
    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = running_corrects.double() / len(loader.dataset)
    return epoch_loss, epoch_acc.item()


def validate(model, loader, criterion, device):
    model.eval()
    running_loss, running_corrects = 0.0, 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            preds = torch.argmax(outputs, dim=1)
            running_corrects += torch.sum(preds == labels.data)
    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = running_corrects.double() / len(loader.dataset)
    return epoch_loss, epoch_acc.item()


def plot_predictions(model, loader, class_names, device, num_images=6):
    """
    Show a few test images with true vs predicted labels.
    """
    model.eval()
    images_shown = 0
    plt.figure(figsize=(12, 8))

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            for i in range(inputs.size(0)):
                if images_shown >= num_images:
                    break
                ax = plt.subplot(2, 3, images_shown + 1)
                img = inputs[i].cpu().permute(1, 2, 0)
                img = img * torch.tensor([0.229, 0.224, 0.225]) + torch.tensor([0.485, 0.456, 0.406])
                img = img.clamp(0, 1)
                ax.imshow(img.numpy())
                ax.set_title(f"True: {class_names[labels[i]]}\nPred: {class_names[preds[i]]}")
                ax.axis('off')
                images_shown += 1
            if images_shown >= num_images:
                break
    plt.tight_layout()
    plt.show()


def main():
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load data
    train_set, val_set, test_set, class_names = get_datasets()
    train_loader, val_loader, test_loader = get_loaders(train_set, val_set, test_set)
    print(f"Classes: {class_names}")

    # Build model
    model = build_model(num_classes=len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_acc = 0.0
    for epoch in range(NUM_EPOCHS):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}: ")
        print(f"  Train Loss: {train_loss:.4f}  Acc: {train_acc:.4f}")
        print(f"  Val   Loss: {val_loss:.4f}  Acc: {val_acc:.4f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_path = MODEL_DIR / 'best_model.pt'
            torch.save(model.state_dict(), best_path)
            print(f"Saved new best model to {best_path}")

    # Final test evaluation
    test_loss, test_acc = validate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}  Acc: {test_acc:.4f}")

    # Plot predictions on test set
    plot_predictions(model, test_loader, class_names, device)

    # Save final model
    final_path = MODEL_DIR / 'final_model.pt'
    torch.save(model.state_dict(), final_path)
    print(f"Saved final model to {final_path}")

    # Save class mapping
    torch.save(class_names, MODEL_DIR / 'classes.pt')
    print(f"Saved class mapping to {MODEL_DIR / 'classes.pt'}")

if __name__ == '__main__':
    main()
