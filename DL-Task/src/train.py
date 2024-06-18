import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.models import resnet34
from dataset import ImageDataset
from model import ImprovedCNN
from complex_cnn import ComplexCNN
import matplotlib.pyplot as plt

# Define paths
data_dir = 'data/processed'
checkpoint_dir = 'checkpoints'
results_dir = 'results'
plots_dir = os.path.join(results_dir, 'plots')
os.makedirs(checkpoint_dir, exist_ok=True)
os.makedirs(plots_dir, exist_ok=True)

def train_model(data_dir, txt_files, batch_size=32, num_epochs=25, learning_rate=0.001, model_type='ComplexCNN'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Start Training...")

    # Load datasets with data augmentation
    train_dataset = ImageDataset(txt_file=txt_files['train'], root_dir=data_dir)
    val_dataset = ImageDataset(txt_file=txt_files['val'], root_dir=data_dir)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model based on the model_type
    if model_type == 'ImprovedCNN':     # task 1 model
        model = ImprovedCNN(num_classes=50).to(device)
    elif model_type == 'ComplexCNN':    # task 2 model
        model = ComplexCNN(num_classes=50).to(device)
    elif model_type == 'ResNet34':
        model = resnet34(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, 50)  # 修改最後的全連接層，適應50個分類
        model = model.to(device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_val_loss = float('inf')
    train_losses, val_losses, train_accuracies, val_accuracies = [], [], [], []

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        total_loss = 0.0
        correct = 0
        total = 0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            if i % 100 == 99:    # Print every 100 mini-batches
                print(f"[{epoch + 1}, {i + 1}] loss: {running_loss / 100:.3f}")
                running_loss = 0.0

        train_loss = total_loss / len(train_loader)
        train_accuracy = 100 * correct / total
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        # Validate the model
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_loss /= len(val_loader)
        val_accuracy = 100 * correct / total
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        print(f"Validation Loss after epoch {epoch + 1}: {val_loss:.3f}")
        print(f"Validation Accuracy after epoch {epoch + 1}: {val_accuracy:.2f}%")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, f"best_model_{model_type}.pth"))

    print("Finished Training")

    # Plot loss and accuracy
    epochs = range(1, num_epochs + 1)
    plt.figure()
    plt.plot(epochs, train_losses, 'b', label='Training')
    plt.plot(epochs, val_losses, 'r', label='Validation')
    plt.title('Train and Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(plots_dir, f'loss_plot_{model_type}.png'))

    plt.figure()
    plt.plot(epochs, train_accuracies, 'b', label='Training')
    plt.plot(epochs, val_accuracies, 'r', label='Validation')
    plt.title('Train and Val Acc')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(os.path.join(plots_dir, f'accuracy_plot_{model_type}.png'))

if __name__ == "__main__":
    txt_files = {
        'train': 'data/train.txt',
        'val': 'data/val.txt'
    }
    # Argument to select model type
    train_model(data_dir, txt_files, model_type='ResNet34')
