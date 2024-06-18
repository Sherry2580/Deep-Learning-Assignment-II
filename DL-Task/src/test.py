import os
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader
from torchvision.models import resnet34
from dataset import ImageDataset
from model import ImprovedCNN
from complex_cnn import ComplexCNN

# Define paths
data_dir = 'data/processed'
txt_file = 'data/test.txt'
checkpoint_dir = 'checkpoints'
best_model_path = os.path.join(checkpoint_dir, 'best_model.pth')
pretrained_model_path = os.path.join(checkpoint_dir, 'pretrained_weight.pth')
results_path = 'results'
os.makedirs(results_path, exist_ok=True)

def test_model(data_dir, txt_file, checkpoint_path, results_path, model_type='ComplexCNN'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize model based on the model_type
    if model_type == 'ImprovedCNN': # original task 1 model
        model = ImprovedCNN(num_classes=50).to(device)
    elif model_type == 'ComplexCNN': # task 2 model
        model = ComplexCNN(num_classes=50).to(device)
    elif model_type == 'ResNet34':
        model = resnet34(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, 50)  # 修改最後的全連接層，適應50個分類
        model = model.to(device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    if os.path.exists(best_model_path):
        print(f"Loading best model from '{best_model_path}'")
        model.load_state_dict(torch.load(best_model_path))
    else:
        print(f"Best model not found. Loading pretrained model from '{pretrained_model_path}'")
        model.load_state_dict(torch.load(pretrained_model_path))
    
    model.eval()

    # Different channel combinations
    channel_combinations = ["RGB", "RG", "GB", "R", "G", "B"]
    results = {'Combination': [], 'Accuracy': []}

    for combo in channel_combinations:
        dataset = ImageDataset(txt_file=txt_file, root_dir=data_dir, channels=combo)
        loader = DataLoader(dataset, batch_size=32, shuffle=False)

        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        accuracy = round(100 * correct / total, 3)
        results['Combination'].append(combo)
        results['Accuracy'].append(accuracy)
        print(f"Accuracy with {combo} channels: {accuracy:.2f}%")

    # Save results to a CSV file
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(results_path, f'test_results_{model_type}.csv'), index=False)

if __name__ == "__main__":
    test_model(data_dir, txt_file, best_model_path, results_path, model_type='ResNet34')
