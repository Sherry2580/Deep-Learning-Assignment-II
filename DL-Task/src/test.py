import os
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader
from torchvision.models import resnet34
from dataset import ImageDataset
from model import ImprovedCNN
from complex_cnn import ComplexCNN
import argparse

# Define paths
data_dir = 'data/processed'
txt_file = 'data/test.txt'
checkpoint_dir = 'checkpoints'
results_path = 'results'

def test_model(data_dir, txt_file, results_path, model_type='ComplexCNN'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize model based on the model_type
    if model_type == 'ImprovedCNN':
        model = ImprovedCNN(num_classes=50).to(device)
    elif model_type == 'ComplexCNN':
        model = ComplexCNN(num_classes=50).to(device)
    elif model_type == 'ResNet34':
        model = resnet34(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, 50)  # 修改最後的全連接層，適應50個分類
        model = model.to(device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model_checkpoint_path = os.path.join(checkpoint_dir, model_type, 'best_model.pth')
    pretrained_weight_path = os.path.join(checkpoint_dir, f'pretrained_weight_{model_type}.pth')

    # if the best model that you train exists, load it; otherwise, load the pretrained weights we provided
    if os.path.exists(model_checkpoint_path):
        print(f"Loading best model from '{model_checkpoint_path}'")
        model.load_state_dict(torch.load(model_checkpoint_path))
    elif os.path.exists(pretrained_weight_path):
        print(f"Loading pretrained weights from '{pretrained_weight_path}'")
        model.load_state_dict(torch.load(pretrained_weight_path))
    else:
        raise FileNotFoundError(f"Model checkpoint '{model_checkpoint_path}' or pretrained weights '{pretrained_weight_path}' not found.")

    model.eval()

    # Different channel combinations
    channel_combinations = ["RGB", "RG", "GB", "R", "G", "B"]
    model_results_path = os.path.join(results_path, model_type)
    os.makedirs(model_results_path, exist_ok=True)

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
    df.to_csv(os.path.join(model_results_path, f'test_results_{model_type}.csv'), index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test a model.')
    parser.add_argument('--model_type', type=str, required=True, help='The model type to test (ImprovedCNN, ComplexCNN, ResNet34).')
    args = parser.parse_args()

    test_model(data_dir, txt_file, results_path, model_type=args.model_type)

