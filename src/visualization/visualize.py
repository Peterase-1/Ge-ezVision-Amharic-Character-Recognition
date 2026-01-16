import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import sys
import os
import random

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.models.dataset import AmharicDataset
from src.models.model import DeepAmharicNet

def plot_confusion_matrix(csv_path='reports/confusion_matrix.csv', save_path='reports/figures/confusion_matrix.png'):
    if not os.path.exists(csv_path):
        print(f"Confusion matrix CSV not found at {csv_path}. Run evaluate_model.py first.")
        return

    print("Generating Confusion Matrix Plot...")
    cm = np.genfromtxt(csv_path, delimiter=',')
    
    # Since 238 classes is huge, let's plot just the top-k most confused classes or a downsampled version?
    # Or just the full matrix but make it large.
    
    plt.figure(figsize=(20, 16))
    sns.heatmap(cm, cmap='Blues', xticklabels=False, yticklabels=False)
    plt.title('Confusion Matrix (238 Classes)', fontsize=20)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"Saved {save_path}")

def plot_sample_predictions(model_path='models/amharic_cnn.pth', save_path='reports/figures/sample_predictions.png'):
    print("Generating Sample Predictions Grid...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load Model
    model = DeepAmharicNet(num_classes=238).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    
    # Load Dataset
    dataset = AmharicDataset(csv_file='data/processed/dataset_index.csv', root_dir='.', split='test', transform=transforms.ToTensor())
    
    # Pick 16 random indices
    indices = random.sample(range(len(dataset)), 16)
    
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    
    with torch.no_grad():
        for i, idx in enumerate(indices):
            image, label_idx = dataset[idx]
            
            # Prediction
            output = model(image.unsqueeze(0).to(device))
            _, pred_idx = torch.max(output, 1)
            pred_idx = pred_idx.item()
            
            true_label = dataset.idx_to_class[label_idx]
            pred_label = dataset.idx_to_class[pred_idx]
            
            # Plot
            ax = axes[i // 4, i % 4]
            ax.imshow(image.squeeze().numpy(), cmap='gray')
            
            color = 'green' if true_label == pred_label else 'red'
            ax.set_title(f"T: {true_label}\nP: {pred_label}", color=color, fontweight='bold')
            ax.axis('off')
            
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"Saved {save_path}")

def plot_training_progression(save_path='reports/figures/training_curve.png'):
    print("Generating Training Progression Curve...")
    # Reconstructed data points from the documentation
    epochs = [1, 5, 7, 20]
    accuracy = [5.75, 78.2, 87.3, 88.32]
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, accuracy, marker='o', linestyle='-', color='b', linewidth=2, markersize=8)
    
    plt.title('Training Progression: Validation Accuracy', fontsize=16)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(range(0, 21, 2))
    plt.ylim(0, 100)
    
    # Annotate points
    for x, y in zip(epochs, accuracy):
        plt.annotate(f"{y}%", (x, y), textcoords="offset points", xytext=(0,10), ha='center')
        
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"Saved {save_path}")

if __name__ == "__main__":
    # Ensure raw output dir exists
    if not os.path.exists('reports'):
        os.makedirs('reports')
        
    # We need the confusion matrix csv first. 
    # If it doesn't exist, we assume the user ran evaluate_model.py. 
    # If not, we skip or could run it. For now, let's assume it exists or skip.
    if os.path.exists('reports/confusion_matrix.csv'):
        plot_confusion_matrix()
    else:
        print("reports/confusion_matrix.csv not found. Skipping CM plot.")
        
    plot_sample_predictions()
    plot_training_progression()
