import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import os
import sys
from tqdm import tqdm
import numpy as np

# Imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.models.dataset import AmharicDataset
from src.models.model import DeepAmharicNet

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = 'models/amharic_cnn.pth'
INDEX_FILE = 'data/processed/dataset_index.csv'
REPORT_DIR = 'reports'

def evaluate():
    if not os.path.exists(REPORT_DIR):
        os.makedirs(REPORT_DIR)

    # 1. Load Data (Test Split)
    print("Loading Test Data...")
    transform = transforms.Compose([transforms.ToTensor()])
    test_dataset = AmharicDataset(INDEX_FILE, '.', split='test', transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # 2. Load Model
    print("Loading Model...")
    model = DeepAmharicNet(num_classes=238).to(DEVICE)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
    else:
        print("Model file not found!")
        return

    model.eval()

    # 3. Inference
    all_preds = []
    all_labels = []
    
    print("Running Inference on Test Set...")
    with torch.no_grad():
        for images, labels in tqdm(test_loader):
            images = images.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    # 4. Metrics
    print("\nCalculating Metrics...")
    
    # Classification Report
    class_names = [test_dataset.idx_to_class[i] for i in range(len(test_dataset.classes))]
    report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(os.path.join(REPORT_DIR, 'classification_report.csv'))
    print(f"Saved classification report to {REPORT_DIR}/classification_report.csv")

    # Confusion Matrix (Plot top 20 confused classes for readability if too large)
    cm = confusion_matrix(all_labels, all_preds)
    
    # Save full CM raw data
    np.savetxt(os.path.join(REPORT_DIR, 'confusion_matrix.csv'), cm, delimiter=",")
    
    print("Evaluation Complete.")
    print(f"Test Accuracy: {report['accuracy']*100:.2f}%")

if __name__ == "__main__":
    evaluate()
