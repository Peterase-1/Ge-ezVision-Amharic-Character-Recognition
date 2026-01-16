import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import os
from tqdm import tqdm

# Import our custom modules
from dataset import AmharicDataset
from model import DeepAmharicNet  # UPDATED: Import DeepAmharicNet

# --- Configuration ---
BATCH_SIZE = 64 # Increased batch size for stability
LEARNING_RATE = 0.001
EPOCHS = 20 # Increased epochs
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

def train():
    # 1. Prepare Data
    print("Loading Data with Augmentation...")
    
    # Train Transform: Add Noise/Rotation to generalize better
    train_transform = transforms.Compose([
        transforms.RandomRotation(10), # Rotate +/- 10 degrees
        transforms.RandomAffine(0, translate=(0.1, 0.1)), # Shift image slightly
        transforms.ToTensor(), 
    ])
    
    # Val Transform: Just standard conversion
    val_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    train_dataset = AmharicDataset(
        csv_file='data/processed/dataset_index.csv',
        root_dir='.',
        split='train',
        transform=train_transform # User augmented transform
    )
    
    val_dataset = AmharicDataset(
        csv_file='data/processed/dataset_index.csv',
        root_dir='.',
        split='val',
        transform=val_transform
    )
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # 2. Initialize Model
    model = DeepAmharicNet(num_classes=238).to(DEVICE) # UPDATED: Use DeepAmharicNet
    
    # 3. Optimizer & Scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Reduce LR if validation loss doesn't improve for 2 epochs
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    
    # 4. Training Loop
    best_acc = 0.0
    
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        print("-" * 10)
        
        # --- TRAIN PHASE ---
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        progress_bar = tqdm(train_loader, desc="Training")
        
        for images, labels in progress_bar:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            progress_bar.set_postfix(loss=loss.item())
            
        train_acc = 100 * correct / total
        avg_train_loss = running_loss / len(train_loader)
        print(f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        
        # --- VALIDATION PHASE ---
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_acc = 100 * val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)
        print(f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        
        # Step Scheduler
        scheduler.step(avg_val_loss)
        
        # Save Best Model
        if val_acc > best_acc:
            best_acc = val_acc
            save_path = 'models/amharic_cnn.pth' # Overwrite previous model
            if not os.path.exists('models'):
                os.makedirs('models')
            torch.save(model.state_dict(), save_path)
            print(f"Create New Record! Model saved to {save_path}")

    print("\nTraining Complete!")
    print(f"Best Accuracy: {best_acc:.2f}%")

if __name__ == "__main__":
    train()
