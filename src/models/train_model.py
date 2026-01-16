import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import os
from tqdm import tqdm

# Import our custom modules
from dataset import AmharicDataset
from model import SimpleCNN

# --- Configuration ---
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 5 # Number of times to show the whole dataset to the AI
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

def train():
    # 1. Prepare Data
    print("Loading Data...")
    
    # Transformations: Convert to Tensor and Normalize (scale usually between 0-1 or -1 to 1)
    transform = transforms.Compose([
        transforms.ToTensor(), # Converts 0-255 image to 0.0-1.0 tensor
    ])
    
    # Load Train and Validation sets
    train_dataset = AmharicDataset(
        csv_file='data/processed/dataset_index.csv',
        root_dir='.',
        split='train',
        transform=transform
    )
    
    val_dataset = AmharicDataset(
        csv_file='data/processed/dataset_index.csv',
        root_dir='.',
        split='val',
        transform=transform
    )
    
    # DataLoaders shuffle and batch the data
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # 2. Initialize Model
    model = SimpleCNN(num_classes=238).to(DEVICE)
    
    # 3. Define Loss Function and Optimizer
    criterion = nn.CrossEntropyLoss() # Standard for classification
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE) # 'Smart' gradient descent
    
    # 4. Training Loop
    best_acc = 0.0
    
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        print("-" * 10)
        
        # --- TRAIN PHASE ---
        model.train() # Set to training mode
        running_loss = 0.0
        correct = 0
        total = 0
        
        progress_bar = tqdm(train_loader, desc="Training")
        
        for images, labels in progress_bar:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass (Learning)
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            progress_bar.set_postfix(loss=loss.item())
            
        train_acc = 100 * correct / total
        avg_train_loss = running_loss / len(train_loader)
        print(f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        
        # --- VALIDATION PHASE ---
        model.eval() # Set to evaluation mode (no dropout)
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad(): # Don't calculate gradients (saves memory)
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
        
        # Save Best Model
        if val_acc > best_acc:
            best_acc = val_acc
            save_path = 'models/amharic_cnn.pth'
            if not os.path.exists('models'):
                os.makedirs('models')
            torch.save(model.state_dict(), save_path)
            print(f"Create New Record! Model saved to {save_path}")

    print("\nTraining Complete!")
    print(f"Best Accuracy: {best_acc:.2f}%")

if __name__ == "__main__":
    train()
