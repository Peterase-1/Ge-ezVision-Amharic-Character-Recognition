import torch
import torchvision.transforms as transforms
from PIL import Image
import sys
import os
import pandas as pd
import re

# Add src to path to import modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from models.model import DeepAmharicNet

# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = 'models/amharic_cnn.pth'
INDEX_FILE = 'data/processed/dataset_index.csv'

def clean_label(label):
    """
    Converts '001he' -> 'HE'
    Converts '002hu' -> 'HU'
    """
    # Remove digits and common delimiters
    text = re.sub(r'[\d_]', '', label)
    return text.upper()

def load_resources():
    print("Loading resources... please wait.")
    
    # 1. Load Class Map
    if not os.path.exists(INDEX_FILE):
        print(f"Error: Index file not found at {INDEX_FILE}")
        return None, None
    
    df = pd.read_csv(INDEX_FILE)
    classes = sorted(df['class'].unique())
    class_map = {i: cls for i, cls in enumerate(classes)}
    
    # 2. Load Model
    model = DeepAmharicNet(num_classes=238)
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at {MODEL_PATH}")
        return None, None
        
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
    model.to(DEVICE)
    model.eval()
    
    print("System Ready!")
    return model, class_map

def predict(model, class_map, image_path):
    # Preprocessing
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ])
    
    try:
        image = Image.open(image_path).convert('L')
        input_tensor = transform(image).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            
            # Top 1
            prob, idx = torch.topk(probabilities, 1)
            class_code = class_map[idx.item()]
            readable_name = clean_label(class_code)
            confidence = prob.item() * 100
            
            return readable_name, confidence
            
    except Exception as e:
        print(f"Error: {e}")
        return None, 0

def main_loop():
    print("\n" + "="*50)
    print("  Ge-ezVision: Interactive Console v1.0")
    print("  Type 'exit' or 'q' to quit.")
    print("="*50 + "\n")
    
    model, class_map = load_resources()
    if not model:
        return

    while True:
        user_input = input("\n[Ge-ezVision] Enter Image Path >> ").strip()
        
        if user_input.lower() in ['exit', 'q', 'quit']:
            print("Goodbye!")
            break
            
        # Remove quotes if user copied as path
        user_input = user_input.replace('"', '').replace("'", "")
        
        if not os.path.exists(user_input):
            print("  [!] File does not exist. Try again.")
            continue
            
        name, conf = predict(model, class_map, user_input)
        
        if name:
            print(f"  Result: {name} ({conf:.1f}%)")

if __name__ == "__main__":
    main_loop()
