import torch
import torchvision.transforms as transforms
from PIL import Image
import sys
import os
import argparse

# Add current directory to path so we can import model
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.models.model import DeepAmharicNet

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = 'models/amharic_cnn.pth'
INDEX_FILE = 'data/processed/dataset_index.csv'

def load_model(model_path, num_classes=238):
    model = DeepAmharicNet(num_classes=num_classes)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
        model.to(DEVICE)
        model.eval()
        return model
    else:
        print(f"Error: Model not found at {model_path}")
        return None

def predict_image(image_path, model, class_map=None):
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return

    # Preprocessing
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(), # Standardizes to 0-1
    ])

    try:
        # Load and transform
        image = Image.open(image_path).convert('L')
        input_tensor = transform(image).unsqueeze(0) # Add batch dimension (1, 1, 32, 32)
        input_tensor = input_tensor.to(DEVICE)

        # Predict
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            
            # Get Top 3 Predictions
            top3_prob, top3_idx = torch.topk(probabilities, 3)

            print(f"\nPrediction for '{image_path}':")
            print("-" * 30)
            for i in range(3):
                class_idx = top3_idx[i].item()
                prob = top3_prob[i].item()
                
                class_name = str(class_idx)
                if class_map:
                    class_name = class_map.get(class_idx, str(class_idx))
                
                print(f"Rank {i+1}: Class '{class_name}' ({prob*100:.2f}%)")

    except Exception as e:
        print(f"Error during prediction: {e}")

def get_class_map_from_csv(csv_path):
    # Reconstruct the class mapping used during training
    import pandas as pd
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        classes = sorted(df['class'].unique())
        # The dataset class maps sorted unique classes to indices 0..N
        return {i: cls for i, cls in enumerate(classes)}
    return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict Amharic Character from Image")
    parser.add_argument("image_path", help="Path to the image file")
    args = parser.parse_args()

    # Load resources
    model = load_model(MODEL_PATH)
    class_map = get_class_map_from_csv(INDEX_FILE)

    if model:
        predict_image(args.image_path, model, class_map)
