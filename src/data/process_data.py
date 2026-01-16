import os
import shutil
import random
import pandas as pd
from PIL import Image
from tqdm import tqdm

# Configuration
RAW_DIRS = [
    r"data/raw/Handwritten-Amharic-character-Dataset/extracted/Amharic Character Dataset 1",
    r"data/raw/Handwritten-Amharic-character-Dataset/extracted/Amharic Character Dataset 2"
]
PROCESSED_DIR = r"data/processed/amharic_chars"
INDEX_FILE = r"data/processed/dataset_index.csv"
IMG_SIZE = (32, 32)
SPLIT_RATIOS = {'train': 0.7, 'val': 0.15, 'test': 0.15}

def process_data():
    if os.path.exists(PROCESSED_DIR):
        shutil.rmtree(PROCESSED_DIR)
    os.makedirs(PROCESSED_DIR)

    data_records = []
    
    print("Collecting and processing images...")
    
    # Collect all file paths first to show accurate progress
    all_files = []
    for d in RAW_DIRS:
        if os.path.exists(d):
            for f in os.listdir(d):
                if f.lower().endswith(('.jpg', '.png', '.jpeg')):
                    all_files.append(os.path.join(d, f))
    
    if not all_files:
        print("No files found!")
        return

    for file_path in tqdm(all_files):
        filename = os.path.basename(file_path)
        # Extract class. Filename format assumption: "001he.1.jpg" -> class "001he"
        parts = filename.split('.')
        if len(parts) > 1:
            class_id = parts[0]
        else:
            continue # Skip invalid filenames

        # Create class directory
        class_dir = os.path.join(PROCESSED_DIR, class_id)
        os.makedirs(class_dir, exist_ok=True)
        
        # New filename
        new_filename = f"{class_id}_{len(os.listdir(class_dir))}.png"
        new_path = os.path.join(class_dir, new_filename)
        
        try:
            with Image.open(file_path) as img:
                # Convert to Grayscale
                img = img.convert('L')
                # Resize
                img = img.resize(IMG_SIZE, Image.Resampling.LANCZOS)
                # Save
                img.save(new_path)
                
                data_records.append({
                    'class': class_id,
                    'path': new_path.replace("\\", "/"), # Normalize for CSV
                    'original_filename': filename
                })
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    # Create Splits
    print("Creating splits...")
    df = pd.DataFrame(data_records)
    
    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Stratified split isn't strictly necessary if balanced, but good practice.
    # We'll do simple random split per class to ensure all classes are in all splits.
    
    final_dfs = []
    for class_id, group in df.groupby('class'):
        n = len(group)
        n_train = int(n * SPLIT_RATIOS['train'])
        n_val = int(n * SPLIT_RATIOS['val'])
        
        train_df = group.iloc[:n_train].copy()
        val_df = group.iloc[n_train:n_train+n_val].copy()
        test_df = group.iloc[n_train+n_val:].copy()
        
        train_df['split'] = 'train'
        val_df['split'] = 'val'
        test_df['split'] = 'test'
        
        final_dfs.extend([train_df, val_df, test_df])
        
    final_df = pd.concat(final_dfs)
    
    # Save CSV
    final_df.to_csv(INDEX_FILE, index=False)
    print(f"Processing complete. Saved {len(final_df)} records to {INDEX_FILE}")
    print(f"Train: {len(final_df[final_df['split']=='train'])}")
    print(f"Val: {len(final_df[final_df['split']=='val'])}")
    print(f"Test: {len(final_df[final_df['split']=='test'])}")

if __name__ == "__main__":
    process_data()
