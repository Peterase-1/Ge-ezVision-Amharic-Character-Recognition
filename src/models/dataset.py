import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import os

class AmharicDataset(Dataset):
    """
    A custom Dataset to load Amharic characters.
    
    How it works:
    1. We give it the 'dataset_index.csv' file which tells us where every image is.
    2. When the model asks for item 'i', we:
       - Find the path in the CSV.
       - Load the image using Pillow.
       - Convert it to a Tensor (numbers for the AI).
       - Return the image and its label (class ID).
    """
    def __init__(self, csv_file, root_dir, split, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            split (string): One of 'train', 'val', or 'test'.
            transform (callable, optional): Optional transform to be applied (like resizing).
        """
        self.df = pd.read_csv(csv_file)
        # Filter: keep only the rows for our split (e.g., only 'train' rows)
        self.df = self.df[self.df['split'] == split].reset_index(drop=True)
        
        self.root_dir = root_dir
        self.transform = transform
        
        # Create a mapping from Class Name (e.g., '001he') to a Number (0, 1, 2...)
        # Neural networks operate on numbers, not strings.
        self.classes = sorted(self.df['class'].unique())
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.idx_to_class = {i: cls_name for i, cls_name in enumerate(self.classes)}

    def __len__(self):
        # The total number of samples in this split
        return len(self.df)

    def __getitem__(self, idx):
        # 1. Get the row from the dataframe
        row = self.df.iloc[idx]
        
        # 2. Construct the full image path
        # Our CSV has 'data/processed/...', but we might run from root, so we handle paths carefully.
        img_path = row['path']
        if not os.path.exists(img_path):
             # Fallback if path is relative
             img_path = os.path.join(self.root_dir, row['path'])
        
        # 3. Load Image
        image = Image.open(img_path).convert('L') # 'L' means Grayscale (1 channel)
        
        # 4. Get Label (Number)
        label_str = row['class']
        label_idx = self.class_to_idx[label_str]
        
        # 5. Apply Transformations (Make it a Tensor)
        if self.transform:
            image = self.transform(image)
        
        return image, label_idx
