from PIL import Image, ImageDraw, ImageFont
import os

# Create directory
output_dir = "data/external_test"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def create_synthetic_char(filename, char_code):
    """
    Since we might not have an Amharic font installed, 
    we will try to draw a rough approximation or save a blank validation.
    Actually, let's try to grab a random PROCESSED image but modify it 
    significantly (invert, noise) to simulate a "new" source.
    """
    pass

# Strategy Change:
# Since we cannot easily download new images and drawing Amharic without a font is hard,
# We will take an image from the dataset, and apply heavy "external" style transforms:
# 1. Invert colors (White on Black -> Black on White) - The model expects White on Black, 
#    so if we feed it Black on White (typical paper), it should fail unless we preprocess.
# 2. Add heavy noise.
# This simulates "Scanning a paper document".

import shutil
import random
import numpy as np

source_class = "001he"
source_dir = f"data/processed/amharic_chars/{source_class}"
# Get random image
img_name = random.choice(os.listdir(source_dir))
img_path = os.path.join(source_dir, img_name)

print(f"Base Image: {img_path}")

# Load
img = Image.open(img_path)

# 1. Simulate "Paper Scan" (Invert + Brightness)
# The dataset is White text on Black background.
# Real world is Black text on White background.
# predict_model.py usually expects the dataset format.
# Let's save a "Normal" looking image (Black text on White) and see if the App handles it.
from PIL import ImageOps
inverted_img = ImageOps.invert(img) # Now Black text on White background
inverted_img.save(os.path.join(output_dir, "test_paper_scan_he.png"))
print("Created: test_paper_scan_he.png (Inverted/Paper style)")

# 2. Simulate "Noisy Scan"
noise = np.random.randint(0, 50, (32, 32), dtype='uint8')
noisy_img = Image.fromarray(np.array(img) + noise)
noisy_img.save(os.path.join(output_dir, "test_noisy_he.png"))
print("Created: test_noisy_he.png")

print("Synthetic external test data created.")
