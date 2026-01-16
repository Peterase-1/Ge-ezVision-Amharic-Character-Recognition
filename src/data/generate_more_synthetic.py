from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import random
import os

def generate_variations(source_path, output_dir, label):
    """
    Generates variations of a source image to simulate external data.
    """
    try:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        img = Image.open(source_path).convert('L') # Ensure grayscale
        base_name = os.path.basename(source_path).split('.')[0]
        
        # 1. Rotation (Should pass)
        # Rotate by -15 to +15 degrees (a bit more than training which is usually 10)
        angle = random.uniform(-15, 15)
        img_rotated = img.rotate(angle, fillcolor=255) # White background
        save_path_rot = os.path.join(output_dir, f"test_rotated_{label}.png")
        img_rotated.save(save_path_rot)
        print(f"Generated Rotated: {save_path_rot}")
        
        # 2. Blur (Simulate out of focus scan)
        img_blurred = img.filter(ImageFilter.GaussianBlur(1.5))
        save_path_blur = os.path.join(output_dir, f"test_blurred_{label}.png")
        img_blurred.save(save_path_blur)
        print(f"Generated Blurred: {save_path_blur}")

        # 3. Brightness/Contrast Change (Simulate overexposed/underexposed scan)
        enhancer = ImageEnhance.Brightness(img)
        img_bright = enhancer.enhance(1.5) # 50% brighter
        # Then boost contrast
        enhancer_c = ImageEnhance.Contrast(img_bright)
        img_contrast = enhancer_c.enhance(2.0)
        save_path_bright = os.path.join(output_dir, f"test_bright_{label}.png")
        img_contrast.save(save_path_bright)
        print(f"Generated Bright/Contrast: {save_path_bright}")
        
        # 4. Inverted (We know this fails, but generating a fresh one for completeness if needed)
        # img_inverted = ImageOps.invert(img)
        # save_path_inv = os.path.join(output_dir, f"test_inverted_{label}.png")
        # img_inverted.save(save_path_inv)

    except Exception as e:
        print(f"Error generating images: {e}")

if __name__ == "__main__":
    # Source image: HU (Character 002hu)
    # Using 002hu_10.png as base
    source_image = "data/processed/amharic_chars/002hu/002hu_10.png"
    output_directory = "data/external_test"
    
    print(f"Generating synthetic external data from {source_image}...")
    generate_variations(source_image, output_directory, "hu")
    print("Done.")
