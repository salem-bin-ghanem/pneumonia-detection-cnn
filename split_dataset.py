"""
Dataset Re-organizer for Pneumonia X-Rays

This script takes the unbalanced Kaggle dataset and redistributes the images
into a clean 70% Train, 15% Validation, and 15% Test split.

It leaves any existing files directly inside "data/" (like data/README.md)
untouched, only clearing out and rebuilding the training/validation/test folders.

Instructions:
1. Extract the downloaded Kaggle dataset into a folder named "kaggle_original".
2. Ensure it looks like this:
   kaggle_original/
       train/
       val/
       test/
3. Run this script. It will copy the images into a new, properly split "data/" folder.
"""

import os
import shutil
import random

# Configuration
SOURCE_DIR = "kaggle_original" # Put the extracted Kaggle folders inside here
DEST_DIR = "data"              # The folder where the split data will go

# Split percentages (Must add up to 1.0)
TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.15
# Test split is the remaining 0.15
random.seed(42)

def create_directory_structure():
    """
    Creates the empty train, val, and test folders with NORMAL/PNEUMONIA subfolders.
    If the subfolders already exist, deletes them to start fresh, but keeps 
    other files in DEST_DIR (like data/README.md) intact.
    """
    # Create the main destination folder if it doesn't exist yet
    os.makedirs(DEST_DIR, exist_ok=True)
        
    for split in ['train', 'val', 'test']:
        split_path = os.path.join(DEST_DIR, split)
        
        # If the split folder (train/val/test) exists, delete it to ensure a clean slate
        if os.path.exists(split_path):
            print(f"Clearing existing split folder: {split_path}")
            shutil.rmtree(split_path)
            
        # Recreate the split folder and its subdirectories
        for category in ['NORMAL', 'PNEUMONIA']:
            os.makedirs(os.path.join(split_path, category), exist_ok=True)
            
def gather_all_images(category):
    """Finds all images for a specific category across the original train/val/test folders."""
    all_images = []
    for split in ['train', 'val', 'test']:
        folder_path = os.path.join(SOURCE_DIR, split, category)
        if os.path.exists(folder_path):
            for filename in os.listdir(folder_path):
                if filename.endswith((".jpeg", ".jpg", ".png")):
                    all_images.append(os.path.join(folder_path, filename))
    return all_images

def copy_images_to_dest(image_paths, split_name, category):
    """Copies a list of images to their new destination folder."""
    dest_folder = os.path.join(DEST_DIR, split_name, category)
    for src_path in image_paths:
        filename = os.path.basename(src_path)
        dest_path = os.path.join(dest_folder, filename)
        shutil.copy2(src_path, dest_path)

def split_and_copy_data(category):
    """Shuffles the data and splits it into 70/15/15."""
    print(f"\nProcessing {category} images...")
    
    # 1. Gather all images and shuffle them randomly
    images = gather_all_images(category)

    random.shuffle(images)
    
    total_images = len(images)
    print(f"Found {total_images} total {category} images.")
    
    # 2. Calculate the exact number of files for each split
    train_count = int(total_images * TRAIN_SPLIT)
    val_count = int(total_images * VAL_SPLIT)
    
    # 3. Slice the list into three chunks
    train_images = images[:train_count]
    val_images = images[train_count : train_count + val_count]
    test_images = images[train_count + val_count:]
    
    # 4. Copy the files
    print(f"Copying {len(train_images)} to Train, {len(val_images)} to Val, {len(test_images)} to Test...")
    copy_images_to_dest(train_images, 'train', category)
    copy_images_to_dest(val_images, 'val', category)
    copy_images_to_dest(test_images, 'test', category)

def main():
    if not os.path.exists(SOURCE_DIR):
        print(f"Error: Could not find the folder '{SOURCE_DIR}'.")
        print("Please extract the downloaded Kaggle dataset into a folder named 'kaggle_original'.")
        return

    print("Rebuilding split directories inside 'data/' (preserving existing files like README)...")
    create_directory_structure()
    
    split_and_copy_data('NORMAL')
    split_and_copy_data('PNEUMONIA')
    
    print("\nDataset successfully split! The 'data' folder is ready for main.py.")

if __name__ == "__main__":
    main()
