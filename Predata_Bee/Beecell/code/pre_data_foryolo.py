
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on April 20, 2025
Updated to use all detected cells and create data.yaml with relative paths for YOLO
@author: avsthiago
"""

import os
import cv2
import numpy as np
import shutil
from tqdm import tqdm

# Configuration
PATH_IMGS = '../original_images/'
ANNOTATIONS_PATH = '../annotations/predictions_corrected/'
OUT_DATASET = 'dataset/'
PATH_TRAIN = os.path.join(OUT_DATASET, 'train')
PATH_VAL = os.path.join(OUT_DATASET, 'validation')
CELL_SIZE = 64  # Output cell image size (pixels)

# Class labels (same as previous code)
LABELS = {
    "Capped_brood": 0,
    "Unlabeled": 1,
    "Honey": 2,
    "Brood": 3,
    "Capped_honey": 4,
    "Other": 5,
    "Pollen": 6
}

def create_folder(path):
    """Create folder if it doesn't exist"""
    if not os.path.exists(path):
        os.makedirs(path)

def create_dataset_structure():
    """Create all necessary folders for the dataset"""
    # Remove existing dataset folder if it exists
    shutil.rmtree(os.path.abspath(OUT_DATASET), ignore_errors=True)
    
    # Create main dataset folders
    create_folder(PATH_TRAIN)
    create_folder(PATH_VAL)
    
    # Create folders for each class in both train and validation
    for class_name in LABELS.keys():
        create_folder(os.path.join(PATH_TRAIN, class_name))
        create_folder(os.path.join(PATH_VAL, class_name))
    
    print("Created folders for all classes:")
    for class_name in LABELS.keys():
        print(f"- {class_name}")

def create_yaml_config():
    """Create data.yaml file for YOLO with relative paths"""
    yaml_content = """train: /train
val: /validation
nc: 7
names: ['Capped_brood', 'Unlabeled', 'Honey', 'Brood', 'Capped_honey', 'Other', 'Pollen']
"""
    yaml_path = os.path.join(OUT_DATASET, 'data.yaml')
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    print(f"Created YOLO config file: {yaml_path}")

def load_annotations(annotations_path):
    """Load annotations from .txt files in the annotations path"""
    annotations = []
    for txt_file in os.listdir(annotations_path):
        if not txt_file.endswith('.txt'):
            continue
        image_name = os.path.splitext(txt_file)[0]
        txt_path = os.path.join(annotations_path, txt_file)
        try:
            with open(txt_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    x, y, r, class_id = map(int, line.strip().split(','))
                    annotations.append([image_name, x, y, r, class_id])
        except Exception as e:
            print(f"Error reading {txt_path}: {str(e)}")
    return annotations

def extract_cell(image, x, y, r):
    """Extract a cell from the image centered at (x, y) with radius r"""
    try:
        # Calculate crop boundaries
        crop_size = int(2 * r)
        x_min = max(0, x - r)
        y_min = max(0, y - r)
        x_max = min(image.shape[1], x + r)
        y_max = min(image.shape[0], y + r)
        
        # Ensure valid crop size
        if x_max <= x_min or y_max <= y_min:
            return None
            
        # Crop the cell
        cell = image[y_min:y_max, x_min:x_max]
        
        # Resize to fixed size
        cell = cv2.resize(cell, (CELL_SIZE, CELL_SIZE), interpolation=cv2.INTER_AREA)
        
        # Check if cell is valid (not empty)
        if cell.shape[0] == 0 or cell.shape[1] == 0 or np.mean(cell) == 0:
            return None
            
        return cell
    except Exception as e:
        print(f"Error extracting cell at ({x}, {y}, {r}): {str(e)}")
        return None

def save_image(data):
    """Save image to specified path"""
    cell, save_path = data
    try:
        cv2.imwrite(save_path, cell)
    except Exception as e:
        print(f"Error saving image to {save_path}: {str(e)}")

def create_dataset():
    """Create the dataset using all verified annotations"""
    # Create all necessary folders first
    create_dataset_structure()
    
    annotations = load_annotations(ANNOTATIONS_PATH)

    # Group annotations by image
    images_dict = {}
    for ann in annotations:
        image_name = ann[0]
        if image_name not in images_dict:
            images_dict[image_name] = []
        images_dict[image_name].append(ann[1:])

    # Initialize dict_classes using the LABELS values
    dict_classes = {class_id: [] for class_id in LABELS.values()}

    print("\nProcessing annotations...")
    with tqdm(total=len(images_dict)) as t:
        for image_name, annots in images_dict.items():
            image_path = os.path.join(PATH_IMGS, image_name + ".jpg")
            image = cv2.imread(image_path)

            if image is None:
                print(f"Could not read image: {image_path}")
                t.update(1)
                continue

            # Process each annotation for this image
            for x, y, r, class_id in annots:
                cell = extract_cell(image, x, y, r)
                if cell is not None:
                    dict_classes[class_id].append((cell, (x, y, r)))

            t.update(1)

    print("\nSaving images to dataset...")
    # Create a reverse mapping from class_id to class_name
    id_to_name = {v: k for k, v in LABELS.items()}
    
    for class_id, samples in dict_classes.items():
        if not samples:
            print(f"No samples for class {id_to_name[class_id]}")
            continue
            
        cl_name = id_to_name[class_id]

        # Shuffle and split into train/validation (80/20)
        np.random.shuffle(samples)
        split_idx = int(len(samples) * 0.8)
        train_samples = samples[:split_idx]
        val_samples = samples[split_idx:]

        # Save images for training and validation
        for dataset, dataset_samples, folder in [("train", train_samples, PATH_TRAIN), 
                                               ("validation", val_samples, PATH_VAL)]:
            print(f"Saving {len(dataset_samples)} images for {cl_name} in {dataset}")
            for idx, (cell, coords) in enumerate(dataset_samples):
                save_path = os.path.join(
                    folder, cl_name, 
                    f"{cl_name}_{idx}_{int(coords[0])}_{int(coords[1])}.jpg"
                )
                save_image((cell, save_path))

    # Create data.yaml for YOLO
    create_yaml_config()

def main():
    """Main function to create the dataset"""
    create_dataset()
    print("\nDataset creation completed!")

if __name__ == "__main__":
    main()
