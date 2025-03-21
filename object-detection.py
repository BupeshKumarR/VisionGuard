import os
import xml.etree.ElementTree as ET



# Define paths based on your structure (local Colab storage)
image_dir = '/content/dataset/images'
annotation_dir = '/content/dataset/annotations'
output_dir = '/content/dataset/labels'

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

def convert_to_yolo(xml_file, img_width, img_height):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    for obj in root.findall('object'):
        name = obj.find('name').text
        if name != 'licence':  # Match your label
            continue
        bndbox = obj.find('bndbox')
        xmin = float(bndbox.find('xmin').text)
        ymin = float(bndbox.find('ymin').text)
        xmax = float(bndbox.find('xmax').text)
        ymax = float(bndbox.find('ymax').text)
        
        x_center = (xmin + xmax) / 2 / img_width
        y_center = (ymin + ymax) / 2 / img_height
        width = (xmax - xmin) / img_width
        height = (ymax - ymin) / img_height
        return f"0 {x_center} {y_center} {width} {height}"
    return None

# Convert all XML files
for xml_file in os.listdir(annotation_dir):
    if xml_file.endswith('.xml'):
        img_name = xml_file.replace('.xml', '.png')  # Match your image extension
        img_path = os.path.join(image_dir, img_name)
        if not os.path.exists(img_path):
            print(f"Image {img_name} not found, skipping...")
            continue
        with open(img_path) as img:
            img_width, img_height = img.size
        
        yolo_annotation = convert_to_yolo(os.path.join(annotation_dir, xml_file), img_width, img_height)
        if yolo_annotation:
            with open(os.path.join(output_dir, xml_file.replace('.xml', '.txt')), 'w') as f:
                f.write(yolo_annotation)
        else:
            print(f"No 'licence' object found in {xml_file}, skipping...")

import shutil
import random

# Define paths
image_dir = '/content/dataset/images'
label_dir = '/content/dataset/labels'

# Create train and val subdirectories
train_img_dir = '/content/dataset/images/train'
val_img_dir = '/content/dataset/images/val'
train_label_dir = '/content/dataset/labels/train'
val_label_dir = '/content/dataset/labels/val'

os.makedirs(train_img_dir, exist_ok=True)
os.makedirs(val_img_dir, exist_ok=True)
os.makedirs(train_label_dir, exist_ok=True)
os.makedirs(val_label_dir, exist_ok=True)

# Get all images and shuffle
all_images = [f for f in os.listdir(image_dir) if f.endswith('.png')]
random.shuffle(all_images)

# Split 80% train, 20% val
train_size = int(0.8 * len(all_images))
train_images = all_images[:train_size]
val_images = all_images[train_size:]

# Copy files to train
for img in train_images:
    shutil.copy(os.path.join(image_dir, img), train_img_dir)
    label_file = img.replace('.png', '.txt')
    if os.path.exists(os.path.join(label_dir, label_file)):
        shutil.copy(os.path.join(label_dir, label_file), train_label_dir)

# Copy files to val
for img in val_images:
    shutil.copy(os.path.join(image_dir, img), val_img_dir)
    label_file = img.replace('.png', '.txt')
    if os.path.exists(os.path.join(label_dir, label_file)):
        shutil.copy(os.path.join(label_dir, label_file), val_label_dir)

with open('/content/dataset/data.yaml', 'w') as f:
    f.write('''
train: /content/dataset/images/train
val: /content/dataset/images/val
nc: 1
names: ['licence']
''')
    
from ultralytics import YOLO

# Load the nano model (fastest)
model = YOLO('yolov8n.pt')

# Train with minimal epochs
model.train(data='/content/dataset/data.yaml', epochs=10, imgsz=640, batch=16)

model.val()

results = model('/content/dataset/images/Cars0.png', conf=0.1)  # Lower confidence threshold
results