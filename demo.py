import os
import sys
import json
import argparse
import random
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms, datasets as torchvision_datasets

import matplotlib.pyplot as plt

from MedMamba import VSSM as medmamba
import datasets as custom_datasets

def parse_args_demo():
    parser = argparse.ArgumentParser(description='Demo prediction on a random test image.')
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to the model checkpoint .pth file.')
    parser.add_argument('--test_dir', type=str, required=True, help='Path to the test dataset (folder or directory containing .npy files or ImageFolder structure).')
    return parser.parse_args()

def main_demo():
    args = parse_args_demo()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device.")

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    if not os.path.isfile(args.checkpoint_path):
        print(f"Error: Checkpoint file not found at {args.checkpoint_path}")
        sys.exit(1)

    print(f"Loading checkpoint: {args.checkpoint_path}")
    checkpoint = torch.load(args.checkpoint_path, map_location=device)

    if 'num_classes' in checkpoint and 'class_indices' in checkpoint:
        num_classes = checkpoint['num_classes']
        class_indices = checkpoint['class_indices']
        print(f"Inferred number of classes: {num_classes}")
    else:
        print("Error: Checkpoint does not contain 'num_classes' or 'class_indices'. Cannot proceed.")
        sys.exit(1)

    test_is_npz = os.path.exists(os.path.join(args.test_dir, 'test_images.npy')) and \
                  os.path.exists(os.path.join(args.test_dir, 'test_labels.npy'))

    if test_is_npz:
        print(f"Loading test data from NPZ files in: {args.test_dir}")
        test_dataset = custom_datasets.NpzDataset(root_dir=args.test_dir, split='test', transform=test_transform)
    else:
        print(f"Loading test data from ImageFolder: {args.test_dir}")
        test_dataset = torchvision_datasets.ImageFolder(root=args.test_dir, transform=test_transform)

    if len(test_dataset) == 0:
        print("Error: Test dataset is empty.")
        sys.exit(1)

    model = medmamba(num_classes=num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    print("Model loaded successfully.")

    random_index = random.randint(0, len(test_dataset) - 1)
    image_tensor, true_label_index = test_dataset[random_index]

    with torch.no_grad():
        image_batch = image_tensor.unsqueeze(0).to(device)
        outputs = model(image_batch)
        probabilities = torch.softmax(outputs, dim=1)
        predicted_probability, predicted_index_tensor = torch.max(probabilities, dim=1)
        predicted_index = predicted_index_tensor.item()
        predicted_probability = predicted_probability.item()

    true_class_name = class_indices.get(str(true_label_index), f'Unknown ({true_label_index})')
    predicted_class_name = class_indices.get(str(predicted_index), f'Unknown ({predicted_index})')

    image_display = image_tensor.clone()
    mean = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
    std = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
    image_display = image_display * std + mean
    image_display = torch.clamp(image_display, 0, 1)
    image_display = image_display.permute(1, 2, 0).numpy()

    plt.imshow(image_display)
    color = 'green' if predicted_index == true_label_index else 'red'
    title = f"True: {true_class_name}\nPredicted: {predicted_class_name} ({predicted_probability:.2f})"
    plt.title(title, color=color)
    plt.axis('off')
    plt.show()

if __name__ == '__main__':
    main_demo()