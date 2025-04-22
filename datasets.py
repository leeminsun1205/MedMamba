import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

class NpzDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform

        img_path = os.path.join(self.root_dir, f"{self.split}_images.npy")
        lbl_path = os.path.join(self.root_dir, f"{self.split}_labels.npy")

        self.images = np.load(img_path)
        self.labels = np.load(lbl_path)

        if self.labels.ndim > 1 and self.labels.shape[1] == 1:
             self.labels = self.labels.squeeze(1)
        elif self.labels.ndim == 0: # Handle case where labels might be saved as a 0-dim array if only one label exists
             self.labels = self.labels[np.newaxis]


        # Ensure labels are integer type for CrossEntropyLoss
        self.labels = self.labels.astype(np.int64)


    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        # Convert numpy array to PIL Image
        # Assumes images are stored in HWC format and uint8 type
        if image.ndim == 2: # Grayscale image
             image = Image.fromarray(image, mode='L')
        elif image.ndim == 3:
             image = Image.fromarray(image, mode='RGB')
        else:
            # Handle potential other formats or raise an error
            # For now, assume HWC or HW
            raise ValueError(f"Unsupported image dimensions: {image.ndim}")


        if self.transform:
            image = self.transform(image)

        # Ensure label is a scalar tensor if needed by the loss function later
        # However, standard CrossEntropyLoss usually handles python int or long tensor directly
        # label = torch.tensor(label, dtype=torch.long)


        return image, label

    def get_num_classes(self):
         # Calculate number of classes from unique labels
         return len(np.unique(self.labels))

    def get_class_to_idx(self):
         # Create a basic index mapping for NpzDataset if needed
         unique_labels = sorted(np.unique(self.labels))
         return {f"class_{i}": i for i in unique_labels} # Map numeric label to "class_X"