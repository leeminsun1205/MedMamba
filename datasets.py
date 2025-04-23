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
        elif self.labels.ndim == 0:
             self.labels = self.labels[np.newaxis]

        self.labels = self.labels.astype(np.int64)


    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        np_image = self.images[idx]
        label = self.labels[idx]

        if np_image.ndim == 2:
             pil_image = Image.fromarray(np_image, mode='L')
             pil_image = pil_image.convert('RGB')
        elif np_image.ndim == 3:
             pil_image = Image.fromarray(np_image, mode='RGB')
        else:
            raise ValueError(f"Unsupported image dimensions: {np_image.ndim}")

        if self.transform:
            image_tensor = self.transform(pil_image)
        else:
             image_tensor = pil_image

        return image_tensor, label

    def get_num_classes(self):
         return len(np.unique(self.labels))

    def get_class_to_idx(self):
         unique_labels = sorted(np.unique(self.labels))
         return {f"class_{i}": i for i in unique_labels}