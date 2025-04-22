# datasets.py

import numpy as np
import torch
from torch.utils.data import Dataset

class MedMNISTDataset(Dataset):
    def __init__(self, npz_path, transform=None):
        data = np.load(npz_path)
        self.images = data['images']  # (N, H, W) hoặc (N, H, W, C)
        self.labels = data['labels']  # (N,) hoặc (N, 1)
        self.transform = transform

        # Chuyển về (N, 1, H, W) nếu cần
        if len(self.images.shape) == 3:
            self.images = self.images[:, np.newaxis, :, :]
        elif len(self.images.shape) == 4 and self.images.shape[-1] == 1:
            self.images = self.images.transpose(0, 3, 1, 2)

        self.images = self.images.astype(np.float32) / 255.0

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = torch.tensor(self.images[idx])
        label = int(self.labels[idx]) if self.labels.ndim == 1 else int(self.labels[idx][0])

        if self.transform:
            image = self.transform(image)

        return image, label
