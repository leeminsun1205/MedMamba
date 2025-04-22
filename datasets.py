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

        self.images = np.load(img_path, mmap_mode='r')
        self.labels = np.load(lbl_path, mmap_mode='r')

        self.single_label_in_file = (self.labels.ndim == 0)
        self.squeezable_labels = (self.labels.ndim > 1 and self.labels.shape[1] == 1)


    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        np_image = self.images[idx]
        label_raw = self.labels[idx]

        label = label_raw
        if self.squeezable_labels:
             label = np.squeeze(label_raw, axis=0)
        elif self.single_label_in_file:
             pass


        label = np.int64(label)


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
            # Fallback nếu không có transform (nên có ToTensor trong transform)
            # Chuyển đổi thủ công sang Tensor RGB float chuẩn
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            np_img_rgb = np.array(pil_image, dtype=np.float32)
            if np_img_rgb.ndim == 2: # Xử lý nếu convert('RGB') vẫn trả về 2D?
                 np_img_rgb = np.stack([np_img_rgb]*3, axis=-1)
            image_tensor = torch.from_numpy(np_img_rgb.transpose((2, 0, 1))) / 255.0


        return image_tensor, torch.tensor(label, dtype=torch.long)

    def get_num_classes(self):
         unique_labels = np.unique(self.labels)
         return len(unique_labels)

    def get_class_to_idx(self):
         unique_labels = sorted(np.unique(self.labels))
         return {f"class_{int(i)}": int(i) for i in unique_labels}