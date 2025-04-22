import os
import sys
import json
import argparse
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast # Thêm import cho AMP
from torchvision import transforms, datasets as torchvision_datasets
import torch.optim as optim
from tqdm import tqdm

from MedMamba import VSSM as medmamba
import datasets as custom_datasets # Đảm bảo dùng datasets.py đã sửa ở Bước 1


def parse_args():
    parser = argparse.ArgumentParser(description='Train a Medmamba model.')
    parser.add_argument('--train_dir', type=str, required=True, help='Path to training dataset.')
    parser.add_argument('--val_dir', type=str, required=True, help='Path to validation dataset.')
    parser.add_argument('--num_classes', type=int, default=None, help='Number of output classes.')
    parser.add_argument('--model_name', type=str, default='Medmamba', help='Model name for saving.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size.')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs.')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate.')
    # Thêm tùy chọn bật/tắt AMP
    parser.add_argument('--use_amp', action='store_true', help='Use Automatic Mixed Precision.')
    return parser.parse_args()


def main():
    args = parse_args()

    # Giữ nguyên phần xác định device (chỉ dùng 1 GPU)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using {} device.".format(device))

    # Khởi tạo GradScaler nếu dùng AMP và có GPU
    scaler = GradScaler(enabled=args.use_amp and torch.cuda.is_available())
    print(f"AMP enabled: {scaler.is_enabled()}")

    data_transform = {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]),
        "val": transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    }

    train_is_npz = os.path.exists(os.path.join(args.train_dir, 'train_images.npy')) and \
                   os.path.exists(os.path.join(args.train_dir, 'train_labels.npy'))

    # Tối ưu: Chỉ gọi get_num_classes/get_class_to_idx nếu thực sự cần
    # Hoặc tốt nhất là dùng --num_classes
    num_classes = args.num_classes
    cla_dict = {}

    if train_is_npz:
        print(f"Loading training data from NPZ files in: {args.train_dir}")
        train_dataset = custom_datasets.NpzDataset(root_dir=args.train_dir, split='train', transform=data_transform["train"])
        if num_classes is None:
             print("Attempting to get num_classes from dataset (might be slow/RAM intensive)...")
             num_classes = train_dataset.get_num_classes()
             print(f"Inferred number of classes: {num_classes}")
             # Chỉ lấy cla_dict nếu cần thiết và num_classes được suy ra
             print("Attempting to get class_to_idx from dataset (might be slow/RAM intensive)...")
             cla_dict = train_dataset.get_class_to_idx()

    else:
        print(f"Loading training data from ImageFolder: {args.train_dir}")
        train_dataset = torchvision_datasets.ImageFolder(root=args.train_dir, transform=data_transform["train"])
        inferred_num_classes = len(train_dataset.classes)
        if num_classes is None:
            num_classes = inferred_num_classes
            cla_dict = {val: key for key, val in train_dataset.class_to_idx.items()}
        elif num_classes != inferred_num_classes:
             print(f"Warning: --num_classes ({num_classes}) overrides inferred classes ({inferred_num_classes}) from ImageFolder.")
             # Tạo cla_dict thủ công nếu cần dựa trên num_classes
             cla_dict = {i: f"class_{i}" for i in range(num_classes)}


    if num_classes is None:
         print("Error: Could not determine number of classes. Please provide --num_classes.")
         sys.exit(1)

    train_num = len(train_dataset)

    if cla_dict: # Chỉ ghi file nếu cla_dict được tạo
        try:
             with open('class_indices.json', 'w') as json_file:
                 json.dump(cla_dict, json_file, indent=4)
        except Exception as e:
             print(f"Could not write class_indices.json: {e}")


    val_is_npz = os.path.exists(os.path.join(args.val_dir, 'val_images.npy')) and \
                 os.path.exists(os.path.join(args.val_dir, 'val_labels.npy'))

    if val_is_npz:
        print(f"Loading validation data from NPZ files in: {args.val_dir}")
        val_dataset = custom_datasets.NpzDataset(root_dir=args.val_dir, split='val', transform=data_transform["val"])
    else:
        print(f"Loading validation data from ImageFolder: {args.val_dir}")
        val_dataset = torchvision_datasets.ImageFolder(root=args.val_dir, transform=data_transform["val"])

    val_num = len(val_dataset)

    nw = min([os.cpu_count(), args.batch_size if args.batch_size > 1 else 0, 8])
    print(f'Using {nw} dataloader workers every process')

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=nw, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=nw, pin_memory=True)

    print(f"Using {train_num} images for training, {val_num} images for validation.")
    print(f"Number of classes: {num_classes}")

    # Cân nhắc thêm Gradient Checkpointing ở đây nếu cần
    # net = medmamba(num_classes=num_classes, use_checkpoint=True)
    net = medmamba(num_classes=num_classes)
    net.to(device)

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=args.lr)

    best_acc = 0.0
    save_path = f'./{args.model_name}_best_amp.pth' if args.use_amp else f'./{args.model_name}_best.pth'
    train_steps = len(train_loader)

    for epoch in range(args.epochs):
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout, ncols=100)

        for step, data in enumerate(train_bar):
            images, labels = data
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad(set_to_none=True) # Tối ưu nhỏ

            # Bọc forward pass và loss bằng autocast
            with autocast(enabled=scaler.is_enabled()):
                outputs = net(images)
                loss = loss_function(outputs, labels)

            # Scale loss và backward
            scaler.scale(loss).backward()

            # Unscale gradients và step
            scaler.step(optimizer)

            # Update scaler
            scaler.update()

            running_loss += loss.item()
            train_bar.set_description(f"train epoch[{epoch + 1}/{args.epochs}] loss:{loss.item():.3f}")

        net.eval()
        acc = 0.0
        with torch.no_grad():
            val_bar = tqdm(val_loader, file=sys.stdout, ncols=100)
            for val_data in val_bar:
                val_images, val_labels = val_data
                val_images, val_labels = val_images.to(device), val_labels.to(device)
                # Dùng autocast cho validation nếu dùng AMP
                with autocast(enabled=scaler.is_enabled()):
                    outputs = net(val_images)
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels).sum().item()
                val_bar.set_description(f"valid epoch[{epoch + 1}/{args.epochs}]")


        val_accuracy = acc / val_num
        avg_train_loss = running_loss / train_steps
        print(f'[epoch {epoch + 1}] train_loss: {avg_train_loss:.3f}  val_accuracy: {val_accuracy:.3f}')

        if val_accuracy > best_acc:
            best_acc = val_accuracy
            # Lưu state_dict gốc, không cần .module
            torch.save(net.state_dict(), save_path)
            print(f'New best model saved to {save_path} with accuracy: {best_acc:.3f}')


    print(f'Finished Training. Best validation accuracy: {best_acc:.3f}')


if __name__ == '__main__':
    main()