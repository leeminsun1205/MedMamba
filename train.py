import os
import sys
import json
import argparse

import torch
import torch.nn as nn
from torchvision import transforms, datasets
import torch.optim as optim
from tqdm import tqdm

from Medmamba import VSSM as medmamba  # Your model
from datasets import MedMNISTDataset  # Custom dataset


def parse_args():
    parser = argparse.ArgumentParser(description='Train a Medmamba model.')
    parser.add_argument('--train_dir', type=str, help='Path to training image folder.')
    parser.add_argument('--val_dir', type=str, help='Path to validation image folder.')
    parser.add_argument('--train_npz', type=str, help='Path to training .npz file.')
    parser.add_argument('--val_npz', type=str, help='Path to validation .npz file.')
    parser.add_argument('--num_classes', type=int, required=True, help='Number of output classes.')
    parser.add_argument('--model_name', type=str, default='Medmamba', help='Model name for saving.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size.')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs.')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate.')
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using {} device.".format(device))

    data_transform = {
        "train": transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.Normalize((0.5,), (0.5,))  # áp dụng cho ảnh grayscale
        ]),
        "val": transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Normalize((0.5,), (0.5,))
        ])
    }

    # --- Load Dataset ---
    if args.train_npz and args.val_npz:
        print("Loading .npz MedMNIST datasets...")
        train_dataset = MedMNISTDataset(args.train_npz, transform=data_transform["train"])
        val_dataset = MedMNISTDataset(args.val_npz, transform=data_transform["val"])
        cla_dict = {i: f"class_{i}" for i in range(args.num_classes)}
    elif args.train_dir and args.val_dir:
        print("Loading image folder datasets...")
        train_dataset = datasets.ImageFolder(root=args.train_dir, transform=data_transform["train"])
        val_dataset = datasets.ImageFolder(root=args.val_dir, transform=data_transform["val"])
        cla_dict = {val: key for key, val in train_dataset.class_to_idx.items()}
    else:
        raise ValueError("You must provide either --train_npz and --val_npz or --train_dir and --val_dir")

    with open('class_indices.json', 'w') as json_file:
        json.dump(cla_dict, json_file, indent=4)

    train_num = len(train_dataset)
    val_num = len(val_dataset)

    nw = min([os.cpu_count(), args.batch_size if args.batch_size > 1 else 0, 8])
    print(f'Using {nw} dataloader workers every process')

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=nw)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=nw)

    print(f"Using {train_num} images for training, {val_num} images for validation.")

    net = medmamba(num_classes=args.num_classes)
    net.to(device)

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=args.lr)

    best_acc = 0.0
    save_path = f'./{args.model_name}_best.pth'
    train_steps = len(train_loader)

    for epoch in range(args.epochs):
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)

        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            outputs = net(images.to(device))
            loss = loss_function(outputs, labels.to(device))
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            train_bar.desc = f"train epoch[{epoch + 1}/{args.epochs}] loss:{loss:.3f}"

        net.eval()
        acc = 0.0
        with torch.no_grad():
            val_bar = tqdm(val_loader, file=sys.stdout)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

        val_accuracy = acc / val_num
        print(f'[epoch {epoch + 1}] train_loss: {running_loss / train_steps:.3f}  val_accuracy: {val_accuracy:.3f}')

        if val_accuracy > best_acc:
            best_acc = val_accuracy
            torch.save(net.state_dict(), save_path)

    print('Finished Training')


if __name__ == '__main__':
    main()
