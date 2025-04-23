import os
import sys
import json
import argparse
import numpy as np
from PIL import Image
import logging

import torch
import torch.nn as nn
from torchvision import transforms, datasets as torchvision_datasets
import torch.optim as optim
from tqdm import tqdm

from MedMamba import VSSM as medmamba
import datasets as custom_datasets

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def parse_args():
    parser = argparse.ArgumentParser(description='Train a Medmamba model.')
    parser.add_argument('--train_dir', type=str, required=True, help='Path to training dataset (folder or directory containing .npy files).')
    parser.add_argument('--val_dir', type=str, required=True, help='Path to validation dataset (folder or directory containing .npy files).')
    parser.add_argument('--num_classes', type=int, default=None, help='Number of output classes. If None and using NPZ, inferred from data.')
    parser.add_argument('--model_name', type=str, default='Medmamba', help='Model name for saving.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size.')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs.')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate.')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint .pth file to resume training from.')
    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using {device} device.")

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

    if train_is_npz:
        logging.info(f"Loading training data from NPZ files in: {args.train_dir}")
        train_dataset = custom_datasets.NpzDataset(root_dir=args.train_dir, split='train', transform=data_transform["train"])
        num_classes = train_dataset.get_num_classes()
        cla_dict_np = train_dataset.get_class_to_idx()
        cla_dict = {k: int(v) for k, v in cla_dict_np.items()}
    else:
        logging.info(f"Loading training data from ImageFolder: {args.train_dir}")
        train_dataset = torchvision_datasets.ImageFolder(root=args.train_dir, transform=data_transform["train"])
        num_classes = len(train_dataset.classes)
        cla_dict = {val: key for key, val in train_dataset.class_to_idx.items()}

    if args.num_classes is not None:
         if train_is_npz and args.num_classes != num_classes:
              logging.warning(f"Warning: --num_classes ({args.num_classes}) overrides inferred classes ({num_classes}) from NPZ.")
         num_classes = args.num_classes
    elif num_classes is None:
         logging.error("Error: Could not determine number of classes and --num_classes not specified.")
         sys.exit(1)

    train_num = len(train_dataset)

    class_indices_path = 'class_indices.json'
    logging.info(f"Saving class indices to {class_indices_path}")
    with open(class_indices_path, 'w') as json_file:
        json.dump(cla_dict, json_file, indent=4)

    val_is_npz = os.path.exists(os.path.join(args.val_dir, 'val_images.npy')) and \
                 os.path.exists(os.path.join(args.val_dir, 'val_labels.npy'))

    if val_is_npz:
        logging.info(f"Loading validation data from NPZ files in: {args.val_dir}")
        val_dataset = custom_datasets.NpzDataset(root_dir=args.val_dir, split='val', transform=data_transform["val"])
    else:
        logging.info(f"Loading validation data from ImageFolder: {args.val_dir}")
        val_dataset = torchvision_datasets.ImageFolder(root=args.val_dir, transform=data_transform["val"])


    val_num = len(val_dataset)

    nw = min([os.cpu_count(), args.batch_size if args.batch_size > 1 else 0, 8])
    logging.info(f'Using {nw} dataloader workers every process')

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=nw, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=nw, pin_memory=True)

    logging.info(f"Using {train_num} images for training, {val_num} images for validation.")
    logging.info(f"Number of classes: {num_classes}")


    net = medmamba(num_classes=num_classes)
    net.to(device)

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=args.lr)

    start_epoch = 0
    best_acc = 0.0
    save_path = f'./{args.model_name}_best.pth'

    if args.resume:
        checkpoint_path = args.resume
        if os.path.isfile(checkpoint_path):
            logging.info(f"Loading checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            net.load_state_dict(checkpoint['model_state_dict'])

            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                logging.info("Optimizer state loaded.")
            else:
                logging.warning("Optimizer state not found in checkpoint, starting optimizer from scratch.")

            if 'epoch' in checkpoint:
                start_epoch = checkpoint['epoch'] + 1
                logging.info(f"Resuming training from epoch {start_epoch}")
            else:
                 logging.warning("Epoch number not found in checkpoint, starting from epoch 0.")

            if 'best_acc' in checkpoint:
                best_acc = checkpoint['best_acc']
                logging.info(f"Loaded best accuracy: {best_acc:.3f}")
            else:
                 logging.warning("Best accuracy not found in checkpoint, starting best_acc from 0.0.")

        else:
            logging.error(f"Checkpoint file not found: {checkpoint_path}. Starting training from scratch.")
            start_epoch = 0
            best_acc = 0.0
    else:
        logging.info("No checkpoint provided, starting training from scratch.")


    train_steps = len(train_loader)

    for epoch in range(start_epoch, args.epochs):
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout, ncols=100, desc=f"Train Epoch {epoch+1}/{args.epochs}")

        for step, data in enumerate(train_bar):
            images, labels = data
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = net(images)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            train_bar.set_postfix(loss=f"{loss.item():.3f}")

        net.eval()
        acc = 0.0
        with torch.no_grad():
            val_bar = tqdm(val_loader, file=sys.stdout, ncols=100, desc=f"Valid Epoch {epoch+1}/{args.epochs}")
            for val_data in val_bar:
                val_images, val_labels = val_data
                val_images, val_labels = val_images.to(device), val_labels.to(device)
                outputs = net(val_images)
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels).sum().item()

        val_accuracy = acc / val_num
        avg_train_loss = running_loss / train_steps
        logging.info(f'[Epoch {epoch + 1}/{args.epochs}] Train Loss: {avg_train_loss:.3f} | Val Accuracy: {val_accuracy:.3f}')

        if val_accuracy > best_acc:
            best_acc = val_accuracy
            checkpoint_data = {
                'epoch': epoch,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
                'num_classes': num_classes,
                'class_indices': cla_dict
            }
            torch.save(checkpoint_data, save_path)
            logging.info(f'New best model checkpoint saved to {save_path} with accuracy: {best_acc:.3f}')

    logging.info(f'Finished Training. Best validation accuracy: {best_acc:.3f}')


if __name__ == '__main__':
    main()