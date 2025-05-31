import os
import sys
import json
import argparse
import numpy as np
# from PIL import Image
import logging
import random
import torch
import torch.nn as nn
from torchvision import transforms, datasets as torchvision_datasets
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm

from MedMamba import VSSM as medmamba
import datasets as custom_datasets

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def check_early_stopping(epochs_without_improvement, patience, epoch, total_epochs):
    if epochs_without_improvement >= patience:
        logging.info(f"Early stopping triggered after {patience} epochs without improvement at epoch {epoch}/{total_epochs}.")
        print(f"Early stopping triggered after {patience} epochs without improvement at epoch {epoch}/{total_epochs}.")
        return True
    return False

def parse_args():
    parser = argparse.ArgumentParser(description='Train a Medmamba model.')
    parser.add_argument('--medmb_size', type=str, default='T', choices=['T', 'S', 'B'], help='Choose medmb size: T, S, or B')
    parser.add_argument('--train_dir', type=str, required=True, help='Path to training dataset (folder or directory containing .npy files).')
    parser.add_argument('--val_dir', type=str, required=True, help='Path to validation dataset (folder or directory containing .npy files).')
    parser.add_argument('--num_classes', type=int, default=None, help='Number of output classes. If None and using NPZ, inferred from data.')
    parser.add_argument('--model_name', type=str, default='Medmamba', help='Model name for saving.')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size. If None, determined by dataset type.')
    parser.add_argument('--epochs', type=int, default=None, help='Number of training epochs. If None, determined by dataset type.')
    parser.add_argument('--lr', type=float, default=None, help='Learning rate. If None, determined by dataset type.')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint .pth file to resume training from.')
    parser.add_argument('--patience', type=int, default=25, help='Patience for early stopping.')
    parser.add_argument('--save_dir', type=str, default='.', help='Directory to save checkpoints.')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--use_early_stopping', action='store_true', default=False)
    return parser.parse_args()

def main():
    args = parse_args()

    set_seed(args.seed)

    os.makedirs(args.save_dir, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using {device} device.")
    print(f"Using {device} device.")

    train_is_npz = os.path.exists(os.path.join(args.train_dir, 'train_images.npy')) and \
                   os.path.exists(os.path.join(args.train_dir, 'train_labels.npy'))

    if train_is_npz:
        logging.info(f"Detected MedMNIST (NPZ) dataset.")
        print(f"Detected MedMNIST (NPZ) dataset.")
        epochs = args.epochs if args.epochs is not None else 100
        batch_size = args.batch_size if args.batch_size is not None else 100
        lr = args.lr if args.lr is not None else 0.001
        lr_decay_epochs = [50, 75]

    else:
        logging.info(f"Detected non-MedMNIST dataset (ImageFolder).")
        print(f"Detected non-MedMNIST dataset (ImageFolder).")
        epochs = args.epochs if args.epochs is not None else 150
        batch_size = args.batch_size if args.batch_size is not None else 64
        lr = args.lr if args.lr is not None else 0.0001
        lr_decay_epochs = []
        
    data_transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    data_transform_val = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    data_transform = {
        "train": data_transform_train,
        "val": data_transform_val
    }

    if train_is_npz:
        logging.info(f"Loading training data from NPZ files in: {args.train_dir}")
        print(f"Loading training data from NPZ files in: {args.train_dir}")
        train_dataset = custom_datasets.NpzDataset(root_dir=args.train_dir, split='train', transform=data_transform["train"])
        num_classes = train_dataset.get_num_classes()
        cla_dict_np = train_dataset.get_class_to_idx()
        cla_dict = {k: int(v) for k, v in cla_dict_np.items()}
    else:
        logging.info(f"Loading training data from ImageFolder: {args.train_dir}")
        print(f"Loading training data from ImageFolder: {args.train_dir}")
        train_dataset = torchvision_datasets.ImageFolder(root=args.train_dir, transform=data_transform["train"])
        num_classes = len(train_dataset.classes)
        cla_dict = {val: key for key, val in train_dataset.class_to_idx.items()}

    if args.num_classes is not None:
         if train_is_npz and args.num_classes != num_classes:
              logging.warning(f"Warning: --num_classes ({args.num_classes}) overrides inferred classes ({num_classes}) from NPZ.")
              print(f"Warning: --num_classes ({args.num_classes}) overrides inferred classes ({num_classes}) from NPZ.")
         num_classes = args.num_classes
    elif num_classes is None:
         logging.error("Error: Could not determine number of classes and --num_classes not specified.")
         print("Error: Could not determine number of classes and --num_classes not specified.")
         sys.exit(1)

    train_num = len(train_dataset)

    class_indices_path = os.path.join(args.save_dir, 'class_indices.json')
    logging.info(f"Saving class indices to {class_indices_path}")
    print(f"Saving class indices to {class_indices_path}")
    with open(class_indices_path, 'w') as json_file:
        json.dump(cla_dict, json_file, indent=4)

    val_is_npz = os.path.exists(os.path.join(args.val_dir, 'val_images.npy')) and \
                 os.path.exists(os.path.join(args.val_dir, 'val_labels.npy'))

    if val_is_npz:
        logging.info(f"Loading validation data from NPZ files in: {args.val_dir}")
        print(f"Loading validation data from NPZ files in: {args.val_dir}")
        val_dataset = custom_datasets.NpzDataset(root_dir=args.val_dir, split='val', transform=data_transform["val"])
    else:
        logging.info(f"Loading validation data from ImageFolder: {args.val_dir}")
        print(f"Loading validation data from ImageFolder: {args.val_dir}")
        val_dataset = torchvision_datasets.ImageFolder(root=args.val_dir, transform=data_transform["val"])

    val_num = len(val_dataset)

    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    logging.info(f'Using {nw} dataloader workers every process')
    print(f'Using {nw} dataloader workers every process')

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=nw, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=nw, pin_memory=True)

    logging.info(f"Using {train_num} images for training, {val_num} images for validation.")
    print(f"Using {train_num} images for training, {val_num} images for validation.")
    logging.info(f"Number of classes: {num_classes}")
    print(f"Number of classes: {num_classes}")
    logging.info(f"Epochs: {epochs}, Batch Size: {batch_size}, Initial LR: {lr}")
    print(f"Epochs: {epochs}, Batch Size: {batch_size}, Initial LR: {lr}")
    
    if args.medmb_size == 'T': net = medmamba(depths=[2, 2, 4, 2],dims=[96,192,384,768],num_classes=num_classes)
    elif args.medmb_size == 'S': net = medmamba(depths=[2, 2, 8, 2],dims=[96,192,384,768],num_classes=num_classes)
    else: net = medmamba(depths=[2, 2, 12, 2],dims=[128,256,512,1024],num_classes=num_classes)
    logging.info(f'Model size: "{args.medmb_size}"')
    print(f'Model size: "{args.medmb_size}"')
    net.to(device)

    loss_function = nn.CrossEntropyLoss()

    if train_is_npz:
        optimizer = optim.AdamW(net.parameters(), lr=lr)
    else:
        optimizer = optim.AdamW(net.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=1e-4)

    if train_is_npz and lr_decay_epochs:
        scheduler = MultiStepLR(optimizer, milestones=lr_decay_epochs, gamma=0.1)
        logging.info(f"Using MultiStepLR with milestones: {lr_decay_epochs} and gamma: 0.1")
        print(f"Using MultiStepLR with milestones: {lr_decay_epochs} and gamma: 0.1")
    else:
        scheduler = None
        logging.info("No learning rate scheduler applied.")
        print("No learning rate scheduler applied.")

    start_epoch = 1
    best_acc = 0.0
    best_save_path = None
    epochs_without_improvement = 0

    if args.resume:
        checkpoint_path = args.resume
        if os.path.isfile(checkpoint_path):
            logging.info(f"Loading checkpoint: {checkpoint_path}")
            print(f"Loading checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            net.load_state_dict(checkpoint['model_state_dict'])

            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                logging.info("Optimizer state loaded.")
                print("Optimizer state loaded.")
            else:
                logging.warning("Optimizer state not found in checkpoint, starting optimizer from scratch.")
                print("Warning: Optimizer state not found in checkpoint, starting optimizer from scratch.")

            if scheduler and 'scheduler_state_dict' in checkpoint:
                 scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                 logging.info("Scheduler state loaded.")
                 print("Scheduler state loaded.")
            elif scheduler:
                 logging.warning("Scheduler state not found in checkpoint. Scheduler will start without loaded state.")
                 print("Warning: Scheduler state not found in checkpoint. Scheduler will start without loaded state.")

            if 'epoch' in checkpoint:
                start_epoch = checkpoint['epoch'] + 1
                logging.info(f"Resuming training from epoch {start_epoch}")
                print(f"Resuming training from epoch {start_epoch}")
            else:
                 logging.warning("Epoch number not found in checkpoint, starting from epoch 1.")
                 print("Warning: Epoch number not found in checkpoint, starting from epoch 1.")
                 start_epoch = 1

            if 'best_acc' in checkpoint:
                best_acc = checkpoint['best_acc']
                logging.info(f"Loaded best accuracy: {best_acc:.3f}")
                print(f"Loaded best accuracy: {best_acc:.3f}")
            else:
                 logging.warning("Best accuracy not found in checkpoint, starting best_acc from 0.0.")
                 print("Warning: Best accuracy not found in checkpoint, starting best_acc from 0.0.")

            best_save_path = None

        else:
            logging.error(f"Checkpoint file not found: {checkpoint_path}. Starting training from scratch.")
            print(f"Error: Checkpoint file not found: {checkpoint_path}. Starting training from scratch.")
            start_epoch = 1
            best_acc = 0.0
            best_save_path = None
    else:
        logging.info("No checkpoint provided, starting training from epoch 1.")
        print("No checkpoint provided, starting training from epoch 1.")
        best_save_path = None

    if epochs < start_epoch:
        logging.warning(f"Target epochs ({epochs}) is less than start epoch ({start_epoch}). No training will occur.")
        print(f"Warning: Target epochs ({epochs}) is less than start epoch ({start_epoch}). No training will occur.")
        print(f'Finished Training (Target Epoch <= Start Epoch). Best validation accuracy recorded: {best_acc:.3f}')
        sys.exit(0)

    train_steps = len(train_loader)
    final_epoch_reached = start_epoch - 1

    for epoch in range(start_epoch, epochs + 1):
        final_epoch_reached = epoch
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout, ncols=100, desc=f"Train Epoch {epoch}/{epochs}")

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

        if scheduler:
            scheduler.step()

        net.eval()
        acc = 0.0
        with torch.no_grad():
            val_bar = tqdm(val_loader, file=sys.stdout, ncols=100, desc=f"Valid Epoch {epoch}/{epochs}")
            for val_data in val_bar:
                val_images, val_labels = val_data
                val_images, val_labels = val_images.to(device), val_labels.to(device)
                outputs = net(val_images)
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels).sum().item()

        val_accuracy = acc / val_num
        avg_train_loss = running_loss / train_steps
        log_message = f'[Epoch {epoch}/{epochs}] Train Loss: {avg_train_loss:.3f} | Val Accuracy: {val_accuracy:.3f}'
        logging.info(log_message)
        print(log_message)

        checkpoint_data = {
            'epoch': epoch,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_acc': best_acc,
            'num_classes': num_classes,
            'class_indices': cla_dict
        }
        if scheduler:
            checkpoint_data['scheduler_state_dict'] = scheduler.state_dict()

        if val_accuracy > best_acc:
            best_acc = val_accuracy
            epochs_without_improvement = 0

            checkpoint_data['best_acc'] = best_acc

            new_best_save_path = os.path.join(args.save_dir, f'{args.model_name}_epoch_{epoch}_best.pth')
            torch.save(checkpoint_data, new_best_save_path)
            log_save_message = f'New best model checkpoint saved to {new_best_save_path} with accuracy: {best_acc:.3f}'
            logging.info(log_save_message)
            print(log_save_message)

            if best_save_path and os.path.exists(best_save_path) and best_save_path != new_best_save_path:
                 log_remove_message = f"Removing old best checkpoint: {best_save_path}"
                 logging.info(log_remove_message)
                 print(log_remove_message)
                 os.remove(best_save_path)

            best_save_path = new_best_save_path
        else:
            epochs_without_improvement += 1
            logging.info(f"Validation accuracy did not improve. Patience: {epochs_without_improvement}/{args.patience}")
            print(f"Validation accuracy did not improve. Patience: {epochs_without_improvement}/{args.patience}")

        if args.use_early_stopping:
            if check_early_stopping(epochs_without_improvement, args.patience, epoch, epochs):
                break

    last_save_path = os.path.join(args.save_dir, f'{args.model_name}_epoch_{final_epoch_reached}_last.pth')

    final_checkpoint_data = {
        'epoch': final_epoch_reached,
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_acc': best_acc,
        'num_classes': num_classes,
        'class_indices': cla_dict
    }
    if scheduler:
        final_checkpoint_data['scheduler_state_dict'] = scheduler.state_dict()

    torch.save(final_checkpoint_data, last_save_path)
    log_last_save_message = f"Saved last checkpoint to {last_save_path}"
    logging.info(log_last_save_message)
    print(log_last_save_message)

    log_finish_message = f'Finished Training. Final Epoch Reached: {final_epoch_reached}. Best validation accuracy: {best_acc:.3f}'
    logging.info(log_finish_message)
    print(log_finish_message)

if __name__ == '__main__':
    main()