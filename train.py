import os
import sys
import json
import argparse

import torch
import torch.nn as nn
from torchvision import transforms, datasets
import torch.optim as optim
from tqdm import tqdm

from MedMamba import VSSM as medmamba  # import model

# MedMNIST imports (for medmnist dataset type)
from medmnist import PathMNIST, ChestMNIST, DermaMNIST, OCTMNIST
from medmnist import PneumoniaMNIST, RetinaMNIST, BreastMNIST, BloodMNIST
from medmnist import TissueMNIST, OrganAMNIST, OrganCMNIST, OrganSMNIST

DATASET_MAPPING = { # For medmnist dataset type selection
    'pathmnist': PathMNIST,
    'chestmnist': ChestMNIST,
    'dermamnist': DermaMNIST,
    'octmnist': OCTMNIST,
    'pneumoniamnist': PneumoniaMNIST,
    'retinamnist': RetinaMNIST,
    'breastmnist': BreastMNIST,
    'bloodmnist': BloodMNIST,
    'tissuemnist': TissueMNIST,
    'organamnist': OrganAMNIST,
    'organcmnist': OrganCMNIST,
    'organsmnist': OrganSMNIST,
}

def parse_args():
    parser = argparse.ArgumentParser(description="Train Medmamba Model with Flexible Dataset")
    parser.add_argument('--dataset_type', type=str, required=True, choices=['imagefolder', 'medmnist'], help="Type of dataset to use: imagefolder or medmnist")

    # ImageFolder dataset arguments
    parser.add_argument('--train_dir', type=str, default=None, help="Path to training dataset (for imagefolder)")
    parser.add_argument('--val_dir', type=str, default=None, help="Path to validation dataset (for imagefolder)")

    # MedMNIST dataset arguments
    parser.add_argument('--medmnist_dataset', type=str, default='organamnist', choices=DATASET_MAPPING.keys(), help=f"MedMNIST dataset to use ({', '.join(DATASET_MAPPING.keys())}) (for medmnist)")

    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for training")
    parser.add_argument('--epochs', type=int, default=100, help="Number of training epochs")
    parser.add_argument('--lr', type=float, default=0.0001, help="Learning rate")
    parser.add_argument('--model_name', type=str, default='VSSMNet', help="Model save name")
    parser.add_argument('--size', type=int, default=224, help="Resize image size")

    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device.")

    size = args.size

    data_transform = {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(size), # Use size from argument
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # Default normalization
        ]),
        "val": transforms.Compose([
            transforms.Resize((size, size)), # Use size from argument
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # Default normalization
        ])
    }

    if args.dataset_type == 'imagefolder':
        if not args.train_dir or not args.val_dir:
            raise ValueError("For 'imagefolder' dataset_type, --train_dir and --val_dir must be provided.")
        train_dataset = datasets.ImageFolder(root=args.train_dir, transform=data_transform["train"])
        validate_dataset = datasets.ImageFolder(root=args.val_dir, transform=data_transform["val"])

        flower_list = train_dataset.class_to_idx
        cla_dict = dict((val, key) for key, val in flower_list.items())
        json_str = json.dumps(cla_dict, indent=4)
        with open('class_indices.json', 'w') as json_file:
            json_file.write(json_str)
    elif args.dataset_type == 'medmnist':
        medmnist_dataset_name = args.medmnist_dataset.lower()
        DatasetClass = DATASET_MAPPING[medmnist_dataset_name]
        train_dataset = DatasetClass(split="train", download=True, transform=data_transform["train"], size=size)
        validate_dataset = DatasetClass(split="val", download=True, transform=data_transform["val"], size=size)
        print(f"Using MedMNIST dataset: {medmnist_dataset_name.upper()}")
        print("Not saving class_indices.json for MedMNIST.")
    
        print("train_dataset.classes:", train_dataset.classes)  # ADD THIS LINE
    
        num_classes = len(train_dataset.classes) if hasattr(train_dataset, 'classes') else len(train_dataset.label_names)
        if not hasattr(train_dataset, 'classes') and not hasattr(train_dataset, 'label_names'):
            print("Warning: Cannot automatically determine num_classes for MedMNIST. Trying to infer from labels.")
            num_classes = len(torch.unique(torch.tensor(train_dataset.labels)))
            print(f"Inferred num_classes from labels: {num_classes}. Please verify.")
    else:
        raise ValueError(f"Invalid dataset_type: {args.dataset_type}. Choose 'imagefolder' or 'medmnist'.")

    train_num = len(train_dataset)
    val_num = len(validate_dataset)
    print(f"using {train_num} images for training, {val_num} images for validation.")

    if args.dataset_type == 'imagefolder':
        num_classes = len(train_dataset.classes)
    elif args.dataset_type == 'medmnist':
        num_classes = len(train_dataset.classes) if hasattr(train_dataset, 'classes') else len(train_dataset.label_names)
        if not hasattr(train_dataset, 'classes') and not hasattr(train_dataset, 'label_names'):
            print("Warning: Cannot automatically determine num_classes for MedMNIST. Trying to infer from labels.")
            num_classes = len(torch.unique(torch.tensor(train_dataset.labels)))
            print(f"Inferred num_classes from labels: {num_classes}. Please verify.")
    else: # Should not reach here due to argparse choices
        num_classes = None # To satisfy type checker, but logic error if reaches here

    print(f"Number of classes: {num_classes}")

    nw = min([os.cpu_count(), args.batch_size if args.batch_size > 1 else 0, 8])
    print(f'Using {nw} dataloader workers every process')

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size, shuffle=True,
                                               num_workers=nw)

    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=args.batch_size, shuffle=False,
                                                  num_workers=nw)


    net = medmamba(num_classes=num_classes) # num_classes is now dynamically determined
    net.to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=args.lr)

    epochs = args.epochs
    best_acc = 0.0
    save_path = f'./{args.model_name}_{args.dataset_type.upper()}.pth' # Include dataset type in save path
    train_steps = len(train_loader)
    for epoch in range(epochs):
        # train
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

            # print statistics
            running_loss += loss.item()

            train_bar.desc = f"train epoch[{epoch + 1}/{epochs}] loss:{loss:.3f}"

        # validate
        net.eval()
        acc = 0.0  # accumulate accurate number / epoch
        with torch.no_grad():
            val_bar = tqdm(validate_loader, file=sys.stdout)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

        val_accurate = acc / val_num
        print(f'[epoch {epoch + 1}] train_loss: {running_loss / train_steps:.3f}  val_accuracy: {val_accurate:.3f}')

        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)

    print('Finished Training')


if __name__ == '__main__':
    main()
