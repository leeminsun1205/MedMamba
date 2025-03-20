import os
import sys
import json
import argparse

import torch
import torch.nn as nn
from torchvision import transforms, datasets
import torch.optim as optim
from tqdm import tqdm

from Medmamba import VSSM as medmamba  # import model

def parse_args():
    parser = argparse.ArgumentParser(description="Train Medmamba Model")
    parser.add_argument('--train_dir', type=str, required=True, help="Path to training dataset")
    parser.add_argument('--val_dir', type=str, required=True, help="Path to validation dataset")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for training")
    parser.add_argument('--epochs', type=int, default=100, help="Number of training epochs")
    parser.add_argument('--lr', type=float, default=0.0001, help="Learning rate")
    parser.add_argument('--model_name', type=str, default='VSSMNet', help="Model save name")
    return parser.parse_args()

def main():
    args = parse_args()
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device.")
    
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

    train_dataset = datasets.ImageFolder(root=args.train_dir, transform=data_transform["train"])
    validate_dataset = datasets.ImageFolder(root=args.val_dir, transform=data_transform["val"])
    
    train_num = len(train_dataset)
    val_num = len(validate_dataset)
    
    class_indices = {v: k for k, v in train_dataset.class_to_idx.items()}
    with open('class_indices.json', 'w') as json_file:
        json.dump(class_indices, json_file, indent=4)
    
    nw = min([os.cpu_count(), args.batch_size if args.batch_size > 1 else 0, 8])  # num workers
    print(f'Using {nw} dataloader workers every process')
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=nw)
    validate_loader = torch.utils.data.DataLoader(validate_dataset, batch_size=args.batch_size, shuffle=False, num_workers=nw)
    
    num_classes = len(train_dataset.classes)
    net = medmamba(num_classes=num_classes).to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    
    best_acc = 0.0
    save_path = f'./{args.model_name}.pth'
    train_steps = len(train_loader)
    
    for epoch in range(args.epochs):
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)
        
        for step, (images, labels) in enumerate(train_bar):
            optimizer.zero_grad()
            outputs = net(images.to(device))
            loss = loss_function(outputs, labels.to(device))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            train_bar.desc = f"train epoch[{epoch+1}/{args.epochs}] loss:{loss:.3f}"
        
        # Validate
        net.eval()
        acc = 0.0
        with torch.no_grad():
            val_bar = tqdm(validate_loader, file=sys.stdout)
            for val_images, val_labels in val_bar:
                outputs = net(val_images.to(device))
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()
        
        val_acc = acc / val_num
        print(f'[epoch {epoch+1}] train_loss: {running_loss/train_steps:.3f}  val_accuracy: {val_acc:.3f}')
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(net.state_dict(), save_path)
    
    print('Finished Training')

if __name__ == '__main__':
    main()
