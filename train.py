import os
import sys
import json
import argparse

import torch
import torch.nn as nn
from torchvision import transforms
import torch.optim as optim
from tqdm import tqdm

from MedMamba import VSSM as medmamba  # import model

# Import MedMNIST datasets - để có sẵn các class dataset
from medmnist import PathMNIST, ChestMNIST, DermaMNIST, OCTMNIST
from medmnist import PneumoniaMNIST, RetinaMNIST, BreastMNIST, BloodMNIST
from medmnist import TissueMNIST, OrganAMNIST, OrganCMNIST, OrganSMNIST

# Dictionary ánh xạ tên dataset (dòng lệnh) với class dataset MedMNIST
DATASET_MAPPING = {
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
    parser = argparse.ArgumentParser(description="Train Medmamba Model with MedMNIST")
    parser.add_argument('--dataset', type=str, required=True, choices=DATASET_MAPPING.keys(), help=f"MedMNIST dataset to use ({', '.join(DATASET_MAPPING.keys())})") # Thêm đối số --dataset, giới hạn lựa chọn
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for training")
    parser.add_argument('--epochs', type=int, default=100, help="Number of training epochs")
    parser.add_argument('--lr', type=float, default=0.0001, help="Learning rate")
    parser.add_argument('--model_name', type=str, default='VSSM_MedMNIST_Net', help="Model save name")
    return parser.parse_args()

def main():
    args = parse_args()
    dataset_name = args.dataset.lower() # Lấy tên dataset từ đối số dòng lệnh và chuyển về lowercase

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device.")

    # Transformations - có thể cần điều chỉnh cho từng dataset cụ thể nếu cần
    data_transform = {
        "train": transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224)), # Resize cho input model
            transforms.Normalize(mean=[0.5], std=[0.5]) # Normalize grayscale
        ]),
        "val": transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=[0.5], std=[0.5]) # Normalize grayscale
        ])
    }

    # Tải dataset MedMNIST động dựa trên đối số dòng lệnh
    DatasetClass = DATASET_MAPPING[dataset_name] # Lấy class dataset từ dictionary dựa trên tên
    train_dataset = DatasetClass(split="train", download=True, transform=data_transform["train"], size=224) # size=224 để resize khi load
    val_dataset = DatasetClass(split="val", download=True, transform=data_transform["val"], size=224)

    train_num = len(train_dataset)
    val_num = len(val_dataset)

    print(f"Using MedMNIST dataset: {dataset_name.upper()}") # In tên dataset đang dùng
    print("Không lưu class_indices.json vì dùng MedMNIST.")


    nw = min([os.cpu_count(), args.batch_size if args.batch_size > 1 else 0, 8])
    print(f'Using {nw} dataloader workers every process')

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=nw)
    validate_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=nw)

    # Xác định số lượng classes - linh hoạt hơn, cố gắng lấy từ dataset
    num_classes = len(train_dataset.classes) if hasattr(train_dataset, 'classes') else len(train_dataset.label_names)
    if not hasattr(train_dataset, 'classes') and not hasattr(train_dataset, 'label_names'):
        print("Cảnh báo: Không tìm thấy thông tin class trong dataset. Cần kiểm tra MedMNIST document.")
        num_classes = len(torch.unique(torch.tensor(train_dataset.labels)))
        print(f"Đoán số classes từ labels: {num_classes}. Hãy kiểm tra lại.")
    print(f"Number of classes: {num_classes}") # In số lượng classes

    net = medmamba(num_classes=num_classes).to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=args.lr)

    best_acc = 0.0
    save_path = f'./{args.model_name}_{dataset_name.upper()}.pth' # Thêm tên dataset vào tên file lưu
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
