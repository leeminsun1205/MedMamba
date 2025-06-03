import os
import sys
import argparse
import logging
import numpy as np # Thêm import numpy
import torch
from torchvision import transforms, datasets as torchvision_datasets
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix # Thêm import từ sklearn

from MedMamba import VSSM as medmamba
import datasets as custom_datasets # Assumes datasets.py contains NpzDataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def parse_args_test():
    parser = argparse.ArgumentParser(description='Evaluate a Medmamba model checkpoint on a test dataset.')
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to the model checkpoint .pth file.')
    parser.add_argument('--test_dir', type=str, required=True, help='Path to the test dataset (folder or directory containing .npy files).')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for testing.')
    parser.add_argument('--num_classes', type=int, default=None, help='Number of output classes. If None, inferred from checkpoint.')
    # Sửa lại lựa chọn cho medmb_size để khớp với logic khởi tạo model
    parser.add_argument('--medmb_size', type=str, default='T', choices=['T', 'S', 'B', 'Te'], help='Choose medmb size: T, S, B, or Te (custom).')
    return parser.parse_args()


def main_test():
    args = parse_args_test()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using {device} device for evaluation.")
    print(f"Using {device} device for evaluation.")

    data_transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    if not os.path.isfile(args.checkpoint_path):
        logging.error(f"Error: Checkpoint file not found at {args.checkpoint_path}")
        print(f"Error: Checkpoint file not found at {args.checkpoint_path}")
        sys.exit(1)

    logging.info(f"Loading checkpoint: {args.checkpoint_path}")
    print(f"Loading checkpoint: {args.checkpoint_path}")
    checkpoint = torch.load(args.checkpoint_path, map_location=device)

    num_classes = args.num_classes
    if num_classes is None:
        if 'num_classes' in checkpoint:
            num_classes = checkpoint['num_classes']
            logging.info(f"Inferred number of classes from checkpoint: {num_classes}")
            print(f"Inferred number of classes from checkpoint: {num_classes}")
        else:
            logging.error("Error: Could not determine number of classes from checkpoint and --num_classes not specified.")
            print("Error: Could not determine number of classes from checkpoint and --num_classes not specified.")
            sys.exit(1)
    else:
         logging.info(f"Using number of classes specified by argument: {num_classes}")
         print(f"Using number of classes specified by argument: {num_classes}")

    test_is_npz = os.path.exists(os.path.join(args.test_dir, 'test_images.npy')) and \
                  os.path.exists(os.path.join(args.test_dir, 'test_labels.npy'))

    if test_is_npz:
        logging.info(f"Loading test data from NPZ files in: {args.test_dir}")
        print(f"Loading test data from NPZ files in: {args.test_dir}")
        test_dataset = custom_datasets.NpzDataset(root_dir=args.test_dir, split='test', transform=data_transform_test)
        if args.num_classes is None and test_dataset.get_num_classes() != num_classes:
             logging.warning(f"Warning: Inferred classes from checkpoint ({num_classes}) differs from NPZ dataset classes ({test_dataset.get_num_classes()}). Using checkpoint value.")
             print(f"Warning: Inferred classes from checkpoint ({num_classes}) differs from NPZ dataset classes ({test_dataset.get_num_classes()}). Using checkpoint value.")
    else:
        logging.info(f"Loading test data from ImageFolder: {args.test_dir}")
        print(f"Loading test data from ImageFolder: {args.test_dir}")
        test_dataset = torchvision_datasets.ImageFolder(root=args.test_dir, transform=data_transform_test)
        if args.num_classes is None and len(test_dataset.classes) != num_classes:
            logging.warning(f"Warning: Inferred classes from checkpoint ({num_classes}) differs from ImageFolder dataset classes ({len(test_dataset.classes)}). Using checkpoint value.")
            print(f"Warning: Inferred classes from checkpoint ({num_classes}) differs from ImageFolder dataset classes ({len(test_dataset.classes)}). Using checkpoint value.")
        elif args.num_classes is not None and len(test_dataset.classes) != num_classes:
             logging.warning(f"Warning: --num_classes ({num_classes}) differs from ImageFolder dataset classes ({len(test_dataset.classes)}). Using --num_classes value.")
             print(f"Warning: --num_classes ({num_classes}) differs from ImageFolder dataset classes ({len(test_dataset.classes)}). Using --num_classes value.")

    test_num = len(test_dataset)
    nw = min([os.cpu_count(), args.batch_size if args.batch_size > 1 else 0, 8])
    logging.info(f'Using {nw} dataloader workers every process')
    print(f'Using {nw} dataloader workers every process')

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=nw, pin_memory=True)

    logging.info(f"Using {test_num} images for testing.")
    print(f"Using {test_num} images for testing.")

    # Initialize model and load state dict
    # Cập nhật logic khởi tạo model để xử lý 'Te' nếu bạn có cấu hình riêng cho nó
    if args.medmb_size == 'T':
        net = medmamba(depths=[2, 2, 4, 2], dims=[96,192,384,768], num_classes=num_classes)
    elif args.medmb_size == 'S':
        net = medmamba(depths=[2, 2, 8, 2], dims=[96,192,384,768], num_classes=num_classes)
    elif args.medmb_size == 'B':
        net = medmamba(depths=[2, 2, 12, 2], dims=[128,256,512,1024], num_classes=num_classes)
    elif args.medmb_size == 'Te': # Giả sử 'Te' là một cấu hình tùy chỉnh bạn đã thử
        # Ví dụ cấu hình tùy chỉnh, bạn cần thay đổi cho phù hợp với thử nghiệm của mình
        net = medmamba(depths=[2, 3, 3, 2],dims=[96,192,384,768],num_classes=num_classes)
        # Hoặc nếu 'Te' chỉ là để dùng cấu hình mặc định của 'else' trước đó:
        # net = medmamba(depths=[2, 3, 3, 2],dims=[96,192,384,768],num_classes=num_classes)
    else: # Xử lý trường hợp không mong muốn, hoặc bạn có thể bỏ qua nếu choices đã bao gồm hết
        logging.error(f"Unknown medmb_size: {args.medmb_size}. Using default Tiny structure.")
        net = medmamba(depths=[2, 2, 4, 2],dims=[96,192,384,768],num_classes=num_classes)


    logging.info(f'Model size: "{args.medmb_size}"')
    print(f'Model size: "{args.medmb_size}"')
    net.to(device)
    net.load_state_dict(checkpoint['model_state_dict'])
    logging.info("Model state loaded successfully.")
    print("Model state loaded successfully.")

    # Evaluate the model
    net.eval()
    
    all_labels = []
    all_predictions = []
    all_probabilities = [] # Để tính AUC

    with torch.no_grad():
        test_bar = tqdm(test_loader, file=sys.stdout, ncols=100, desc="Testing")
        for test_data in test_bar:
            test_images, test_labels = test_data
            test_images, test_labels = test_images.to(device), test_labels.to(device)
            
            outputs = net(test_images) # Đây là logits
            probabilities = torch.softmax(outputs, dim=1) # Chuyển logits thành xác suất
            _, predict_y = torch.max(outputs, dim=1) # Lấy nhãn dự đoán

            all_labels.extend(test_labels.cpu().numpy())
            all_predictions.extend(predict_y.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())

    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)
    all_probabilities = np.array(all_probabilities)

    # Tính toán các chỉ số
    overall_accuracy = accuracy_score(all_labels, all_predictions)
    
    # Đối với multi-class, precision, recall, f1 thường dùng 'macro' hoặc 'weighted' average
    # 'macro': tính cho từng lớp rồi lấy trung bình không trọng số
    # 'weighted': tính cho từng lớp rồi lấy trung bình có trọng số (theo số lượng mẫu mỗi lớp)
    # zero_division=0: để tránh warning nếu một lớp không có dự đoán nào (Precision/Recall có thể = 0)
    precision_macro = precision_score(all_labels, all_predictions, average='macro', zero_division=0)
    sensitivity_macro = recall_score(all_labels, all_predictions, average='macro', zero_division=0) # Sensitivity là Recall
    f1_macro = f1_score(all_labels, all_predictions, average='macro', zero_division=0)

    # Tính Specificity
    cm = confusion_matrix(all_labels, all_predictions)
    specificity_per_class = []
    for i in range(num_classes):
        tn = 0
        for j in range(num_classes):
            for k in range(num_classes):
                if i != j and i != k: # Bỏ qua hàng và cột của lớp hiện tại
                     # Sửa lỗi logic: TN là tất cả các trường hợp không phải lớp i và cũng không bị dự đoán là lớp i
                     # TN_i = sum of cm[row, col] where row != i and col != i
                     pass # Logic tính TN phức tạp hơn khi duyệt cm, cách dễ hơn là dùng công thức
        
        # Cách tính TN, FP cho từng lớp i:
        tp_i = cm[i, i]
        fp_i = cm[:, i].sum() - tp_i # Tổng cột i (dự đoán là i) trừ đi TP
        fn_i = cm[i, :].sum() - tp_i # Tổng hàng i (thực tế là i) trừ đi TP
        tn_i = cm.sum() - (tp_i + fp_i + fn_i) # Tổng số mẫu trừ đi TP, FP, FN của lớp i
        
        class_specificity = tn_i / (tn_i + fp_i) if (tn_i + fp_i) > 0 else 0.0
        specificity_per_class.append(class_specificity)
    
    specificity_macro = np.mean(specificity_per_class) if len(specificity_per_class) > 0 else 0.0
    
    # Tính AUC
    # Đối với multi-class, roc_auc_score cần xác suất của các lớp
    # Sử dụng One-vs-Rest (ovr) hoặc One-vs-One (ovo) strategy
    if num_classes == 2: # Trường hợp nhị phân đơn giản hơn
        # all_probabilities[:, 1] lấy xác suất của lớp dương (thường là lớp 1)
        auc_score = roc_auc_score(all_labels, all_probabilities[:, 1]) if all_probabilities.shape[1] > 1 else "N/A (single class probabilities)"
    elif num_classes > 2 :
        try:
            auc_score = roc_auc_score(all_labels, all_probabilities, multi_class='ovr', average='macro')
        except ValueError as e:
            auc_score = f"N/A ({e})" # Xử lý trường hợp không tính được AUC (ví dụ: chỉ có 1 lớp trong y_true)
            logging.warning(f"Could not calculate AUC: {e}")
            print(f"Could not calculate AUC: {e}")
    else: # num_classes < 2
        auc_score = "N/A (num_classes < 2)"


    print("\n--- Evaluation Metrics ---")
    print(f"Overall Accuracy: {overall_accuracy*100:.2f}")
    print(f"Precision (Macro): {precision_macro*100:.2f}")
    print(f"Sensitivity/Recall (Macro): {sensitivity_macro*100:.2f}")
    print(f"Specificity (Macro): {specificity_macro*100:.2f}")
    print(f"F1-score (Macro): {f1_macro*100:.2f}")
    print(f"AUC (Macro OvR): {auc_score if isinstance(auc_score, str) else auc_score:.3f}")
    print("-------------------------\n")

    # In ma trận nhầm lẫn để tham khảo
    print("Confusion Matrix:\n", cm)


if __name__ == '__main__':
    main_test()