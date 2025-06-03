import os
import sys
import argparse
import logging

import torch
from torchvision import transforms, datasets as torchvision_datasets
from tqdm import tqdm

from MedMamba import VSSM as medmamba
import datasets as custom_datasets # Assumes datasets.py contains NpzDataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def parse_args_test():
    parser = argparse.ArgumentParser(description='Evaluate a Medmamba model checkpoint on a test dataset.')
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to the model checkpoint .pth file.')
    parser.add_argument('--test_dir', type=str, required=True, help='Path to the test dataset (folder or directory containing .npy files).')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for testing.')
    parser.add_argument('--num_classes', type=int, default=None, help='Number of output classes. If None, inferred from checkpoint.')
    parser.add_argument('--medmb_size', type=str, default='T', choices=['T', 'S', 'B', 'Te'], help='Choose medmb size: T, S, or B')
    return parser.parse_args()


def main_test():
    args = parse_args_test()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using {device} device for evaluation.")
    print(f"Using {device} device for evaluation.")

    # Define data transform for testing - similar to validation
    data_transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Check if checkpoint file exists
    if not os.path.isfile(args.checkpoint_path):
        logging.error(f"Error: Checkpoint file not found at {args.checkpoint_path}")
        print(f"Error: Checkpoint file not found at {args.checkpoint_path}")
        sys.exit(1)

    # Load the checkpoint
    logging.info(f"Loading checkpoint: {args.checkpoint_path}")
    print(f"Loading checkpoint: {args.checkpoint_path}")
    checkpoint = torch.load(args.checkpoint_path, map_location=device)

    # Determine number of classes from checkpoint if not specified by argument
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


    # Load test dataset
    test_is_npz = os.path.exists(os.path.join(args.test_dir, 'test_images.npy')) and \
                  os.path.exists(os.path.join(args.test_dir, 'test_labels.npy'))

    if test_is_npz:
        logging.info(f"Loading test data from NPZ files in: {args.test_dir}")
        print(f"Loading test data from NPZ files in: {args.test_dir}")
        # Assuming NpzDataset handles 'test' split
        test_dataset = custom_datasets.NpzDataset(root_dir=args.test_dir, split='test', transform=data_transform_test)
        # Optional: Check consistency if --num_classes was given and dataset has a different count
        if args.num_classes is None and test_dataset.get_num_classes() != num_classes:
             logging.warning(f"Warning: Inferred classes from checkpoint ({num_classes}) differs from NPZ dataset classes ({test_dataset.get_num_classes()}). Using checkpoint value.")
             print(f"Warning: Inferred classes from checkpoint ({num_classes}) differs from NPZ dataset classes ({test_dataset.get_num_classes()}). Using checkpoint value.")

    else:
        logging.info(f"Loading test data from ImageFolder: {args.test_dir}")
        print(f"Loading test data from ImageFolder: {args.test_dir}")
        test_dataset = torchvision_datasets.ImageFolder(root=args.test_dir, transform=data_transform_test)
        # Optional: Check consistency if --num_classes was given and dataset has a different count
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
    if args.medmb_size == 'T': net = medmamba(depths=[2, 2, 4, 2],dims=[96,192,384,768],num_classes=num_classes)
    elif args.medmb_size == 'S': net = medmamba(depths=[2, 2, 8, 2],dims=[96,192,384,768],num_classes=num_classes)
    elif args.medmb_size == 'B': net = medmamba(depths=[2, 2, 12, 2],dims=[128,256,512,1024],num_classes=num_classes)
    else: net = medmamba(depths=[2, 3, 3, 2],dims=[96,192,384,768],num_classes=num_classes)
    logging.info(f'Model size: "{args.medmb_size}"')
    print(f'Model size: "{args.medmb_size}"')
    net.to(device)
    net.load_state_dict(checkpoint['model_state_dict'])
    logging.info("Model state loaded successfully.")
    print("Model state loaded successfully.")


    # Evaluate the model
    net.eval()
    acc = 0.0
    with torch.no_grad():
        test_bar = tqdm(test_loader, file=sys.stdout, ncols=100, desc="Testing")
        for test_data in test_bar:
            test_images, test_labels = test_data
            test_images, test_labels = test_images.to(device), test_labels.to(device)
            outputs = net(test_images)
            predict_y = torch.max(outputs, dim=1)[1]
            acc += torch.eq(predict_y, test_labels).sum().item()

    test_accuracy = acc / test_num

    log_result_message = f'Finished Testing. Test Accuracy: {test_accuracy:.3f}'
    logging.info(log_result_message)
    print(log_result_message)


if __name__ == '__main__':
    main_test()