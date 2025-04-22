import os
import sys
import json
import argparse
import numpy as np
from PIL import Image
import builtins # Needed for print suppression

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from torchvision import transforms, datasets as torchvision_datasets
from tqdm import tqdm

from MedMamba import VSSM as medmamba
import datasets as custom_datasets


def setup_for_distributed(is_master):
    builtin_print = builtins.print
    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)
    builtins.print = print

def init_ddp(rank, world_size, backend='nccl'):
    os.environ['MASTER_ADDR'] = os.getenv('MASTER_ADDR', 'localhost')
    os.environ['MASTER_PORT'] = os.getenv('MASTER_PORT', '12355') # Example port, ensure it's free

    dist.init_process_group(backend=backend, init_method='env://', world_size=world_size, rank=rank)
    torch.cuda.set_device(rank)
    print(f"DDP Setup: Rank {rank}/{world_size}, Device cuda:{rank}")
    setup_for_distributed(rank == 0)

def cleanup():
    dist.destroy_process_group()

def parse_args():
    parser = argparse.ArgumentParser(description='Train a Medmamba model using DDP spawn.')
    parser.add_argument('--train_dir', type=str, required=True, help='Path to training dataset.')
    parser.add_argument('--val_dir', type=str, required=True, help='Path to validation dataset.')
    parser.add_argument('--num_classes', type=int, default=None, help='Number of output classes.')
    parser.add_argument('--model_name', type=str, default='Medmamba', help='Model name for saving.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size PER GPU.')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs.')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate.')
    return parser.parse_args()

# Main worker function to be spawned
def main_worker(rank, world_size, args):
    init_ddp(rank, world_size)
    device = torch.device(f'cuda:{rank}')

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
        print(f"Rank {rank}: Loading training data from NPZ files in: {args.train_dir}")
        train_dataset = custom_datasets.NpzDataset(root_dir=args.train_dir, split='train', transform=data_transform["train"])
        num_classes = train_dataset.get_num_classes()
        if rank == 0:
            cla_dict_np = train_dataset.get_class_to_idx()
            cla_dict = {k: int(v) for k, v in cla_dict_np.items()}
    else:
        print(f"Rank {rank}: Loading training data from ImageFolder: {args.train_dir}")
        train_dataset = torchvision_datasets.ImageFolder(root=args.train_dir, transform=data_transform["train"])
        num_classes = len(train_dataset.classes)
        if rank == 0:
            cla_dict = {val: key for key, val in train_dataset.class_to_idx.items()}

    if args.num_classes is not None:
         if train_is_npz and args.num_classes != num_classes and rank == 0:
              print(f"Warning: --num_classes ({args.num_classes}) overrides inferred classes ({num_classes}) from NPZ.")
         num_classes = args.num_classes
    elif not train_is_npz and args.num_classes is None:
         if rank == 0:
             print("Error: --num_classes must be specified when using ImageFolder.", force=True)
         cleanup()
         sys.exit(1)

    train_num = len(train_dataset)

    if rank == 0:
        with open('class_indices.json', 'w') as json_file:
            json.dump(cla_dict, json_file, indent=4)

    val_is_npz = os.path.exists(os.path.join(args.val_dir, 'val_images.npy')) and \
                 os.path.exists(os.path.join(args.val_dir, 'val_labels.npy'))

    if val_is_npz:
        print(f"Rank {rank}: Loading validation data from NPZ files in: {args.val_dir}")
        val_dataset = custom_datasets.NpzDataset(root_dir=args.val_dir, split='val', transform=data_transform["val"])
    else:
        print(f"Rank {rank}: Loading validation data from ImageFolder: {args.val_dir}")
        val_dataset = torchvision_datasets.ImageFolder(root=args.val_dir, transform=data_transform["val"])

    val_num = len(val_dataset)

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)

    nw = min([os.cpu_count() // world_size, args.batch_size if args.batch_size > 1 else 0, 8])
    print(f'Rank {rank}: Using {nw} dataloader workers.')

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=nw, pin_memory=True, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, sampler=val_sampler, num_workers=nw, pin_memory=True, shuffle=False)

    if rank == 0:
        print(f"Using {train_num} images for training, {val_num} images for validation (across all GPUs).")
        print(f"Effective batch size: {args.batch_size * world_size}")
        print(f"Number of classes: {num_classes}")

    net = medmamba(num_classes=num_classes)
    net.to(device)
    net = DDP(net, device_ids=[rank])

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=args.lr)

    best_acc = 0.0
    save_path = f'./{args.model_name}_best_ddp.pth'
    train_steps = len(train_loader)

    for epoch in range(args.epochs):
        net.train()
        train_sampler.set_epoch(epoch)
        running_loss = 0.0

        train_iterator = train_loader
        if rank == 0:
            train_iterator = tqdm(train_loader, file=sys.stdout, ncols=100, desc=f"train epoch[{epoch + 1}/{args.epochs}]")

        for step, data in enumerate(train_iterator):
            images, labels = data
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs = net(images)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if rank == 0 and isinstance(train_iterator, tqdm):
                 train_iterator.set_postfix(loss=f"{loss.item():.3f}")


        net.eval()
        acc = 0.0
        total_samples = 0

        val_iterator = val_loader
        if rank == 0:
            val_iterator = tqdm(val_loader, file=sys.stdout, ncols=100, desc=f"valid epoch[{epoch + 1}/{args.epochs}]")

        with torch.no_grad():
            for val_data in val_iterator:
                val_images, val_labels = val_data
                val_images, val_labels = val_images.to(device, non_blocking=True), val_labels.to(device, non_blocking=True)
                outputs = net(val_images)
                predict_y = torch.max(outputs, dim=1)[1]
                correct_predictions = torch.eq(predict_y, val_labels).sum()

                dist.all_reduce(correct_predictions, op=dist.ReduceOp.SUM)

                batch_sample_count = torch.tensor(val_labels.size(0)).to(device)
                dist.all_reduce(batch_sample_count, op=dist.ReduceOp.SUM)

                acc += correct_predictions.item()
                total_samples += batch_sample_count.item()

        val_accuracy = acc / total_samples if total_samples > 0 else 0.0
        avg_train_loss = running_loss / train_steps # Note: This loss average is local to each rank

        if rank == 0:
            # Gather average loss across ranks if needed, or just report rank 0's avg loss
            print(f'[epoch {epoch + 1}] rank0_train_loss: {avg_train_loss:.3f}  val_accuracy: {val_accuracy:.3f}')

            if val_accuracy > best_acc:
                best_acc = val_accuracy
                torch.save(net.module.state_dict(), save_path)
                print(f'New best model saved to {save_path} with accuracy: {best_acc:.3f}')

        dist.barrier()

    if rank == 0:
        print(f'Finished Training. Best validation accuracy: {best_acc:.3f}')

    cleanup()


if __name__ == '__main__':
    args = parse_args()
    world_size = torch.cuda.device_count()

    if world_size < 2:
        print("Warning: DDP Spawn expects at least 2 GPUs. Running in single GPU mode might require code adjustments or will be inefficient.", force=True)
        # Simple fallback or error - For now, let's prevent spawn if only 1 GPU found with DDP code.
        if world_size == 1:
             print("Only 1 GPU found. Consider running without DDP spawn or adapting the code for single GPU.", force=True)
        else:
             print("No GPUs found. Cannot run DDP.", force=True)
        sys.exit(1) # Exit if not enough GPUs for this DDP spawn setup

    print(f"Found {world_size} GPUs. Spawning DDP processes.")

    mp.spawn(main_worker,
             nprocs=world_size,
             args=(world_size, args))