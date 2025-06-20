# Chạy trong Kaggle Notebook: hiển thị đồ họa inline
%matplotlib inline

import os
import sys
import argparse
import random
import logging

import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from IPython.display import Image as IPyImage, display

# Import model và Grad-CAM utils
try:
    from MedMamba import VSSM as medmamba
    from grad_cam.utils import GradCAM, show_cam_on_image
except ImportError as e:
    print(f"Lỗi import: {e}")
    print("Vui lòng đảm bảo file MedMamba.py và thư mục 'grad_cam' ở đúng vị trí.")
    sys.exit(1)

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

class MedMambaReshapeTransform:
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # Chuyển (B, H, W, C) → (B, C, H, W)
        if x.ndim == 4 and x.shape[1] == x.shape[2]:
            return x.permute(0, 3, 1, 2)
        return x

def parse_args():
    parser = argparse.ArgumentParser(
        description='Visualize Grad-CAM for MedMamba on random images.')
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help='Path to the model checkpoint .pth file.')
    parser.add_argument('--test_dir', type=str, required=True,
                        help='Thư mục test dataset, có subfolder cho từng class.')
    parser.add_argument('--output_dir', type=str, default='gradcam_results',
                        help='Thư mục lưu Grad-CAM visualizations.')
    parser.add_argument('--num_images', type=int, default=5,
                        help='Số lượng ảnh ngẫu nhiên để xử lý.')
    parser.add_argument('--num_classes', type=int, required=True,
                        help='Số lớp đầu ra của model.')
    parser.add_argument('--medmb_size', type=str, default='T',
                        choices=['T', 'S', 'B', 'Te'],
                        help='Chọn kích cỡ MedMamba: T, S, B, hoặc Te.')
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    logging.info(f"Kết quả sẽ được lưu vào: {args.output_dir}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info(f"Sử dụng thiết bị: {device}")

    # 1. Khởi tạo và load checkpoint
    logging.info("Khởi tạo model...")
    if args.medmb_size == 'T':
        net = medmamba(depths=[2,2,4,2], dims=[96,192,384,768], num_classes=args.num_classes)
    elif args.medmb_size == 'S':
        net = medmamba(depths=[2,2,8,2], dims=[96,192,384,768], num_classes=args.num_classes)
    elif args.medmb_size == 'B':
        net = medmamba(depths=[2,2,12,2], dims=[128,256,512,1024], num_classes=args.num_classes)
    else:  # 'Te'
        net = medmamba(depths=[2,3,3,2], dims=[96,192,384,768], num_classes=args.num_classes)

    net.to(device)
    logging.info(f"Đang tải checkpoint từ {args.checkpoint_path}")
    ckpt = torch.load(args.checkpoint_path, map_location=device)
    net.load_state_dict(ckpt['model_state_dict'])
    net.eval()
    logging.info("Load model xong.")

    # 2. Tập hợp và chọn ngẫu nhiên ảnh
    all_image_paths = []
    for root, _, files in os.walk(args.test_dir):
        for fn in files:
            if fn.lower().endswith(('.png','.jpg','.jpeg','.bmp')):
                all_image_paths.append(os.path.join(root, fn))
    if not all_image_paths:
        logging.error("Không tìm thấy ảnh nào trong test_dir.")
        sys.exit(1)
    selected = random.sample(all_image_paths, min(args.num_images, len(all_image_paths)))
    logging.info(f"Chọn {len(selected)} ảnh ngẫu nhiên.")

    # 3. Thiết lập transform và Grad-CAM
    data_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])
    # Lấy target layer cuối cùng
    try:
        target_layer = net.layers[-1].blocks[-1].conv33conv33conv11[-2]
    except Exception as e:
        logging.error(f"Lỗi truy cập target layer: {e}")
        sys.exit(1)
    cam = GradCAM(model=net,
                  target_layers=[target_layer],
                  use_cuda=torch.cuda.is_available(),
                  reshape_transform=MedMambaReshapeTransform())

    # 4. Xử lý và hiển thị
    for img_path in tqdm(selected, desc="Grad-CAM"):
        pil = Image.open(img_path).convert('RGB')
        inp = data_transform(pil).unsqueeze(0).to(device)

        with torch.no_grad():
            out = net(inp)
            probs = torch.softmax(out, dim=1)
            conf, idx = torch.max(probs,1)
            idx, conf = idx.item(), conf.item()

        gray = cam(input_tensor=inp, target_category=idx)
        if gray is None:
            logging.warning(f"Không tạo được Grad-CAM cho {img_path}")
            continue
        gray = gray[0]

        img_disp = np.array(pil.resize((224,224))) / 255.0
        cam_img = show_cam_on_image(img_disp, gray, use_rgb=True)

        # Vẽ 2 cột: gốc + cam
        fig, axs = plt.subplots(1,2, figsize=(10,5))
        fig.suptitle(f"{os.path.basename(img_path)} — GT: {os.path.basename(os.path.dirname(img_path))}", fontsize=14)
        axs[0].imshow(img_disp); axs[0].set_title("Original"); axs[0].axis('off')
        axs[1].imshow(cam_img); axs[1].set_title(f"Pred: {idx} (Conf: {conf:.2f})"); axs[1].axis('off')
        plt.tight_layout(rect=[0,0.03,1,0.95])

        # Hiển thị inline trong notebook
        plt.show()

        # Lưu file
        out_fn = os.path.join(args.output_dir, f"gradcam_{os.path.basename(img_path)}")
        fig.savefig(out_fn)
        plt.close(fig)

        # (tuỳ chọn) hiển thị file đã lưu
        display(IPyImage(filename=out_fn))

    logging.info("Hoàn tất tất cả ảnh.")

if __name__ == '__main__':
    main()
