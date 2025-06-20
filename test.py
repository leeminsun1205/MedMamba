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

# Import model và các thành phần Grad-CAM
# Đảm bảo MedMamba.py và thư mục grad_cam nằm ở vị trí có thể truy cập
try:
    from MedMamba import VSSM as medmamba
    from grad_cam.utils import GradCAM, show_cam_on_image
except ImportError as e:
    print(f"Lỗi import: {e}")
    print("Vui lòng đảm bảo file MedMamba.py và thư mục thư viện 'grad_cam' ở đúng vị trí.")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Lớp này rất quan trọng, được lấy từ file demo của bạn
# để xử lý đúng định dạng tensor cho MedMamba
class MedMambaReshapeTransform:
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4 and x.shape[1] == x.shape[2]: # (B, H, W, C)
             # Chuyển thành (B, C, H, W) mà các lớp CNN mong đợi
            return x.permute(0, 3, 1, 2)
        return x

def parse_args():
    parser = argparse.ArgumentParser(description='Visualize Grad-CAM for MedMamba model on random images.')
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to the model checkpoint .pth file.')
    parser.add_argument('--test_dir', type=str, required=True, help='Path to the test dataset directory with class subfolders.')
    parser.add_argument('--output_dir', type=str, default='gradcam_results', help='Directory to save the Grad-CAM visualizations.')
    parser.add_argument('--num_images', type=int, default=5, help='Number of random images to process.')
    parser.add_argument('--num_classes', type=int, required=True, help='Number of output classes for the model.')
    parser.add_argument('--medmb_size', type=str, default='T', choices=['T', 'S', 'B', 'Te'], help='Choose medmb size: T, S, B, or Te.')
    return parser.parse_args()

def main():
    args = parse_args()

    # Tạo thư mục output nếu chưa có
    os.makedirs(args.output_dir, exist_ok=True)
    logging.info(f"Kết quả sẽ được lưu vào thư mục: {args.output_dir}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info(f"Sử dụng thiết bị: {device}")

    # 1. Tải mô hình
    logging.info("Khởi tạo mô hình...")
    if args.medmb_size == 'T':
        net = medmamba(depths=[2, 2, 4, 2], dims=[96,192,384,768], num_classes=args.num_classes)
    elif args.medmb_size == 'S':
        net = medmamba(depths=[2, 2, 8, 2], dims=[96,192,384,768], num_classes=args.num_classes)
    elif args.medmb_size == 'B':
        net = medmamba(depths=[2, 2, 12, 2], dims=[128,256,512,1024], num_classes=args.num_classes)
    elif args.medmb_size == 'Te':
        net = medmamba(depths=[2, 3, 3, 2],dims=[96,192,384,768],num_classes=args.num_classes)

    net.to(device)

    logging.info(f"Đang tải checkpoint từ: {args.checkpoint_path}")
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    net.load_state_dict(checkpoint['model_state_dict'])
    net.eval()
    logging.info("Tải mô hình thành công.")

    # 2. Tìm và chọn ngẫu nhiên các ảnh
    all_image_paths = []
    for root, _, files in os.walk(args.test_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                all_image_paths.append(os.path.join(root, file))

    if not all_image_paths:
        logging.error(f"Không tìm thấy ảnh nào trong thư mục: {args.test_dir}")
        sys.exit(1)

    num_to_sample = min(args.num_images, len(all_image_paths))
    selected_images = random.sample(all_image_paths, num_to_sample)
    logging.info(f"Đã chọn ngẫu nhiên {len(selected_images)} ảnh để xử lý.")

    # 3. Chuẩn bị các bước xử lý và Grad-CAM
    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Đây là lớp mục tiêu quan trọng, được lấy từ logic trong file demo của bạn
    # Điều này đảm bảo chúng ta đang xem xét đúng lớp convolution cuối cùng
    try:
        target_layer = net.layers[-1].blocks[-1].conv33conv33conv11[-2]
        target_layers = [target_layer]
    except (AttributeError, IndexError) as e:
        logging.error(f"Lỗi khi truy cập target layer: {e}")
        logging.error("Cấu trúc model có thể đã thay đổi hoặc không khớp với mong đợi.")
        sys.exit(1)

    reshape_transform = MedMambaReshapeTransform()
    cam_algorithm = GradCAM(model=net,
                            target_layers=target_layers,
                            use_cuda=torch.cuda.is_available(),
                            reshape_transform=reshape_transform)

    # 4. Vòng lặp xử lý từng ảnh
    for img_path in tqdm(selected_images, desc="Đang tạo Grad-CAM"):
        pil_image = Image.open(img_path).convert('RGB')
        
        # Chuẩn bị tensor đầu vào cho model
        input_tensor = data_transform(pil_image).unsqueeze(0).to(device)

        # Dự đoán lớp để biết nên tạo Grad-CAM cho lớp nào
        with torch.no_grad():
            outputs = net(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_prob, predicted_idx_tensor = torch.max(probabilities, 1)
            predicted_idx = predicted_idx_tensor.item()
            predicted_confidence = predicted_prob.item()

        # Tạo Grad-CAM cho lớp được dự đoán
        grayscale_cam = cam_algorithm(input_tensor=input_tensor,
                                      target_category=predicted_idx)
        
        if grayscale_cam is None:
            logging.warning(f"Không thể tạo Grad-CAM cho ảnh: {os.path.basename(img_path)}")
            continue

        grayscale_cam = grayscale_cam[0, :]

        # Chuẩn bị ảnh để hiển thị (un-normalize)
        img_for_display = np.array(pil_image.resize((224, 224))) / 255.0
        cam_image = show_cam_on_image(img_for_display, grayscale_cam, use_rgb=True)

        # Lấy nhãn thực tế từ tên thư mục cha
        ground_truth_class = os.path.basename(os.path.dirname(img_path))

        # 5. Vẽ và lưu kết quả
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        fig.suptitle(f"Ảnh: {os.path.basename(img_path)}\nLớp thực tế: {ground_truth_class}", fontsize=14)
        
        axs[0].imshow(img_for_display)
        axs[0].set_title("Ảnh Gốc")
        axs[0].axis('off')

        axs[1].imshow(cam_image)
        axs[1].set_title(f"Grad-CAM cho lớp dự đoán: {predicted_idx}\n(Confidence: {predicted_confidence:.2f})")
        axs[1].axis('off')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # Lưu file ảnh kết quả
        output_filename = f"gradcam_{os.path.basename(img_path)}"
        save_path = os.path.join(args.output_dir, output_filename)
        plt.savefig(save_path)
        plt.close(fig)

    logging.info(f"Hoàn tất! Đã lưu {len(selected_images)} ảnh vào thư mục '{args.output_dir}'.")

if __name__ == '__main__':
    main()