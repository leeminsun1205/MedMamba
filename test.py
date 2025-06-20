import os
import sys
import random
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm # Sử dụng tqdm.notebook cho giao diện đẹp hơn trong Kaggle

# ==============================================================================
# --- CÁC THAM SỐ CẦN THAY ĐỔI ---
# ==============================================================================
# Thay đổi các đường dẫn và giá trị này cho phù hợp với notebook của bạn
CHECKPOINT_PATH = "/kaggle/input/your-dataset/your_model.pth"
TEST_DIR = "/kaggle/input/your-dataset/test_data"
# Thư mục output sẽ nằm trong /kaggle/working/ để bạn có thể thấy và tải về
OUTPUT_DIR = "/kaggle/working/gradcam_results"
NUM_IMAGES = 5
NUM_CLASSES = 3  # Thay bằng số lớp của bạn
MEDMB_SIZE = 'T' # Chọn 'T', 'S', 'B', hoặc 'Te' cho phù hợp với model

# ==============================================================================
# --- IMPORT CÁC FILE MÃ NGUỒN PHỤ TRỢ ---
# ==============================================================================
# Nếu các file .py của bạn (MedMamba.py, grad_cam) nằm trong một thư mục dataset,
# bạn cần thêm đường dẫn đó vào sys.path để Python có thể tìm thấy.
# Ví dụ:
# sys.path.append('/kaggle/input/my-python-scripts')

# Hoặc copy trực tiếp file vào thư mục làm việc hiện tại
# !cp /kaggle/input/my-python-scripts/MedMamba.py .
# !cp -r /kaggle/input/my-python-scripts/grad_cam .

try:
    from MedMamba import VSSM as medmamba
    from grad_cam.utils import GradCAM, show_cam_on_image
except ImportError as e:
    print(f"Lỗi import: {e}")
    print("CHÚ Ý: Hãy chắc chắn rằng bạn đã thêm đường dẫn đến các file .py cần thiết bằng sys.path.append() hoặc copy chúng vào thư mục làm việc.")
    # Dừng thực thi nếu không import được
    raise e

# ==============================================================================
# --- CÁC LỚP VÀ HÀM HỖ TRỢ ---
# ==============================================================================
class MedMambaReshapeTransform:
    """
    Xử lý đúng định dạng tensor cho MedMamba, chuyển từ (B, H, W, C) sang (B, C, H, W).
    """
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4 and x.shape[1] == x.shape[2]:
            return x.permute(0, 3, 1, 2)
        return x

# ==============================================================================
# --- PHẦN CODE CHÍNH ĐỂ THỰC THI ---
# ==============================================================================

# Tạo thư mục output nếu chưa có
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Thư mục output: {OUTPUT_DIR}")

# Chọn thiết bị (GPU nếu có)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Sử dụng thiết bị: {device}")

# 1. Tải mô hình
print("Bắt đầu khởi tạo mô hình...")
if MEDMB_SIZE == 'T':
    net = medmamba(depths=[2, 2, 4, 2], dims=[96,192,384,768], num_classes=NUM_CLASSES)
elif MEDMB_SIZE == 'S':
    net = medmamba(depths=[2, 2, 8, 2], dims=[96,192,384,768], num_classes=NUM_CLASSES)
elif MEDMB_SIZE == 'B':
    net = medmamba(depths=[2, 2, 12, 2], dims=[128,256,512,1024], num_classes=NUM_CLASSES)
elif MEDMB_SIZE == 'Te':
    net = medmamba(depths=[2, 3, 3, 2],dims=[96,192,384,768],num_classes=NUM_CLASSES)

net.to(device)
print(f"Đang tải checkpoint từ: {CHECKPOINT_PATH}")
checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
net.load_state_dict(checkpoint['model_state_dict'])
net.eval()
print("Tải mô hình thành công.")

# 2. Tìm và chọn ngẫu nhiên các ảnh từ thư mục test
all_image_paths = []
for root, _, files in os.walk(TEST_DIR):
    for file in files:
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            all_image_paths.append(os.path.join(root, file))

if not all_image_paths:
    print(f"Lỗi: Không tìm thấy ảnh nào trong thư mục: {TEST_DIR}")
else:
    num_to_sample = min(NUM_IMAGES, len(all_image_paths))
    selected_images = random.sample(all_image_paths, num_to_sample)
    print(f"Đã chọn ngẫu nhiên {len(selected_images)} ảnh để xử lý.")

    # 3. Chuẩn bị các bước tiền xử lý ảnh và Grad-CAM
    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Xác định lớp mục tiêu cho Grad-CAM (thường là lớp convolution cuối)
    target_layer = net.layers[-1].blocks[-1].conv33conv33conv11[-2]
    target_layers = [target_layer]

    reshape_transform = MedMambaReshapeTransform()
    cam_algorithm = GradCAM(model=net,
                            target_layers=target_layers,
                            use_cuda=torch.cuda.is_available(),
                            reshape_transform=reshape_transform)

    # 4. Vòng lặp xử lý và hiển thị từng ảnh
    print("\nBắt đầu tạo và hiển thị Grad-CAM...")
    for img_path in tqdm(selected_images, desc="Processing Images"):
        pil_image = Image.open(img_path).convert('RGB')
        input_tensor = data_transform(pil_image).unsqueeze(0).to(device)

        # Dự đoán để xác định lớp mục tiêu
        with torch.no_grad():
            outputs = net(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_prob, predicted_idx_tensor = torch.max(probabilities, 1)
            predicted_idx = predicted_idx_tensor.item()
            predicted_confidence = predicted_prob.item()

        # Tạo Grad-CAM
        grayscale_cam = cam_algorithm(input_tensor=input_tensor, target_category=predicted_idx)
        grayscale_cam = grayscale_cam[0, :]

        # Chuẩn bị ảnh để hiển thị (bỏ chuẩn hóa)
        img_for_display = np.array(pil_image.resize((224, 224))) / 255.0
        cam_image = show_cam_on_image(img_for_display, grayscale_cam, use_rgb=True)
        
        # Lấy nhãn thực tế từ tên thư mục
        ground_truth_class = os.path.basename(os.path.dirname(img_path))

        # 5. Vẽ và hiển thị kết quả bằng matplotlib
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        fig.suptitle(f"Ảnh: {os.path.basename(img_path)}\nLớp thực tế: {ground_truth_class}", fontsize=14)
        
        axs[0].imshow(img_for_display)
        axs[0].set_title("Ảnh Gốc")
        axs[0].axis('off')

        axs[1].imshow(cam_image)
        axs[1].set_title(f"Grad-CAM cho lớp dự đoán: {predicted_idx}\n(Confidence: {predicted_confidence:.2f})")
        axs[1].axis('off')

        plt.tight_layout(rect=[0, 0.03, 1, 0.93])
        
        # Lưu ảnh vào thư mục output (tùy chọn)
        output_filename = f"gradcam_{os.path.basename(img_path)}"
        save_path = os.path.join(OUTPUT_DIR, output_filename)
        plt.savefig(save_path)

        # Hiển thị đồ thị ngay trong notebook
        plt.show()

    print(f"\nHoàn tất! Đã hiển thị {len(selected_images)} ảnh và lưu chúng vào thư mục '{OUTPUT_DIR}'.")