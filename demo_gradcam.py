import os
import argparse
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
import json
import cv2 # Import OpenCV để làm mịn

# Giả sử các import này hoạt động đúng trong môi trường của bạn
try:
    from MedMamba import VSSM as medmamba
    from grad_cam.utils import GradCAM, show_cam_on_image
except ImportError as e:
    # Gợi ý cách khắc phục nếu streamlit chạy từ thư mục khác
    # import sys
    # sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) # Thêm thư mục cha vào path
    # from MedMamba import VSSM as medmamba
    # from grad_cam.utils import GradCAM, show_cam_on_image
    print(f"Lỗi import: {e}. Hãy đảm bảo MedMamba.py và thư mục grad_cam/utils.py ở đúng vị trí.")
    exit()


class MedMambaReshapeTransform:
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4:
            if x.shape[1] == x.shape[2] and x.shape[1] > x.shape[3]: # (B, H, W, C)
                return x.permute(0, 3, 1, 2)
        return x


def main_gradcam_demo():
    parser = argparse.ArgumentParser(description='MedMamba Grad-CAM Demo')
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help='Path to the MedMamba model checkpoint (.pth).')
    parser.add_argument('--image_path', type=str, required=True,
                        help='Path to the input image.')
    parser.add_argument('--class_indices_path', type=str, default=None,
                        help='Path to class_indices.json.')
    parser.add_argument('--target_layer_type', type=str, default='ss2d_out_norm',
                        choices=['ss2d_out_norm', 'conv_branch_last_conv', 'block_input_norm'],
                        help='Type of target layer for Grad-CAM.')
    parser.add_argument('--target_category', type=int, default=None,
                        help='Target category index. If None, uses predicted class.')
    parser.add_argument('--img_size', type=int, default=224,
                        help='Image size for model input.')
    parser.add_argument('--smooth_cam', action='store_true',
                        help='Apply Gaussian blur to smooth the CAM heatmap.')
    parser.add_argument('--gaussian_ksize', type=int, default=7,
                        help='Kernel size for Gaussian blur (odd number).')


    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    class_indices = None
    if args.class_indices_path and os.path.exists(args.class_indices_path):
        try:
            with open(args.class_indices_path, 'r') as f:
                class_indices = json.load(f)
            print(f"Loaded class_indices from: {args.class_indices_path}")
        except Exception as e:
            print(f"Could not load class_indices from {args.class_indices_path}: {e}")

    print(f"Loading checkpoint: {args.checkpoint_path}")
    if not os.path.exists(args.checkpoint_path):
        print(f"Error: Checkpoint file not found at {args.checkpoint_path}")
        return
    
    # Sử dụng weights_only=True nếu bạn chắc chắn về nguồn gốc checkpoint để tăng bảo mật
    # Tuy nhiên, một số checkpoint cũ hoặc tùy chỉnh có thể yêu cầu weights_only=False
    try:
        checkpoint = torch.load(args.checkpoint_path, map_location=device, weights_only=False)
    except Exception as e:
        print(f"Pytorch Security Warning or Error loading checkpoint: {e}")
        print("Trying with weights_only=True...")
        try:
            checkpoint = torch.load(args.checkpoint_path, map_location=device, weights_only=True)
        except Exception as e_true:
            print(f"Failed to load checkpoint with weights_only=True as well: {e_true}")
            return


    num_classes = checkpoint.get('num_classes')
    if num_classes is None:
        print("Error: Checkpoint must contain 'num_classes'.")
        return

    if class_indices is None and 'class_indices' in checkpoint:
        class_indices = checkpoint['class_indices']
        print("Loaded class_indices from checkpoint.")
        if isinstance(class_indices, dict) and all(isinstance(k, int) for k in class_indices.keys()):
             class_indices = {str(k): v for k,v in class_indices.items()}

    model = medmamba(num_classes=num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model.to(device)
    print(f"Model loaded successfully. Number of classes: {num_classes}")

    try:
        if args.target_layer_type == 'ss2d_out_norm':
            target_layer = model.layers[-1].blocks[-1].self_attention.out_norm
            layer_name = "model.layers[-1].blocks[-1].self_attention.out_norm"
        elif args.target_layer_type == 'conv_branch_last_conv':
            target_layer = model.layers[-1].blocks[-1].conv33conv33conv11[-2]
            layer_name = "model.layers[-1].blocks[-1].conv33conv33conv11[-2]"
        elif args.target_layer_type == 'block_input_norm':
            target_layer = model.layers[-1].blocks[-1].ln_1
            layer_name = "model.layers[-1].blocks[-1].ln_1"
        else:
            raise ValueError(f"Unknown target_layer_type: {args.target_layer_type}")
        target_layers = [target_layer]
        print(f"Using target layer: {layer_name}")
    except AttributeError as e:
        print(f"Error accessing target layer: {e}. Please check model structure and target_layer_type.")
        return

    img_size = args.img_size
    data_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    try:
        original_pil_img = Image.open(args.image_path).convert('RGB')
    except FileNotFoundError:
        print(f"Error: Image file not found at {args.image_path}")
        return
    except Exception as e:
        print(f"Error opening image {args.image_path}: {e}")
        return

    img_tensor_transformed = data_transform(original_pil_img)
    input_tensor = torch.unsqueeze(img_tensor_transformed, dim=0).to(device)

    img_for_display_unnormalized = img_tensor_transformed.cpu().clone()
    mean = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
    std = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
    img_for_display_unnormalized = img_for_display_unnormalized * std + mean
    img_for_display_unnormalized = torch.clamp(img_for_display_unnormalized, 0, 1)
    img_for_display = img_for_display_unnormalized.permute(1, 2, 0).numpy()

    predicted_class_name = "Unknown"
    target_gradcam_class_name = "Unknown"
    predicted_idx = -1
    predicted_confidence = 0.0

    with torch.no_grad():
        output_logits = model(input_tensor)
        probabilities = torch.softmax(output_logits, dim=1)
        predicted_prob_tensor, predicted_idx_tensor = torch.max(probabilities, dim=1)
        predicted_idx = predicted_idx_tensor.item()
        predicted_confidence = predicted_prob_tensor.item()

    target_category_for_gradcam = args.target_category if args.target_category is not None else predicted_idx

    if class_indices:
        _class_indices_str_keys = {str(k): v for k, v in class_indices.items()}
        predicted_class_name = _class_indices_str_keys.get(str(predicted_idx), str(predicted_idx))
        target_gradcam_class_name = _class_indices_str_keys.get(str(target_category_for_gradcam), str(target_category_for_gradcam))
    else:
        predicted_class_name = str(predicted_idx)
        target_gradcam_class_name = str(target_category_for_gradcam)

    print(f"Predicted class: {predicted_class_name} (Index: {predicted_idx}) with confidence: {predicted_confidence:.4f}")
    print(f"Generating Grad-CAM for class: {target_gradcam_class_name} (Index: {target_category_for_gradcam})")

    reshape_transform = MedMambaReshapeTransform()
    cam_algorithm = GradCAM(model=model,
                            target_layers=target_layers,
                            use_cuda=torch.cuda.is_available(),
                            reshape_transform=reshape_transform)

    grayscale_cam = cam_algorithm(input_tensor=input_tensor, target_category=target_category_for_gradcam)
    if grayscale_cam is None:
        print("Error: Grad-CAM did not produce an output.")
        return
    grayscale_cam = grayscale_cam[0, :]

    # ÁP DỤNG LÀM MỊN NẾU CÓ FLAG
    if args.smooth_cam:
        ksize = args.gaussian_ksize
        if ksize % 2 == 0: # Kernel size phải là số lẻ
            ksize +=1
        grayscale_cam = cv2.GaussianBlur(grayscale_cam, (ksize, ksize), 0)
        print(f"Applied Gaussian Blur with kernel size ({ksize},{ksize})")


    cam_image = show_cam_on_image(img_for_display, grayscale_cam, use_rgb=True)

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle(f"Grad-CAM Demo for MedMamba\nImage: {os.path.basename(args.image_path)}", fontsize=14)

    axs[0].imshow(img_for_display)
    axs[0].set_title(f"Original Image\nPredicted: {predicted_class_name} ({predicted_confidence:.2f})")
    axs[0].axis('off')

    axs[1].imshow(cam_image)
    title_smooth = " (Smoothed)" if args.smooth_cam else ""
    axs[1].set_title(f"Grad-CAM for: {target_gradcam_class_name}{title_smooth}")
    axs[1].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    output_filename = "gradcam_result_updated.png"
    try:
        plt.savefig(output_filename)
        print(f"Grad-CAM result saved to: {os.path.abspath(output_filename)}")
    except Exception as e:
        print(f"Error saving figure: {e}")
    
    # plt.show() # Bỏ comment nếu bạn muốn thử hiển thị trực tiếp (có thể không hoạt động trên Kaggle terminal)
    plt.close(fig)


if __name__ == '__main__':
    main_gradcam_demo()