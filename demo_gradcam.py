import os
import argparse
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
import json # Để đọc class_indices

# Đảm bảo các tệp này có thể được import.
# Nếu demo_gradcam.py nằm trong thư mục MedMamba, và MedMamba.py cũng ở đó,
# và thư mục grad_cam là một thư mục con của MedMamba:
from MedMamba import VSSM as medmamba
from grad_cam.utils import GradCAM, show_cam_on_image

class MedMambaReshapeTransform:
    """
    Reshapes the input tensor for GradCAM compatibility with MedMamba.
    MedMamba layers typically output (Batch, Height, Width, Channels).
    GradCAM (as implemented in utils.py) expects (Batch, Channels, Height, Width)
    for calculating channel weights.
    """
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4:
            # Heuristic: If H == W and H > C, it's likely (B, H, W, C)
            if x.shape[1] == x.shape[2] and x.shape[1] > x.shape[3]:
                return x.permute(0, 3, 1, 2)
        # Add other conditions if your target layer has different output shapes.
        # For example, if it's (B, L, C) where L = H*W and you need to reconstruct H, W.
        # For now, this handles the common (B,H,W,C) case from MedMamba's deeper layers.
        # If the tensor is already (B, C, H, W), permute(0,3,1,2) would mess it up,
        # so the condition is important.
        return x


def main_gradcam_demo():
    parser = argparse.ArgumentParser(description='MedMamba Grad-CAM Demo')
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help='Path to the MedMamba model checkpoint (.pth).')
    parser.add_argument('--image_path', type=str, required=True,
                        help='Path to the input image.')
    parser.add_argument('--class_indices_path', type=str, default=None,
                        help='Path to class_indices.json. If not provided, will try to find it in checkpoint or use raw index.')
    parser.add_argument('--target_layer_type', type=str, default='ss2d_out_norm',
                        choices=['ss2d_out_norm', 'conv_branch_last_conv', 'block_input_norm', 'final_layer_norm'],
                        help='Type of target layer to use for Grad-CAM. "final_layer_norm" refers to model.layers[-1].blocks[-1].norm_layer (if exists and appropriate) or similar final normalization before pooling.')
    parser.add_argument('--target_category', type=int, default=None,
                        help='Target category index for Grad-CAM. If None, uses the predicted class.')
    parser.add_argument('--img_size', type=int, default=224,
                        help='Image size for input to the model.')

    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Nạp class_indices (nếu có từ đường dẫn)
    class_indices = None
    if args.class_indices_path and os.path.exists(args.class_indices_path):
        try:
            with open(args.class_indices_path, 'r') as f:
                class_indices = json.load(f)
                print(f"Loaded class_indices from: {args.class_indices_path}")
        except Exception as e:
            print(f"Could not load class_indices from {args.class_indices_path}: {e}")


    # 2. Nạp mô hình
    print(f"Loading checkpoint: {args.checkpoint_path}")
    if not os.path.exists(args.checkpoint_path):
        print(f"Error: Checkpoint file not found at {args.checkpoint_path}")
        return
    checkpoint = torch.load(args.checkpoint_path, map_location=device)

    num_classes = checkpoint.get('num_classes')
    if num_classes is None:
        print("Error: Checkpoint must contain 'num_classes'.")
        return

    # Thử nạp class_indices từ checkpoint nếu chưa có
    if class_indices is None and 'class_indices' in checkpoint:
        class_indices = checkpoint['class_indices']
        print("Loaded class_indices from checkpoint.")
        # Đảm bảo key là string nếu nó được đọc từ JSON trong checkpoint
        if isinstance(class_indices, dict) and all(isinstance(k, int) for k in class_indices.keys()):
             class_indices = {str(k): v for k,v in class_indices.items()}


    model = medmamba(num_classes=num_classes) # Sử dụng VSSM đã import
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model.to(device)
    print(f"Model loaded successfully. Number of classes: {num_classes}")

    # 3. Chọn target_layers dựa trên args.target_layer_type
    # Truy cập an toàn vào các lớp, kiểm tra sự tồn tại nếu cần
    try:
        if args.target_layer_type == 'ss2d_out_norm':
            # Lớp out_norm của SS2D trong block cuối cùng của stage cuối cùng
            target_layer = model.layers[-1].blocks[-1].self_attention.out_norm
            layer_name = "model.layers[-1].blocks[-1].self_attention.out_norm"
        elif args.target_layer_type == 'conv_branch_last_conv':
            # Lớp Conv2d cuối cùng trong nhánh Conv của block cuối cùng (trước ReLU cuối)
            target_layer = model.layers[-1].blocks[-1].conv33conv33conv11[-2]
            layer_name = "model.layers[-1].blocks[-1].conv33conv33conv11[-2]"
        elif args.target_layer_type == 'block_input_norm':
            # Lớp ln_1 (LayerNorm) đầu vào của block cuối cùng
            target_layer = model.layers[-1].blocks[-1].ln_1
            layer_name = "model.layers[-1].blocks[-1].ln_1"
        elif args.target_layer_type == 'final_layer_norm':
            # Thử nghiệm với lớp norm cuối cùng trước avgpool trong MedMamba.py gốc (không có trong VSSM)
            # Nếu VSSM của bạn có 1 lớp norm tổng thể cuối cùng trước pooling, hãy dùng nó.
            # Ví dụ, nếu model.norm tồn tại: target_layer = model.norm
            # Hiện tại, VSSM trong MedMamba.py không có model.norm riêng biệt ở cuối cùng.
            # Ta có thể chọn lớp norm của block cuối cùng, layer cuối cùng.
            # Hoặc output của VSSLayer cuối cùng (model.layers[-1]) trước khi permute và avgpool
            # Output của model.layers[-1] là output của VSSLayer cuối cùng.
            # Tuy nhiên, GradCAM thường hoạt động tốt hơn với các lớp cụ thể như Norm hoặc Conv.
            # Hãy thử với LayerNorm cuối cùng của block cuối cùng trong layer cuối cùng, nếu nó tồn tại.
            # Ví dụ, ln_1 của SS_Conv_SSM cuối:
            target_layer = model.layers[-1].blocks[-1].ln_1 # Tạm thời chọn lại, bạn cần xác định lớp này
            layer_name = "Final normalization layer (e.g., model.layers[-1].blocks[-1].ln_1)"
            print(f"Warning: 'final_layer_norm' is ambiguous for current VSSM. Using {layer_name} as a proxy.")
        else:
            raise ValueError(f"Unknown target_layer_type: {args.target_layer_type}")
        target_layers = [target_layer]
        print(f"Using target layer: {layer_name}")
    except AttributeError as e:
        print(f"Error accessing target layer: {e}. Please check model structure and target_layer_type.")
        return


    # 4. Chuẩn bị ảnh
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

    # Ảnh để hiển thị (numpy array, scale [0,1], đã un-normalize)
    img_for_display_unnormalized = img_tensor_transformed.cpu().clone()
    mean = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
    std = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
    img_for_display_unnormalized = img_for_display_unnormalized * std + mean
    img_for_display_unnormalized = torch.clamp(img_for_display_unnormalized, 0, 1)
    img_for_display = img_for_display_unnormalized.permute(1, 2, 0).numpy()


    # 5. Dự đoán và lấy target_category
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

    # Lấy tên lớp nếu có class_indices
    if class_indices:
        # Xử lý class_indices có thể là {idx_str: name} hoặc {name: idx_int} hoặc {idx_int: name}
        # Cách an toàn nhất là thử tìm theo cả key dạng str và int cho predicted_idx
        # Và đảm bảo key trong class_indices cũng được chuẩn hóa (ví dụ, tất cả là str)
        # Nếu class_indices từ JSON (được tạo bởi train.py) thì key số sẽ là string.
        _class_indices_str_keys = {str(k): v for k, v in class_indices.items()}

        predicted_class_name = _class_indices_str_keys.get(str(predicted_idx), str(predicted_idx))
        target_gradcam_class_name = _class_indices_str_keys.get(str(target_category_for_gradcam), str(target_category_for_gradcam))

    else:
        predicted_class_name = str(predicted_idx)
        target_gradcam_class_name = str(target_category_for_gradcam)

    print(f"Predicted class: {predicted_class_name} (Index: {predicted_idx}) with confidence: {predicted_confidence:.4f}")
    print(f"Generating Grad-CAM for class: {target_gradcam_class_name} (Index: {target_category_for_gradcam})")


    # 6. Grad-CAM
    reshape_transform = MedMambaReshapeTransform()
    cam_algorithm = GradCAM(model=model,
                            target_layers=target_layers,
                            use_cuda=torch.cuda.is_available(),
                            reshape_transform=reshape_transform)

    grayscale_cam = cam_algorithm(input_tensor=input_tensor, target_category=target_category_for_gradcam)
    if grayscale_cam is None :
        print("Error: Grad-CAM did not produce an output. Check target layers and model structure.")
        return
    grayscale_cam = grayscale_cam[0, :] # Lấy heatmap

    # 7. Hiển thị
    cam_image = show_cam_on_image(img_for_display, grayscale_cam, use_rgb=True)

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle(f"Grad-CAM Demo for MedMamba\nImage: {os.path.basename(args.image_path)}", fontsize=14)

    axs[0].imshow(img_for_display)
    axs[0].set_title(f"Original Image\nPredicted: {predicted_class_name} ({predicted_confidence:.2f})")
    axs[0].axis('off')

    axs[1].imshow(cam_image)
    axs[1].set_title(f"Grad-CAM for: {target_gradcam_class_name}")
    axs[1].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout to make space for suptitle
    plt.show()

if __name__ == '__main__':
    main_gradcam_demo()