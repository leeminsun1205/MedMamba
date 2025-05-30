import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
import os
import random
import json
import cv2

from grad_cam.utils import GradCAM, show_cam_on_image

from MedMamba import VSSM as medmamba

# --- Lớp và Hàm cho Grad-CAM ---
class MedMambaReshapeTransform: #
    """
    Reshape transform cho MedMamba, điều chỉnh tensor nếu nó có dạng (B, H, W, C)
    thay vì (B, C, H, W) mà GradCAM mong đợi cho một số bước trung gian.
    """
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4: #
            if x.shape[1] == x.shape[2] and x.shape[1] > x.shape[3]: # (B, H, W, C)
                return x.permute(0, 3, 1, 2) # Chuyển thành (B, C, H, W)
        return x

def generate_gradcam_image(model, device, pil_image, target_category_for_gradcam, class_indices,
                           img_size=224, smooth_cam=False, gaussian_ksize=7):
    try:
        if not (hasattr(model, 'layers') and model.layers and
                len(model.layers) > 0 and hasattr(model.layers[-1], 'blocks') and
                model.layers[-1].blocks and len(model.layers[-1].blocks) > 0 and
                hasattr(model.layers[-1].blocks[-1], 'conv33conv33conv11') and
                model.layers[-1].blocks[-1].conv33conv33conv11 and
                len(model.layers[-1].blocks[-1].conv33conv33conv11) >= 2):
            st.error("Cấu trúc model không phù hợp hoặc không đủ sâu để lấy target_layer cho Grad-CAM (ví dụ: model.layers[-1].blocks[-1].conv33conv33conv11[-2]).")
            return None, None
        
        target_layer = model.layers[-1].blocks[-1].conv33conv33conv11[-2] #
        target_layers = [target_layer] #

        # 2. Chuẩn bị ảnh đầu vào
        data_transform_gradcam = transforms.Compose([ #
            transforms.Resize((img_size, img_size)), #
            transforms.ToTensor(), #
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) #
        ])
        img_tensor_transformed = data_transform_gradcam(pil_image.convert('RGB')) #
        input_tensor = torch.unsqueeze(img_tensor_transformed, dim=0).to(device) #

        # Chuẩn bị ảnh để hiển thị (unnormalized)
        img_for_display_unnormalized = img_tensor_transformed.cpu().clone() #
        mean = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1) #
        std = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1) #
        img_for_display_unnormalized = img_for_display_unnormalized * std + mean #
        img_for_display_unnormalized = torch.clamp(img_for_display_unnormalized, 0, 1) #
        img_for_display = img_for_display_unnormalized.permute(1, 2, 0).numpy() #

        # 3. Khởi tạo GradCAM
        reshape_transform = MedMambaReshapeTransform() #
        cam_algorithm = GradCAM(model=model, #
                                target_layers=target_layers, #
                                use_cuda=torch.cuda.is_available(), #
                                reshape_transform=reshape_transform) #

        # 4. Tính toán Grad-CAM
        # Đảm bảo model đang ở chế độ eval
        model.eval()
        grayscale_cam = cam_algorithm(input_tensor=input_tensor, target_category=target_category_for_gradcam) #
        
        if grayscale_cam is None: #
            st.error("Grad-CAM không tạo ra output. Điều này có thể xảy ra nếu target_layer không phù hợp hoặc gradient là zero.")
            return None, None
        grayscale_cam = grayscale_cam[0, :] #

        # 5. Làm mịn (nếu có) và phủ màu
        if smooth_cam: #
            ksize = gaussian_ksize #
            if ksize % 2 == 0: ksize +=1 # # Kernel size phải là số lẻ
            grayscale_cam = cv2.GaussianBlur(grayscale_cam, (ksize, ksize), 0) #

        cam_image_result = show_cam_on_image(img_for_display, grayscale_cam, use_rgb=True) #

        return img_for_display, cam_image_result

    except AttributeError as e:
        st.error(f"Lỗi thuộc tính khi tạo Grad-CAM: {e}. Điều này có thể do cấu trúc model không như mong đợi hoặc target_layer không tồn tại.")
        return None, None
    except Exception as e:
        st.error(f"Lỗi không xác định khi tạo Grad-CAM: {e}")
        return None, None

# --- Cấu hình và Hàm hỗ trợ (Giữ nguyên từ file gốc của bạn) ---
def get_transform(img_size=224): #
    """Trả về transform chuẩn cho ảnh đầu vào."""
    return transforms.Compose([ #
        transforms.Resize((img_size, img_size)), #
        transforms.ToTensor(), #
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) #
    ])

@st.cache_resource #
def load_medmamba_model(checkpoint_path, num_classes): #
    """Nạp mô hình MedMamba từ checkpoint."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") #
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False) #
        
        actual_num_classes = checkpoint.get('num_classes') #
        if actual_num_classes is None: #
            st.warning(f"Checkpoint không chứa 'num_classes'. Sử dụng giá trị đầu vào: {num_classes}.") #
        elif actual_num_classes != num_classes: #
            st.warning(f"Số lớp trong checkpoint ({actual_num_classes}) khác với số lớp nhập vào ({num_classes}). Sử dụng giá trị từ checkpoint: {actual_num_classes}.") #
            num_classes = actual_num_classes #
            
        model = medmamba(num_classes=num_classes) #
        model.load_state_dict(checkpoint['model_state_dict']) #
        model.eval() #
        model.to(device) #
        
        class_indices_from_ckpt = checkpoint.get('class_indices') #
        if class_indices_from_ckpt and isinstance(class_indices_from_ckpt, dict): #
            if all(isinstance(k, int) for k in class_indices_from_ckpt.keys()): #
                 class_indices_from_ckpt = {str(k): v for k,v in class_indices_from_ckpt.items()} #
        
        st.success(f"Mô hình được nạp thành công từ '{os.path.basename(checkpoint_path)}' trên {device}.") #
        return model, device, class_indices_from_ckpt, num_classes #
    except FileNotFoundError: #
        st.error(f"Không tìm thấy tệp checkpoint tại: {checkpoint_path}") #
    except KeyError as e: #
        st.error(f"Lỗi KeyError khi nạp checkpoint (có thể thiếu 'model_state_dict' hoặc 'num_classes'): {e}") #
    except Exception as e: #
        st.error(f"Lỗi khi nạp mô hình: {e}") #
    return None, None, None, num_classes #


@st.cache_data #
def load_class_indices_from_file(class_indices_path): #
    """Nạp class_indices từ tệp JSON được chỉ định."""
    if class_indices_path and os.path.exists(class_indices_path): #
        try:
            with open(class_indices_path, 'r') as f: #
                class_indices = json.load(f) #
            if isinstance(class_indices, dict) and any(isinstance(k, int) for k in class_indices.keys()): #
                 class_indices = {str(k): v for k,v in class_indices.items()} #
            st.info(f"Đã nạp class_indices từ tệp: {class_indices_path}") #
            return class_indices #
        except Exception as e: #
            st.warning(f"Không thể nạp class_indices từ '{class_indices_path}': {e}") #
    return None #

def predict(model, device, image_pil, transform, class_indices): #
    """Thực hiện dự đoán trên ảnh PIL đã cho."""
    img_tensor = transform(image_pil.convert('RGB')) #
    input_tensor = torch.unsqueeze(img_tensor, dim=0).to(device) #

    with torch.no_grad(): #
        output_logits = model(input_tensor) #
        probabilities = torch.softmax(output_logits, dim=1) #
        predicted_prob_tensor, predicted_idx_tensor = torch.max(probabilities, dim=1) #
        predicted_idx = predicted_idx_tensor.item() #
        predicted_confidence = predicted_prob_tensor.item() #

    predicted_class_name = str(predicted_idx) #
    if class_indices: #
        predicted_class_name = class_indices.get(str(predicted_idx), f"Lớp không xác định (Index: {predicted_idx})") #
    
    return predicted_class_name, predicted_confidence, predicted_idx #

# --- Giao diện Streamlit ---
def main_app(): #
    st.set_page_config(page_title="Demo MedMamba", layout="wide") #
    st.title("🐍 Demo Phân Loại Ảnh Y Tế với MedMamba") #

    st.sidebar.header("⚙️ Cấu Hình Mô Hình") #
    
    default_checkpoint_path = "YOUR_MODEL_CHECKPOINT.pth" #
    default_class_indices_path = "class_indices.json" #
    
    checkpoint_path_input = st.sidebar.text_input( #
        "Đường dẫn đến Checkpoint (.pth)", 
        value=default_checkpoint_path, #
        help="Cung cấp đường dẫn đầy đủ đến tệp checkpoint của mô hình MedMamba." #
    )
    class_indices_path_input = st.sidebar.text_input( #
        "Đường dẫn đến Class Indices (.json) (Tùy chọn)", 
        value=default_class_indices_path, #
        help="Tệp JSON chứa ánh xạ từ index sang tên lớp." #
    )
    num_classes_input = st.sidebar.number_input( #
        "Số Lượng Lớp (nếu không có trong checkpoint)", 
        min_value=1, value=3, step=1, #
        help="Số lớp đầu ra của mô hình. Sẽ được ghi đè nếu checkpoint chứa thông tin này." #
    )

    if 'model' not in st.session_state: #
        st.session_state.model = None #
        st.session_state.device = None #
        st.session_state.class_indices = None #
        st.session_state.num_classes_loaded = num_classes_input #
        st.session_state.model_loaded_path = "" #
        st.session_state.last_prediction_info = None # Khởi tạo cho Grad-CAM

    if st.sidebar.button("Nạp Mô Hình & Class Indices", key="load_model_button"): #
        st.session_state.model = None #
        st.session_state.last_prediction_info = None # Reset khi nạp model mới
        if checkpoint_path_input: #
            model, device, class_indices_from_ckpt, num_classes_final = load_medmamba_model(checkpoint_path_input, num_classes_input) #
            
            if model and device: #
                st.session_state.model = model #
                st.session_state.device = device #
                st.session_state.num_classes_loaded = num_classes_final #
                st.session_state.model_loaded_path = checkpoint_path_input #

                class_indices_from_file = load_class_indices_from_file(class_indices_path_input) #
                if class_indices_from_file: #
                    st.session_state.class_indices = class_indices_from_file #
                elif class_indices_from_ckpt: #
                    st.session_state.class_indices = class_indices_from_ckpt #
                    st.sidebar.info("Đã sử dụng class_indices từ checkpoint.") #
                else: #
                    st.session_state.class_indices = None #
                    st.sidebar.warning("Không tìm thấy class_indices. Dự đoán sẽ chỉ hiển thị index của lớp.") #
            else: #
                st.session_state.model = None #
                st.session_state.class_indices = None #
        else: #
            st.sidebar.error("Vui lòng cung cấp đường dẫn đến checkpoint.") #

    if st.session_state.model is None: #
        st.warning("Mô hình chưa được nạp. Vui lòng cấu hình và nhấn 'Nạp Mô Hình' trong thanh sidebar.") #
        st.stop() #

    st.success(f"Mô hình **{os.path.basename(st.session_state.model_loaded_path)}** đã được nạp và sẵn sàng!") #
    st.info(f"Số lớp của mô hình: **{st.session_state.num_classes_loaded}**") #
    if st.session_state.class_indices: #
        st.write("Các lớp được phát hiện:") #
        st.json(st.session_state.class_indices, expanded=False) #
    
    img_transform = get_transform() #

    st.markdown("---") #
    st.header("🔬 Chế Độ Dự Đoán") #
    prediction_mode = st.radio( #
        "Chọn chế độ dự đoán:", #
        ("Tải Ảnh Lên", "Dự Đoán Ảnh Ngẫu Nhiên Từ Thư Mục"), #
        key="prediction_mode_radio" #
    )

    image_pil_for_prediction = None

    if prediction_mode == "Tải Ảnh Lên": #
        uploaded_file = st.file_uploader( #
            "Chọn một hình ảnh...", 
            type=["png", "jpg", "jpeg", "bmp"], #
            key="file_uploader" #
        )
        if uploaded_file is not None: #
            try:
                image_pil_for_prediction = Image.open(uploaded_file).convert('RGB') #
                
                col1, col2 = st.columns([2,3]) #
                with col1: #
                    st.image(image_pil_for_prediction, caption="Ảnh Đã Tải Lên", use_column_width=True) #

                with col2: #
                    if st.button("Thực Hiện Dự Đoán", key="predict_uploaded_button"): #
                        with st.spinner("Đang dự đoán..."): #
                            class_name, confidence, class_idx = predict( #
                                st.session_state.model, #
                                st.session_state.device, #
                                image_pil_for_prediction, #
                                img_transform, #
                                st.session_state.class_indices #
                            )
                        st.session_state.last_prediction_info = { #
                            "image_pil": image_pil_for_prediction, #
                            "class_name": class_name, #
                            "confidence": confidence, #
                            "predicted_idx": class_idx #
                        }
                        st.subheader("Kết Quả Dự Đoán:") #
                        st.markdown(f"**Lớp Dự Đoán:** `{class_name}` (Index: {class_idx})") #
                        st.markdown(f"**Độ Tin Cậy:** `{confidence:.4f}`") #
            except Exception as e: #
                st.error(f"Lỗi xử lý ảnh tải lên: {e}") #

    elif prediction_mode == "Dự Đoán Ảnh Ngẫu Nhiên Từ Thư Mục": #
        default_test_dir = "PATH_TO_YOUR_TEST_IMAGE_FOLDER" #
        test_dir_input = st.text_input( #
            "Đường dẫn đến Thư Mục Ảnh Test", 
            value=default_test_dir, #
            help="Cung cấp đường dẫn đến thư mục chứa các ảnh để chọn ngẫu nhiên." #
        )

        if st.button("Dự Đoán Ảnh Ngẫu Nhiên", key="predict_random_button"): #
            if not os.path.isdir(test_dir_input): #
                st.error(f"Không tìm thấy thư mục: {test_dir_input}") #
            else: #
                image_files = [] #
                for root, _, files in os.walk(test_dir_input): #
                    for file in files: #
                        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')): #
                            image_files.append(os.path.join(root, file)) #
                
                if not image_files: #
                    st.warning(f"Không tìm thấy tệp ảnh nào trong thư mục: {test_dir_input}") #
                else: #
                    random_image_path = random.choice(image_files) #
                    try:
                        image_pil_for_prediction = Image.open(random_image_path).convert('RGB') #
                        
                        col1, col2 = st.columns([2,3]) #
                        with col1: #
                            st.image(image_pil_for_prediction, caption=f"Ảnh Ngẫu Nhiên: {os.path.basename(random_image_path)}", use_column_width=True) #
                        
                        with col2: #
                            with st.spinner("Đang dự đoán..."): #
                                class_name, confidence, class_idx = predict( #
                                    st.session_state.model, #
                                    st.session_state.device, #
                                    image_pil_for_prediction, #
                                    img_transform, #
                                    st.session_state.class_indices #
                                )
                            st.session_state.last_prediction_info = { #
                                "image_pil": image_pil_for_prediction, #
                                "class_name": class_name, #
                                "confidence": confidence, #
                                "predicted_idx": class_idx #
                            }
                            st.subheader("Kết Quả Dự Đoán:") #
                            st.markdown(f"**Ảnh Được Chọn:** `{os.path.basename(random_image_path)}`") #
                            st.markdown(f"**Lớp Dự Đoán:** `{class_name}` (Index: {class_idx})") #
                            st.markdown(f"**Độ Tin Cậy:** `{confidence:.4f}`") #
                    except Exception as e: #
                        st.error(f"Lỗi xử lý ảnh {random_image_path}: {e}") #
    
    # --- Phần Grad-CAM ---
    st.markdown("---")
    st.header("🔥 Grad-CAM Visualization (Tùy Chọn)")

    last_prediction_info = st.session_state.get('last_prediction_info', None)

    enable_gradcam = st.checkbox("Bật Grad-CAM", key="enable_gradcam_checkbox")

    if enable_gradcam:
        if not last_prediction_info or last_prediction_info.get("image_pil") is None:
            st.warning("Vui lòng thực hiện một dự đoán với ảnh hợp lệ trước khi tạo Grad-CAM.")
        else:
            pil_image_for_gradcam = last_prediction_info["image_pil"]
            predicted_idx_for_gradcam = last_prediction_info["predicted_idx"]
            
            predicted_class_name_display = last_prediction_info["class_name"]

            target_category_input_gradcam = st.text_input(
                f"Index Lớp Mục Tiêu cho Grad-CAM (mặc định: lớp được dự đoán - '{predicted_class_name_display}' (Index: {predicted_idx_for_gradcam}))",
                key="target_category_gradcam_input",
                placeholder=str(predicted_idx_for_gradcam)
            )

            target_category_for_gradcam = predicted_idx_for_gradcam
            if target_category_input_gradcam:
                try:
                    target_category_for_gradcam = int(target_category_input_gradcam)
                except ValueError:
                    st.warning(f"Giá trị '{target_category_input_gradcam}' không hợp lệ cho Target Category. Sử dụng index lớp được dự đoán ({predicted_idx_for_gradcam}).")
                    target_category_for_gradcam = predicted_idx_for_gradcam
            
            num_classes_loaded = st.session_state.num_classes_loaded
            if not (0 <= target_category_for_gradcam < num_classes_loaded):
                st.error(f"Target Category Index ({target_category_for_gradcam}) nằm ngoài khoảng hợp lệ [0, {num_classes_loaded-1}]. Vui lòng chọn một index hợp lệ.")
            else:
                smooth_cam_gradcam = st.checkbox("Làm mịn bản đồ CAM (Smooth CAM)", value=False, key="smooth_cam_checkbox") #
                gaussian_ksize_gradcam = 7
                if smooth_cam_gradcam: #
                    gaussian_ksize_gradcam = st.number_input( #
                        "Kích thước Kernel Gaussian (số lẻ)", 
                        min_value=3, max_value=21, value=7, step=2, 
                        key="gaussian_ksize_gradcam_input"
                    )

                if st.button("Tạo Grad-CAM", key="generate_gradcam_button"):
                    with st.spinner("Đang tạo Grad-CAM..."):
                        original_img_display, cam_image = generate_gradcam_image(
                            st.session_state.model,
                            st.session_state.device,
                            pil_image_for_gradcam,
                            target_category_for_gradcam,
                            st.session_state.class_indices,
                            img_size=224, # Có thể đặt làm tham số cấu hình nếu muốn
                            smooth_cam=smooth_cam_gradcam,
                            gaussian_ksize=gaussian_ksize_gradcam
                        )

                    if original_img_display is not None and cam_image is not None:
                        st.subheader("Kết Quả Grad-CAM")
                        
                        target_gradcam_class_name_display = str(target_category_for_gradcam)
                        if st.session_state.class_indices:
                            target_gradcam_class_name_display = st.session_state.class_indices.get(str(target_category_for_gradcam), str(target_category_for_gradcam))
                        
                        col_gradcam1, col_gradcam2 = st.columns(2)
                        with col_gradcam1:
                            caption_original = f"Ảnh Gốc"
                            if last_prediction_info:
                                caption_original += f"\nDự đoán: {last_prediction_info['class_name']} ({last_prediction_info['confidence']:.2f})"
                            st.image(original_img_display, caption=caption_original, use_column_width=True)
                        with col_gradcam2:
                            smooth_text = " (Làm mịn)" if smooth_cam_gradcam else "" #
                            st.image(cam_image, caption=f"Grad-CAM cho lớp: {target_gradcam_class_name_display}{smooth_text}", use_column_width=True)
                    # Không cần else ở đây vì generate_gradcam_image đã hiển thị lỗi bằng st.error()
    
    st.sidebar.markdown("---") #
    st.sidebar.markdown("Được hỗ trợ bởi Đối tác Lập trình Gemini") #

if __name__ == '__main__': #
    main_app() #