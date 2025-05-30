import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
import os
import random
import json

# Giả sử MedMamba.py nằm cùng cấp hoặc trong PYTHONPATH
# Nếu app_streamlit_demo.py nằm trong thư mục MedMamba/
# và MedMamba.py cũng ở trong MedMamba/, thì import này sẽ hoạt động.
try:
    from MedMamba import VSSM as medmamba
except ImportError:
    st.error("Không thể import MedMamba. Hãy đảm bảo MedMamba.py ở đúng vị trí.")
    st.stop() # Dừng ứng dụng nếu không import được model

# --- Cấu hình và Hàm hỗ trợ ---

# Transform ảnh chuẩn
def get_transform(img_size=224):
    """Trả về transform chuẩn cho ảnh đầu vào."""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # Sử dụng normalize giống train.py và demo.py
    ])

# Hàm nạp mô hình (sử dụng cache của Streamlit để tăng tốc)
@st.cache_resource # Cache resource để không nạp lại model mỗi lần tương tác UI
def load_medmamba_model(checkpoint_path, num_classes):
    """Nạp mô hình MedMamba từ checkpoint."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    try:
        # Thêm weights_only=True để tăng cường bảo mật nếu bạn tin tưởng nguồn checkpoint
        # Hoặc để False nếu checkpoint chứa các đối tượng tùy chỉnh cần thiết
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
        actual_num_classes = checkpoint.get('num_classes')
        if actual_num_classes is None:
            st.warning(f"Checkpoint không chứa 'num_classes'. Sử dụng giá trị đầu vào: {num_classes}.")
        elif actual_num_classes != num_classes:
            st.warning(f"Số lớp trong checkpoint ({actual_num_classes}) khác với số lớp nhập vào ({num_classes}). Sử dụng giá trị từ checkpoint: {actual_num_classes}.")
            num_classes = actual_num_classes
            
        model = medmamba(num_classes=num_classes)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        model.to(device)
        
        # Lấy class_indices từ checkpoint nếu có
        class_indices_from_ckpt = checkpoint.get('class_indices')
        if class_indices_from_ckpt and isinstance(class_indices_from_ckpt, dict):
            # Đảm bảo key của class_indices là string nếu nó được đọc từ JSON
            if all(isinstance(k, int) for k in class_indices_from_ckpt.keys()):
                 class_indices_from_ckpt = {str(k): v for k,v in class_indices_from_ckpt.items()}
        
        st.success(f"Mô hình được nạp thành công từ '{os.path.basename(checkpoint_path)}' trên {device}.")
        return model, device, class_indices_from_ckpt, num_classes
    except FileNotFoundError:
        st.error(f"Không tìm thấy tệp checkpoint tại: {checkpoint_path}")
    except KeyError as e:
        st.error(f"Lỗi KeyError khi nạp checkpoint (có thể thiếu 'model_state_dict' hoặc 'num_classes'): {e}")
    except Exception as e:
        st.error(f"Lỗi khi nạp mô hình: {e}")
    return None, None, None, num_classes


@st.cache_data # Cache data cho class_indices từ file
def load_class_indices_from_file(class_indices_path):
    """Nạp class_indices từ tệp JSON được chỉ định."""
    if class_indices_path and os.path.exists(class_indices_path):
        try:
            with open(class_indices_path, 'r') as f:
                class_indices = json.load(f)
            # Đảm bảo key là string nếu nó được đọc từ JSON
            if isinstance(class_indices, dict) and any(isinstance(k, int) for k in class_indices.keys()):
                 class_indices = {str(k): v for k,v in class_indices.items()}
            st.info(f"Đã nạp class_indices từ tệp: {class_indices_path}")
            return class_indices
        except Exception as e:
            st.warning(f"Không thể nạp class_indices từ '{class_indices_path}': {e}")
    return None

# Hàm dự đoán
def predict(model, device, image_pil, transform, class_indices):
    """Thực hiện dự đoán trên ảnh PIL đã cho."""
    img_tensor = transform(image_pil)
    input_tensor = torch.unsqueeze(img_tensor, dim=0).to(device)

    with torch.no_grad():
        output_logits = model(input_tensor)
        probabilities = torch.softmax(output_logits, dim=1)
        predicted_prob_tensor, predicted_idx_tensor = torch.max(probabilities, dim=1)
        predicted_idx = predicted_idx_tensor.item()
        predicted_confidence = predicted_prob_tensor.item()

    predicted_class_name = str(predicted_idx) # Mặc định là index nếu không có class_indices
    if class_indices:
        # class_indices có thể là {idx_str: name} hoặc {name: idx_int}
        # Giả sử train.py lưu dạng {idx_str: name} hoặc {idx_int: name} rồi json.dump đổi int key -> str key
        predicted_class_name = class_indices.get(str(predicted_idx), f"Lớp không xác định (Index: {predicted_idx})")
    
    return predicted_class_name, predicted_confidence, predicted_idx

# --- Giao diện Streamlit ---
def main_app():
    st.set_page_config(page_title="Demo MedMamba", layout="wide")
    st.title("🐍 Demo Phân Loại Ảnh Y Tế với MedMamba")

    # --- Cấu hình Model trong Sidebar ---
    st.sidebar.header("⚙️ Cấu Hình Mô Hình")
    
    # Cung cấp đường dẫn mặc định (có thể cần điều chỉnh cho môi trường của bạn)
    # Ví dụ: nếu file checkpoint và class_indices nằm cùng thư mục với app_streamlit_demo.py
    default_checkpoint_path = "YOUR_MODEL_CHECKPOINT.pth" # THAY THẾ BẰNG ĐƯỜNG DẪN ĐÚNG
    default_class_indices_path = "class_indices.json"    # THAY THẾ BẰNG ĐƯỜNG DẪN ĐÚNG
    
    checkpoint_path_input = st.sidebar.text_input(
        "Đường dẫn đến Checkpoint (.pth)", 
        value=default_checkpoint_path,
        help="Cung cấp đường dẫn đầy đủ đến tệp checkpoint của mô hình MedMamba."
    )
    class_indices_path_input = st.sidebar.text_input(
        "Đường dẫn đến Class Indices (.json) (Tùy chọn)", 
        value=default_class_indices_path,
        help="Tệp JSON chứa ánh xạ từ index sang tên lớp."
    )
    num_classes_input = st.sidebar.number_input(
        "Số Lượng Lớp (nếu không có trong checkpoint)", 
        min_value=1, value=3, step=1, # Giá trị mặc định, bạn có thể thay đổi
        help="Số lớp đầu ra của mô hình. Sẽ được ghi đè nếu checkpoint chứa thông tin này."
    )

    # Khởi tạo session_state nếu chưa có
    if 'model' not in st.session_state:
        st.session_state.model = None
        st.session_state.device = None
        st.session_state.class_indices = None
        st.session_state.num_classes_loaded = num_classes_input
        st.session_state.model_loaded_path = ""

    if st.sidebar.button("Nạp Mô Hình & Class Indices", key="load_model_button"):
        st.session_state.model = None # Reset để nạp lại
        if checkpoint_path_input:
            # Nạp model và class_indices từ checkpoint (nếu có)
            model, device, class_indices_from_ckpt, num_classes_final = load_medmamba_model(checkpoint_path_input, num_classes_input)
            
            if model and device:
                st.session_state.model = model
                st.session_state.device = device
                st.session_state.num_classes_loaded = num_classes_final # Cập nhật số lớp thực tế
                st.session_state.model_loaded_path = checkpoint_path_input

                # Ưu tiên class_indices từ file JSON nếu được cung cấp và tồn tại
                class_indices_from_file = load_class_indices_from_file(class_indices_path_input)
                if class_indices_from_file:
                    st.session_state.class_indices = class_indices_from_file
                elif class_indices_from_ckpt:
                    st.session_state.class_indices = class_indices_from_ckpt
                    st.sidebar.info("Đã sử dụng class_indices từ checkpoint.")
                else:
                    st.session_state.class_indices = None
                    st.sidebar.warning("Không tìm thấy class_indices. Dự đoán sẽ chỉ hiển thị index của lớp.")
            else: # Lỗi khi nạp model
                st.session_state.model = None
                st.session_state.class_indices = None # Reset luôn class_indices
        else:
            st.sidebar.error("Vui lòng cung cấp đường dẫn đến checkpoint.")

    # Kiểm tra xem model đã được nạp chưa
    if st.session_state.model is None:
        st.warning("Mô hình chưa được nạp. Vui lòng cấu hình và nhấn 'Nạp Mô Hình' trong thanh sidebar.")
        st.stop() # Dừng ở đây nếu model chưa được nạp

    st.success(f"Mô hình **{os.path.basename(st.session_state.model_loaded_path)}** đã được nạp và sẵn sàng!")
    st.info(f"Số lớp của mô hình: **{st.session_state.num_classes_loaded}**")
    if st.session_state.class_indices:
        st.write("Các lớp được phát hiện:")
        st.json(st.session_state.class_indices, expanded=False)
    
    # Lấy transform
    img_transform = get_transform()

    # --- Lựa chọn Chế Độ Demo ---
    st.markdown("---")
    st.header("🔬 Chế Độ Dự Đoán")
    prediction_mode = st.radio(
        "Chọn chế độ dự đoán:",
        ("Tải Ảnh Lên", "Dự Đoán Ảnh Ngẫu Nhiên Từ Thư Mục"),
        key="prediction_mode_radio"
    )

    if prediction_mode == "Tải Ảnh Lên":
        uploaded_file = st.file_uploader(
            "Chọn một hình ảnh...", 
            type=["png", "jpg", "jpeg", "bmp"],
            key="file_uploader"
        )
        if uploaded_file is not None:
            try:
                image_pil = Image.open(uploaded_file).convert('RGB')
                
                col1, col2 = st.columns([2,3]) # Chia cột, cột ảnh nhỏ hơn, cột kết quả lớn hơn
                with col1:
                    st.image(image_pil, caption="Ảnh Đã Tải Lên", use_column_width=True)

                with col2:
                    if st.button("Thực Hiện Dự Đoán", key="predict_uploaded_button"):
                        with st.spinner("Đang dự đoán..."):
                            class_name, confidence, class_idx = predict(
                                st.session_state.model,
                                st.session_state.device,
                                image_pil,
                                img_transform,
                                st.session_state.class_indices
                            )
                        st.subheader("Kết Quả Dự Đoán:")
                        st.markdown(f"**Lớp Dự Đoán:** `{class_name}` (Index: {class_idx})")
                        st.markdown(f"**Độ Tin Cậy:** `{confidence:.4f}`")
            except Exception as e:
                st.error(f"Lỗi xử lý ảnh tải lên: {e}")

    elif prediction_mode == "Dự Đoán Ảnh Ngẫu Nhiên Từ Thư Mục":
        # Cung cấp đường dẫn mặc định (có thể cần điều chỉnh)
        default_test_dir = "PATH_TO_YOUR_TEST_IMAGE_FOLDER" # THAY THẾ BẰNG ĐƯỜNG DẪN ĐÚNG
        test_dir_input = st.text_input(
            "Đường dẫn đến Thư Mục Ảnh Test", 
            value=default_test_dir,
            help="Cung cấp đường dẫn đến thư mục chứa các ảnh để chọn ngẫu nhiên."
        )

        if st.button("Dự Đoán Ảnh Ngẫu Nhiên", key="predict_random_button"):
            if not os.path.isdir(test_dir_input):
                st.error(f"Không tìm thấy thư mục: {test_dir_input}")
            else:
                image_files = []
                # Quét ảnh trong thư mục và thư mục con
                for root, _, files in os.walk(test_dir_input):
                    for file in files:
                        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                            image_files.append(os.path.join(root, file))
                
                if not image_files:
                    st.warning(f"Không tìm thấy tệp ảnh nào trong thư mục: {test_dir_input}")
                else:
                    random_image_path = random.choice(image_files)
                    try:
                        image_pil = Image.open(random_image_path).convert('RGB')
                        
                        col1, col2 = st.columns([2,3])
                        with col1:
                            st.image(image_pil, caption=f"Ảnh Ngẫu Nhiên: {os.path.basename(random_image_path)}", use_column_width=True)
                        
                        with col2:
                            with st.spinner("Đang dự đoán..."):
                                class_name, confidence, class_idx = predict(
                                    st.session_state.model,
                                    st.session_state.device,
                                    image_pil,
                                    img_transform,
                                    st.session_state.class_indices
                                )
                            st.subheader("Kết Quả Dự Đoán:")
                            st.markdown(f"**Ảnh Được Chọn:** `{os.path.basename(random_image_path)}`")
                            st.markdown(f"**Lớp Dự Đoán:** `{class_name}` (Index: {class_idx})")
                            st.markdown(f"**Độ Tin Cậy:** `{confidence:.4f}`")
                    except Exception as e:
                        st.error(f"Lỗi xử lý ảnh {random_image_path}: {e}")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("Được hỗ trợ bởi Đối tác Lập trình Gemini")

if __name__ == '__main__':
    main_app()