import os
import numpy as np
import tensorflow as tf
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import nibabel as nib

# Set page config
st.set_page_config(
    page_title="BraTS 3D Brain Tumor Segmentation System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .sub-title {
        font-size: 1.2rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🧠 BraTS 3D Brain Tumor Segmentation System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">AI-Powered 3D MRI Segmentation using Deep Learning (3D U-Net)</div>', unsafe_allow_html=True)

# Set environment framework variable for segmentation-models-3D
os.environ["SM_FRAMEWORK"] = "tf.keras"

import zipfile
import tempfile

def build_default_unet():
    """Constructs the standard BraTS 3D U-Net architecture (128x128x128x3 -> 4 classes)."""
    inputs = tf.keras.layers.Input((128, 128, 128, 3))
    kernel_initializer = 'he_uniform'
    
    # Encoder
    c1 = tf.keras.layers.Conv3D(16, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(inputs)
    c1 = tf.keras.layers.BatchNormalization()(c1)
    c1 = tf.keras.layers.Dropout(0.1)(c1)
    c1 = tf.keras.layers.Conv3D(16, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c1)
    p1 = tf.keras.layers.MaxPooling3D((2, 2, 2))(c1)

    c2 = tf.keras.layers.Conv3D(32, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(p1)
    c2 = tf.keras.layers.BatchNormalization()(c2)
    c2 = tf.keras.layers.Dropout(0.1)(c2)
    c2 = tf.keras.layers.Conv3D(32, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c2)
    p2 = tf.keras.layers.MaxPooling3D((2, 2, 2))(c2)

    c3 = tf.keras.layers.Conv3D(64, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(p2)
    c3 = tf.keras.layers.BatchNormalization()(c3)
    c3 = tf.keras.layers.Dropout(0.2)(c3)
    c3 = tf.keras.layers.Conv3D(64, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c3)
    p3 = tf.keras.layers.MaxPooling3D((2, 2, 2))(c3)

    c4 = tf.keras.layers.Conv3D(128, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(p3)
    c4 = tf.keras.layers.BatchNormalization()(c4)
    c4 = tf.keras.layers.Dropout(0.2)(c4)
    c4 = tf.keras.layers.Conv3D(128, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c4)
    p4 = tf.keras.layers.MaxPooling3D((2, 2, 2))(c4)

    # Bottleneck
    c5 = tf.keras.layers.Conv3D(256, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(p4)
    c5 = tf.keras.layers.BatchNormalization()(c5)
    c5 = tf.keras.layers.Dropout(0.3)(c5)
    c5 = tf.keras.layers.Conv3D(256, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c5)

    # Decoder
    u6 = tf.keras.layers.Conv3DTranspose(128, (2, 2, 2), strides=(2, 2, 2), padding='same')(c5)
    u6 = tf.keras.layers.concatenate([u6, c4])
    c6 = tf.keras.layers.Conv3D(128, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(u6)
    c6 = tf.keras.layers.Dropout(0.2)(c6)
    c6 = tf.keras.layers.Conv3D(128, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c6)

    u7 = tf.keras.layers.Conv3DTranspose(64, (2, 2, 2), strides=(2, 2, 2), padding='same')(c6)
    u7 = tf.keras.layers.concatenate([u7, c3])
    c7 = tf.keras.layers.Conv3D(64, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(u7)
    c7 = tf.keras.layers.Dropout(0.2)(c7)
    c7 = tf.keras.layers.Conv3D(64, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c7)

    u8 = tf.keras.layers.Conv3DTranspose(32, (2, 2, 2), strides=(2, 2, 2), padding='same')(c7)
    u8 = tf.keras.layers.concatenate([u8, c2])
    c8 = tf.keras.layers.Conv3D(32, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(u8)
    c8 = tf.keras.layers.Dropout(0.1)(c8)
    c8 = tf.keras.layers.Conv3D(32, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c8)

    u9 = tf.keras.layers.Conv3DTranspose(16, (2, 2, 2), strides=(2, 2, 2), padding='same')(c8)
    u9 = tf.keras.layers.concatenate([u9, c1])
    c9 = tf.keras.layers.Conv3D(16, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(u9)
    c9 = tf.keras.layers.Dropout(0.1)(c9)
    c9 = tf.keras.layers.Conv3D(16, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c9)

    outputs = tf.keras.layers.Conv3D(4, (1, 1, 1), activation='softmax', dtype='float32')(c9)
    return tf.keras.models.Model(inputs=[inputs], outputs=[outputs])

@st.cache_resource
def load_trained_model(model_path='brats_3d_best.keras'):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    candidate_paths = [
        os.path.abspath(os.path.join(base_dir, 'brats_3d_best.keras')),
        os.path.abspath(os.path.join(base_dir, 'brats_3d.hdf5')),
        os.path.abspath(os.path.join(os.getcwd(), 'brats_3d_best.keras')),
        os.path.abspath(os.path.join(os.getcwd(), 'brats_3d.hdf5')),
    ]
    if model_path:
        candidate_paths.insert(0, os.path.abspath(os.path.join(base_dir, model_path)))
        candidate_paths.insert(1, os.path.abspath(model_path))
    
    resolved_path = None
    for p in candidate_paths:
        if p and os.path.exists(p) and os.path.isfile(p):
            resolved_path = p
            break
    
    if not resolved_path:
        return None, f"Model file 'brats_3d_best.keras' not found. Checked paths in base directory: {base_dir}"
    
    # Method 1: Standard tf.keras load_model with absolute path
    try:
        model = tf.keras.models.load_model(resolved_path, compile=False)
        return model, None
    except Exception:
        pass

    # Method 2: Standalone keras.models.load_model if available
    try:
        import keras
        model = keras.models.load_model(resolved_path, compile=False)
        return model, None
    except Exception:
        pass

    # Method 3: Fallback - Rebuild 3D U-Net architecture & load weights
    try:
        model = build_default_unet()
        if zipfile.is_zipfile(resolved_path):
            with zipfile.ZipFile(resolved_path, 'r') as zf:
                if 'model.weights.h5' in zf.namelist():
                    with tempfile.NamedTemporaryFile(suffix='.weights.h5', delete=False) as tmp_w:
                        tmp_w.write(zf.read('model.weights.h5'))
                        tmp_w_path = tmp_w.name
                    try:
                        model.load_weights(tmp_w_path)
                    finally:
                        if os.path.exists(tmp_w_path):
                            os.remove(tmp_w_path)
                    return model, None
        model.load_weights(resolved_path)
        return model, None
    except Exception as e_final:
        return None, f"Error loading model from '{resolved_path}': {str(e_final)}"


# Sidebar Setup
st.sidebar.header("🧠 Model Configuration")

model_choice = st.sidebar.selectbox(
    "Select Model for Processing:",
    ["Default Pretrained 3D U-Net (BraTS)", "Upload Custom Model (.keras / .h5)"]
)

model = None
model_name = ""

if model_choice == "Default Pretrained 3D U-Net (BraTS)":
    model, load_err = load_trained_model('brats_3d_best.keras')
    if load_err:
        st.sidebar.error(load_err)
    else:
        model_name = "Default 3D U-Net (BraTS)"
        st.sidebar.success(f"✅ Active: {model_name}")
        st.sidebar.caption(f"📊 Parameters: {model.count_params():,} | Input: (128, 128, 128, 3)")
else:
    st.sidebar.info("Upload your custom trained 3D segmentation model (.keras or .h5 format):")
    custom_model_file = st.sidebar.file_uploader("Upload Custom Model", type=['keras', 'h5', 'hdf5'])
    
    if custom_model_file:
        custom_path = os.path.abspath(f"temp_custom_{custom_model_file.name}")
        with open(custom_path, "wb") as f:
            f.write(custom_model_file.getbuffer())
        
        custom_model, c_err = load_trained_model(custom_path)
        if c_err:
            st.sidebar.error(f"Failed to load custom model: {c_err}")
            model = None
        else:
            model = custom_model
            model_name = f"Custom Model ({custom_model_file.name})"
            st.sidebar.success(f"✅ Active: {model_name}")
            st.sidebar.caption(f"📊 Parameters: {model.count_params():,} | Input: {model.input_shape}")
    else:
        st.sidebar.warning("Please upload a `.keras` or `.h5` model file to continue.")

if model is None:
    st.warning("⚠️ No model loaded. Please select the Default Model or upload a Custom Model in the sidebar to proceed.")
    st.stop()

# Input Mode
st.sidebar.markdown("---")
st.sidebar.header("📁 Input Data Settings")
input_mode = st.sidebar.radio(
    "Choose Input Data Source:",
    ["Upload Preprocessed .npy Volume", "Upload Raw NIfTI (.nii / .nii.gz) Files", "Generate Synthetic Test Volume"]
)

volume_3d = None

if input_mode == "Upload Preprocessed .npy Volume":
    uploaded_npy = st.file_uploader("Upload 3D Volume (.npy file with shape 128x128x128x3)", type=['npy'])
    if uploaded_npy:
        try:
            volume_3d = np.load(uploaded_npy).astype(np.float32)
            if volume_3d.ndim == 3:
                volume_3d = np.repeat(volume_3d[:, :, :, np.newaxis], 3, axis=-1)
            st.success(f"Loaded .npy volume with shape: {volume_3d.shape}")
        except Exception as e:
            st.error(f"Error reading .npy file: {e}")

elif input_mode == "Upload Raw NIfTI (.nii / .nii.gz) Files":
    st.info("Upload FLAIR, T1CE, and T2 MRI NIfTI scans for the patient:")
    flair_file = st.file_uploader("Upload FLAIR Scan (*flair.nii / .nii.gz)", type=['nii', 'gz'])
    t1ce_file = st.file_uploader("Upload T1CE Scan (*t1ce.nii / .nii.gz)", type=['nii', 'gz'])
    t2_file = st.file_uploader("Upload T2 Scan (*t2.nii / .nii.gz)", type=['nii', 'gz'])

    if flair_file and t1ce_file and t2_file:
        with st.spinner("Processing & Normalizing NIfTI MRI Volumes..."):
            try:
                def save_temp_nifti(uploaded_file, prefix):
                    ext = ".nii.gz" if uploaded_file.name.lower().endswith(".gz") else ".nii"
                    filename = f"{prefix}{ext}"
                    with open(filename, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    return filename

                flair_path = save_temp_nifti(flair_file, "temp_flair")
                t1ce_path = save_temp_nifti(t1ce_file, "temp_t1ce")
                t2_path = save_temp_nifti(t2_file, "temp_t2")

                scaler = MinMaxScaler()
                flair = nib.load(flair_path).get_fdata().astype(np.float32)
                flair = scaler.fit_transform(flair.reshape(-1, flair.shape[-1])).reshape(flair.shape)

                t1ce = nib.load(t1ce_path).get_fdata().astype(np.float32)
                t1ce = scaler.fit_transform(t1ce.reshape(-1, t1ce.shape[-1])).reshape(t1ce.shape)

                t2 = nib.load(t2_path).get_fdata().astype(np.float32)
                t2 = scaler.fit_transform(t2.reshape(-1, t2.shape[-1])).reshape(t2.shape)

                combined = np.stack([flair, t1ce, t2], axis=3).astype(np.float32)
                
                if combined.shape[0] >= 128 and combined.shape[1] >= 128 and combined.shape[2] >= 128:
                    x_start = (combined.shape[0] - 128) // 2
                    y_start = (combined.shape[1] - 128) // 2
                    z_start = (combined.shape[2] - 128) // 2
                    volume_3d = combined[x_start:x_start+128, y_start:y_start+128, z_start:z_start+128]
                else:
                    volume_3d = combined

                st.success(f"NIfTI volumes processed & cropped to shape: {volume_3d.shape}")
            except Exception as e:
                st.error(f"Error processing NIfTI scans: {e}")

elif input_mode == "Generate Synthetic Test Volume":
    if st.button("Generate Test Sample"):
        volume_3d = np.zeros((128, 128, 128, 3), dtype=np.float32)
        grid_x, grid_y, grid_z = np.ogrid[:128, :128, :128]
        distance = np.sqrt((grid_x - 64)**2 + (grid_y - 64)**2 + (grid_z - 64)**2)
        brain_mask = distance <= 50
        tumor_mask = distance <= 18
        volume_3d[brain_mask, :] = 0.5
        volume_3d[tumor_mask, 0] = 0.9
        volume_3d[tumor_mask, 1] = 0.8
        st.success("Generated synthetic 3D MRI sample volume!")

if volume_3d is not None:
    st.markdown("---")
    st.subheader("🔍 Brain Tumor AI Segmentation")

    if st.button("🚀 Analyze & Segment Tumor Now", type="primary"):
        with st.spinner("Running 3D U-Net Model Inference..."):
            input_tensor = np.expand_dims(volume_3d, axis=0)
            pred_probs = model.predict(input_tensor)
            pred_mask = np.argmax(pred_probs, axis=-1)[0]

            st.session_state['volume_3d'] = volume_3d
            st.session_state['pred_mask'] = pred_mask
            st.success("Tumor Segmentation Complete!")

if 'pred_mask' in st.session_state and st.session_state['pred_mask'] is not None:
    volume_3d = st.session_state['volume_3d']
    pred_mask = st.session_state['pred_mask']

    st.markdown("### 📊 Segmentation Quantitative Summary")
    c1, c2, c3, c4 = st.columns(4)

    total_voxels = pred_mask.size
    healthy_voxels = np.sum(pred_mask == 0)
    necrotic_voxels = np.sum(pred_mask == 1)
    edema_voxels = np.sum(pred_mask == 2)
    enhancing_voxels = np.sum(pred_mask == 3)

    c1.metric("Healthy Tissue", f"{(healthy_voxels / total_voxels)*100:.1f}%")
    c2.metric("Necrotic Core (Label 1)", f"{necrotic_voxels} voxels", f"{necrotic_voxels/1000:.2f} cm³")
    c3.metric("Edema Region (Label 2)", f"{edema_voxels} voxels", f"{edema_voxels/1000:.2f} cm³")
    c4.metric("Enhancing Tumor (Label 3)", f"{enhancing_voxels} voxels", f"{enhancing_voxels/1000:.2f} cm³")

    st.markdown("---")
    st.markdown("### 🖼️ Interactive 3D Multi-Planar Slice Viewer")

    col_controls1, col_controls2 = st.columns(2)
    with col_controls1:
        plane = st.selectbox("Select Viewing Plane:", ["Axial (Transverse)", "Coronal", "Sagittal"])
    with col_controls2:
        slice_idx = st.slider("Slice Index:", 0, 127, 64)

    if plane == "Axial (Transverse)":
        mri_slice = volume_3d[:, :, slice_idx, 0]
        mask_slice = pred_mask[:, :, slice_idx]
    elif plane == "Coronal":
        mri_slice = volume_3d[:, slice_idx, :, 0]
        mask_slice = pred_mask[:, slice_idx, :]
    else:
        mri_slice = volume_3d[slice_idx, :, :, 0]
        mask_slice = pred_mask[slice_idx, :, :]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    from matplotlib.colors import ListedColormap
    cmap_custom = ListedColormap(['black', 'red', 'green', 'yellow'])

    axes[0].imshow(mri_slice, cmap='gray')
    axes[0].set_title(f"FLAIR MRI Scan ({plane} Slice {slice_idx})")
    axes[0].axis('off')

    axes[1].imshow(mask_slice, cmap=cmap_custom, vmin=0, vmax=3)
    axes[1].set_title("Predicted Segmentation Mask")
    axes[1].axis('off')

    axes[2].imshow(mri_slice, cmap='gray')
    axes[2].imshow(mask_slice, cmap=cmap_custom, alpha=0.5, vmin=0, vmax=3)
    axes[2].set_title("MRI + Tumor Overlay")
    axes[2].axis('off')

    st.pyplot(fig)

    st.markdown("""
    **Color Legend:**
    - ⬛ **Black**: Background / Healthy Brain Tissue
    - 🔴 **Red**: Necrotic Core & Non-Enhancing Tumor (Label 1)
    - 🟢 **Green**: Peritumoral Edema (Label 2)
    - 🟡 **Yellow**: Enhancing Tumor (Label 3)
    """)
