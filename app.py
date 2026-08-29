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

@st.cache_resource
def load_trained_model(model_path='brats_3d_best.keras'):
    if not os.path.exists(model_path):
        for p in ['brats_3d_best.keras', 'brats_3d.hdf5', '../brats_3d_best.keras']:
            if os.path.exists(p):
                model_path = p
                break
    
    if not os.path.exists(model_path):
        return None, f"Model file '{model_path}' not found in current directory."
    
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
        return model, None
    except Exception as e:
        return None, f"Error loading model: {str(e)}"

# Load model
model, load_err = load_trained_model()

# Sidebar Setup
st.sidebar.header("📁 Model & Input Settings")

if load_err:
    st.sidebar.error(load_err)
    uploaded_model = st.sidebar.file_uploader("Upload 'brats_3d_best.keras' file", type=['keras', 'h5', 'hdf5'])
    if uploaded_model:
        temp_model_path = "brats_3d_best.keras"
        with open(temp_model_path, "wb") as f:
            f.write(uploaded_model.getbuffer())
        model, load_err = load_trained_model(temp_model_path)
        if model:
            st.sidebar.success("Custom model loaded successfully!")

if model is None:
    st.warning("⚠️ Please place or upload your trained `brats_3d_best.keras` model file to proceed.")
    st.stop()
else:
    st.sidebar.success("✅ 3D U-Net Model Loaded Successfully")

# Input Mode
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
