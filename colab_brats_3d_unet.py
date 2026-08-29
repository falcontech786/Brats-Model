# ==============================================================================
# BraTS 3D Brain Tumor Segmentation - Google Colab Complete Master Pipeline
# ==============================================================================
# Instructions for Google Colab:
# 1. Open https://colab.research.google.com
# 2. Click "New Notebook"
# 3. Change Runtime to GPU: Runtime -> Change runtime type -> Select T4 GPU -> Save
# 4. Copy and run the code cells below in your Colab notebook.
# ==============================================================================

# ------------------------------------------------------------------------------
# CELL 1: Verify GPU & Install Dependencies
# ------------------------------------------------------------------------------
import os
import glob
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# GPU verification
import tensorflow as tf
print("TensorFlow Version:", tf.__version__)
print("GPUs Available:", tf.config.list_physical_devices('GPU'))

# Install required packages in Colab without Keras version conflicts:
# !pip install -q nibabel split-folders tifffile scikit-learn
# !pip install -q segmentation-models-3D --no-deps
# !pip install -q classification-models-3D --no-deps

import nibabel as nib
from tifffile import imwrite as imsave
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.utils import to_categorical

os.environ["SM_FRAMEWORK"] = "tf.keras"
import segmentation_models_3D as sm

# Optional: Enable Mixed Precision for 2x faster GPU training
# tf.keras.mixed_precision.set_global_policy('mixed_float16')

# ------------------------------------------------------------------------------
# CELL 2: Google Drive Mounting & Dataset Path Configuration
# ------------------------------------------------------------------------------
# Run this in Colab to mount Google Drive:
# from google.colab import drive
# drive.mount('/content/drive')

# Update this path to where your raw dataset is stored inside Google Drive:
DRIVE_DATASET_PATH = '/content/drive/MyDrive/Data'

# Working directory on fast local Colab SSD (Lightning fast processing, zero Drive corruption!)
LOCAL_WORK_DIR = '/content/BraTS2020_TrainingData'
INPUT_3CHAN_DIR = os.path.join(LOCAL_WORK_DIR, 'input_data_3channels')
INPUT_128_DIR = os.path.join(LOCAL_WORK_DIR, 'input_data_128')

os.makedirs(os.path.join(INPUT_3CHAN_DIR, 'images'), exist_ok=True)
os.makedirs(os.path.join(INPUT_3CHAN_DIR, 'masks'), exist_ok=True)
os.makedirs(INPUT_128_DIR, exist_ok=True)

# ------------------------------------------------------------------------------
# CELL 3: Data Preparation (NIfTI -> 128x128x128 3D Patches -> Train/Val Split)
# ------------------------------------------------------------------------------
def prepare_brats_data(dataset_path):
    train_img_dir = os.path.join(INPUT_128_DIR, 'train/images')
    train_mask_dir = os.path.join(INPUT_128_DIR, 'train/masks')
    val_img_dir = os.path.join(INPUT_128_DIR, 'val/images')
    val_mask_dir = os.path.join(INPUT_128_DIR, 'val/masks')

    os.makedirs(train_img_dir, exist_ok=True)
    os.makedirs(train_mask_dir, exist_ok=True)
    os.makedirs(val_img_dir, exist_ok=True)
    os.makedirs(val_mask_dir, exist_ok=True)

    if len(os.listdir(train_img_dir)) > 0:
        print("Processed dataset already exists on local SSD! Ready for training.")
        return

    # Check Drive for existing .npy files from previous runs
    drive_3chan_img = '/content/drive/MyDrive/BraTS2020_TrainingData/input_data_3channels/images'
    drive_3chan_mask = '/content/drive/MyDrive/BraTS2020_TrainingData/input_data_3channels/masks'

    if os.path.exists(drive_3chan_img) and len(os.listdir(drive_3chan_img)) > 0:
        print("Found existing .npy files on Google Drive! Fast copying to local SSD...")
        import shutil
        all_imgs = [f for f in os.listdir(drive_3chan_img) if f.endswith('.npy')]
        valid_pairs = []
        for img_f in all_imgs:
            mask_f = img_f.replace('image_', 'mask_')
            if os.path.exists(os.path.join(drive_3chan_mask, mask_f)):
                valid_pairs.append((img_f, mask_f))

        random.seed(42)
        random.shuffle(valid_pairs)
        split_idx = int(len(valid_pairs) * 0.75)
        train_pairs = valid_pairs[:split_idx]
        val_pairs = valid_pairs[split_idx:]

        for idx, (img_f, mask_f) in enumerate(train_pairs):
            shutil.copy2(os.path.join(drive_3chan_img, img_f), os.path.join(train_img_dir, f'image_{idx}.npy'))
            shutil.copy2(os.path.join(drive_3chan_mask, mask_f), os.path.join(train_mask_dir, f'mask_{idx}.npy'))

        for idx, (img_f, mask_f) in enumerate(val_pairs):
            shutil.copy2(os.path.join(drive_3chan_img, img_f), os.path.join(val_img_dir, f'image_{idx}.npy'))
            shutil.copy2(os.path.join(drive_3chan_mask, mask_f), os.path.join(val_mask_dir, f'mask_{idx}.npy'))

        print(f"FAST SSD COPY COMPLETE: {len(train_pairs)} Train pairs, {len(val_pairs)} Val pairs ready on local SSD!")
        return

    print("Starting data preparation from:", dataset_path)
    scaler = MinMaxScaler()
    
    # Recursive search for all modality files (.nii or .nii.gz)
    t2_list = sorted(glob.glob(os.path.join(dataset_path, '**/*t2.nii*'), recursive=True))
    t1ce_list = sorted(glob.glob(os.path.join(dataset_path, '**/*t1ce.nii*'), recursive=True))
    flair_list = sorted(glob.glob(os.path.join(dataset_path, '**/*flair.nii*'), recursive=True))
    mask_list = sorted(glob.glob(os.path.join(dataset_path, '**/*seg.nii*'), recursive=True))

    # Fallback auto-search across Google Drive if not found at dataset_path
    if len(t2_list) == 0:
        print(f"No files found at '{dataset_path}'. Auto-searching entire Google Drive...")
        drive_root = '/content/drive/MyDrive'
        t2_list = sorted(glob.glob(os.path.join(drive_root, '**/*t2.nii*'), recursive=True))
        t1ce_list = sorted(glob.glob(os.path.join(drive_root, '**/*t1ce.nii*'), recursive=True))
        flair_list = sorted(glob.glob(os.path.join(drive_root, '**/*flair.nii*'), recursive=True))
        mask_list = sorted(glob.glob(os.path.join(drive_root, '**/*seg.nii*'), recursive=True))

    print(f"Found {len(t2_list)} patient volumes.")
    valid_pairs = []

    for img in range(len(t2_list)):
        print(f"Processing volume {img+1}/{len(t2_list)}: {os.path.basename(t2_list[img])}")
        
        t2 = nib.load(t2_list[img]).get_fdata().astype(np.float32)
        t2 = scaler.fit_transform(t2.reshape(-1, t2.shape[-1])).reshape(t2.shape)

        t1ce = nib.load(t1ce_list[img]).get_fdata().astype(np.float32)
        t1ce = scaler.fit_transform(t1ce.reshape(-1, t1ce.shape[-1])).reshape(t1ce.shape)

        flair = nib.load(flair_list[img]).get_fdata().astype(np.float32)
        flair = scaler.fit_transform(flair.reshape(-1, flair.shape[-1])).reshape(flair.shape)

        mask = nib.load(mask_list[img]).get_fdata().astype(np.uint8)
        mask[mask == 4] = 3

        combined = np.stack([flair, t1ce, t2], axis=3).astype(np.float32)
        combined = combined[56:184, 56:184, 13:141]
        mask = mask[56:184, 56:184, 13:141]

        val, counts = np.unique(mask, return_counts=True)
        if (1.0 - (counts[0] / counts.sum())) > 0.01:
            mask_cat = to_categorical(mask, num_classes=4).astype(np.float32)
            valid_pairs.append((combined, mask_cat, f"volume_{img}"))

    # Pair-aligned shuffle & split (75% train / 25% val)
    random.seed(42)
    random.shuffle(valid_pairs)
    split_idx = int(len(valid_pairs) * 0.75)
    train_pairs = valid_pairs[:split_idx]
    val_pairs = valid_pairs[split_idx:]

    for idx, (img_arr, mask_arr, name) in enumerate(train_pairs):
        np.save(os.path.join(train_img_dir, f'image_{idx}.npy'), img_arr)
        np.save(os.path.join(train_mask_dir, f'mask_{idx}.npy'), mask_arr)

    for idx, (img_arr, mask_arr, name) in enumerate(val_pairs):
        np.save(os.path.join(val_img_dir, f'image_{idx}.npy'), img_arr)
        np.save(os.path.join(val_mask_dir, f'mask_{idx}.npy'), mask_arr)

    print(f"Data successfully split with guaranteed pair alignment: {len(train_pairs)} Train, {len(val_pairs)} Val!")

# ------------------------------------------------------------------------------
# CELL 4: Custom 3D Data Generator with Augmentation
# ------------------------------------------------------------------------------
def is_valid_npy(path):
    if not os.path.exists(path) or os.path.getsize(path) < 128:
        return False
    try:
        with open(path, 'rb') as f:
            magic = f.read(6)
            return len(magic) == 6 and magic.startswith(b'\x93NUMPY')
    except Exception:
        return False

def load_img_batch(img_dir, img_list):
    images = []
    for img_name in img_list:
        if img_name.endswith('.npy'):
            img_path = os.path.join(img_dir, img_name)
            if is_valid_npy(img_path):
                try:
                    images.append(np.load(img_path))
                except (EOFError, OSError, ValueError):
                    pass
    return np.array(images, dtype=np.float32)

def imageLoader(img_dir, img_list, mask_dir, mask_list, batch_size, augment=False):
    valid_pairs = []
    for f in img_list:
        if f.endswith('.npy'):
            img_p = os.path.join(img_dir, f)
            mask_p = os.path.join(mask_dir, f.replace('image_', 'mask_'))
            if is_valid_npy(img_p) and is_valid_npy(mask_p):
                valid_pairs.append(f)

    L = len(valid_pairs)
    if L == 0:
        raise ValueError(f"No valid non-corrupted .npy file pairs found in {img_dir}")

    while True:
        batch_start = 0
        batch_end = batch_size
        while batch_start < L:
            limit = min(batch_end, L)
            batch_imgs = valid_pairs[batch_start:limit]
            batch_masks = [f.replace('image_', 'mask_') for f in batch_imgs]

            X = load_img_batch(img_dir, batch_imgs)
            Y = load_img_batch(mask_dir, batch_masks)

            if len(X) > 0 and len(X) == len(Y):
                # 3D Data Augmentation during training
                if augment:
                    for i in range(len(X)):
                        if random.random() > 0.5:  # Random horizontal flip
                            X[i] = np.flip(X[i], axis=0)
                            Y[i] = np.flip(Y[i], axis=0)
                        if random.random() > 0.5:  # Random vertical flip
                            X[i] = np.flip(X[i], axis=1)
                            Y[i] = np.flip(Y[i], axis=1)

                yield (X, Y)

            batch_start += batch_size
            batch_end += batch_size

# ------------------------------------------------------------------------------
# CELL 5: Build 3D U-Net Model
# ------------------------------------------------------------------------------
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv3D, MaxPooling3D, concatenate, Conv3DTranspose, BatchNormalization, Dropout

def simple_unet_model(IMG_HEIGHT=128, IMG_WIDTH=128, IMG_DEPTH=128, IMG_CHANNELS=3, num_classes=4):
    kernel_initializer = 'he_uniform'
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH, IMG_CHANNELS))
    
    # Encoder
    c1 = Conv3D(16, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(inputs)
    c1 = BatchNormalization()(c1)
    c1 = Dropout(0.1)(c1)
    c1 = Conv3D(16, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c1)
    p1 = MaxPooling3D((2, 2, 2))(c1)

    c2 = Conv3D(32, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(p1)
    c2 = BatchNormalization()(c2)
    c2 = Dropout(0.1)(c2)
    c2 = Conv3D(32, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c2)
    p2 = MaxPooling3D((2, 2, 2))(c2)

    c3 = Conv3D(64, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(p2)
    c3 = BatchNormalization()(c3)
    c3 = Dropout(0.2)(c3)
    c3 = Conv3D(64, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c3)
    p3 = MaxPooling3D((2, 2, 2))(c3)

    c4 = Conv3D(128, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(p3)
    c4 = BatchNormalization()(c4)
    c4 = Dropout(0.2)(c4)
    c4 = Conv3D(128, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c4)
    p4 = MaxPooling3D((2, 2, 2))(c4)

    # Bottleneck
    c5 = Conv3D(256, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(p4)
    c5 = BatchNormalization()(c5)
    c5 = Dropout(0.3)(c5)
    c5 = Conv3D(256, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c5)

    # Decoder
    u6 = Conv3DTranspose(128, (2, 2, 2), strides=(2, 2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv3D(128, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(u6)
    c6 = Dropout(0.2)(c6)
    c6 = Conv3D(128, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c6)

    u7 = Conv3DTranspose(64, (2, 2, 2), strides=(2, 2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv3D(64, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(u7)
    c7 = Dropout(0.2)(c7)
    c7 = Conv3D(64, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c7)

    u8 = Conv3DTranspose(32, (2, 2, 2), strides=(2, 2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv3D(32, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(u8)
    c8 = Dropout(0.1)(c8)
    c8 = Conv3D(32, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c8)

    u9 = Conv3DTranspose(16, (2, 2, 2), strides=(2, 2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1])
    c9 = Conv3D(16, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(u9)
    c9 = Dropout(0.1)(c9)
    c9 = Conv3D(16, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c9)

    outputs = Conv3D(num_classes, (1, 1, 1), activation='softmax', dtype='float32')(c9)

    model = Model(inputs=[inputs], outputs=[outputs])
    return model

# ------------------------------------------------------------------------------
# CELL 6: Model Training Pipeline with Callbacks & Hybrid Loss
# ------------------------------------------------------------------------------
def train_model(epochs=50, batch_size=2):
    train_img_dir = os.path.join(INPUT_128_DIR, 'train/images/')
    train_mask_dir = os.path.join(INPUT_128_DIR, 'train/masks/')
    val_img_dir = os.path.join(INPUT_128_DIR, 'val/images/')
    val_mask_dir = os.path.join(INPUT_128_DIR, 'val/masks/')

    os.makedirs(train_img_dir, exist_ok=True)
    os.makedirs(train_mask_dir, exist_ok=True)
    os.makedirs(val_img_dir, exist_ok=True)
    os.makedirs(val_mask_dir, exist_ok=True)

    train_img_list = sorted(os.listdir(train_img_dir))
    train_mask_list = sorted(os.listdir(train_mask_dir))
    val_img_list = sorted(os.listdir(val_img_dir))
    val_mask_list = sorted(os.listdir(val_mask_dir))

    if len(train_img_list) == 0:
        raise RuntimeError(
            f"No dataset files found in '{train_img_dir}'. "
            "Please make sure Cell 3 (prepare_brats_data) has run successfully and found NIfTI volumes in your Google Drive."
        )

    train_gen = imageLoader(train_img_dir, train_img_list, train_mask_dir, train_mask_list, batch_size, augment=True)
    val_gen = imageLoader(val_img_dir, val_img_list, val_mask_dir, val_mask_list, batch_size, augment=False)

    steps_per_epoch = max(1, len(train_img_list) // batch_size)
    val_steps_per_epoch = max(1, len(val_img_list) // batch_size)

    # Hybrid Loss: Weighted Dice Loss + Focal Loss
    wt0, wt1, wt2, wt3 = 0.25, 0.25, 0.25, 0.25
    dice_loss = sm.losses.DiceLoss(class_weights=np.array([wt0, wt1, wt2, wt3], dtype=np.float32))
    focal_loss = sm.losses.CategoricalFocalLoss()
    total_loss = dice_loss + (1.0 * focal_loss)

    metrics = ['accuracy', sm.metrics.IOUScore(threshold=0.5)]

    model = simple_unet_model(128, 128, 128, 3, 4)
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    model.compile(optimizer=optimizer, loss=total_loss, metrics=metrics)

    # Callbacks for maximum performance
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint('/content/drive/MyDrive/brats_3d_best.keras', save_best_only=True, monitor='val_loss', mode='min'),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1)
    ]

    print("Starting Training...")
    history = model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_data=val_gen,
        validation_steps=val_steps_per_epoch,
        callbacks=callbacks
    )
    return model, history

# ------------------------------------------------------------------------------
# CELL 7: Evaluation & Visual Inspection
# ------------------------------------------------------------------------------
def evaluate_and_plot(history, model, val_img_dir, val_mask_dir):
    # Plot training curves
    plt.figure(figsize=(12, 5))
    plt.subplot(121)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Loss Curve')
    plt.legend()

    plt.subplot(122)
    plt.plot(history.history['iou_score'], label='Train IoU')
    plt.plot(history.history['val_iou_score'], label='Val IoU')
    plt.title('IoU Score Curve')
    plt.legend()
    plt.show()

    # Qualitative Slice Comparison
    val_files = [f for f in os.listdir(val_img_dir) if f.endswith('.npy')]
    if val_files:
        test_img = np.load(os.path.join(val_img_dir, val_files[0]))
        test_mask = np.load(os.path.join(val_mask_dir, val_files[0].replace('image_', 'mask_')))
        test_mask_argmax = np.argmax(test_mask, axis=3)

        pred = model.predict(np.expand_dims(test_img, axis=0))
        pred_argmax = np.argmax(pred, axis=4)[0]

        slice_idx = 55
        plt.figure(figsize=(15, 5))
        plt.subplot(131)
        plt.title('FLAIR Image Slice')
        plt.imshow(test_img[:, :, slice_idx, 0], cmap='gray')
        plt.subplot(132)
        plt.title('Ground Truth Mask')
        plt.imshow(test_mask_argmax[:, :, slice_idx])
        plt.subplot(133)
        plt.title('Predicted Segmentation')
        plt.imshow(pred_argmax[:, :, slice_idx])
        plt.show()

if __name__ == '__main__':
    print("Script template ready. Run individual sections in Google Colab cells.")
