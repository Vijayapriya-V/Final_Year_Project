from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import VGG16, InceptionV3, ResNet152
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_preprocess
from tensorflow.keras.applications.inception_v3 import preprocess_input as inception_preprocess
from tensorflow.keras.applications.resnet import preprocess_input as resnet_preprocess
from tensorflow.keras.preprocessing import image
import cv2

# Dataset paths
TRAIN_IMAGE_PATH = Path("dataset/images").resolve()
TRAIN_VIDEO_PATH = Path("dataset/videos").resolve()
TEST_IMAGE_PATH = Path("test_dataset/images").resolve()
TEST_VIDEO_PATH = Path("test_dataset/videos").resolve()

# Output feature file paths
TRAIN_IMAGE_FEATURES_FILE = "train_image_features.npz"
TRAIN_VIDEO_FEATURES_FILE = "train_video_features.npz"
TEST_IMAGE_FEATURES_FILE = "test_image_features.npz"
TEST_VIDEO_FEATURES_FILE = "test_video_features.npz"

# Load CNN models (no top layers, avg pooled)
vgg_model = VGG16(weights="imagenet", include_top=False, pooling="avg")
inception_model = InceptionV3(weights="imagenet", include_top=False, pooling="avg")
resnet_model = ResNet152(weights="imagenet", include_top=False, pooling="avg")

def extract_features(img_path, model, preprocess_func):
    """Extract features from an image using a specific model."""
    try:
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_func(img_array)
        features = model.predict(img_array, verbose=0).flatten()
        return features
    except Exception as e:
        print(f"❌ Error extracting features from {img_path}: {e}")
        return None

def extract_combined_features(img_path):
    """Combine features from VGG16, InceptionV3, and ResNet152."""
    vgg_features = extract_features(img_path, vgg_model, vgg_preprocess)
    inception_features = extract_features(img_path, inception_model, inception_preprocess)
    resnet_features = extract_features(img_path, resnet_model, resnet_preprocess)

    if vgg_features is not None and inception_features is not None and resnet_features is not None:
        return np.concatenate([vgg_features, inception_features, resnet_features])
    else:
        return None

def extract_keyframes(video_path, output_folder, interval=30):
    """Extract keyframes from a video every N frames."""
    if not output_folder.exists():
        output_folder.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    keyframe_files = []

    for frame_no in range(0, frame_count, interval):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        ret, frame = cap.read()
        if not ret:
            break

        frame_filename = output_folder / f"{video_path.stem}_frame_{frame_no:04d}.jpg"
        cv2.imwrite(str(frame_filename), frame)
        keyframe_files.append(frame_filename)

    cap.release()
    return keyframe_files

def extract_all_features(image_path, video_path, output_image_file, output_video_file):
    """Extract features from images and videos, save to disk."""
    print(f"🔍 Extracting features from:\n📷 Images: {image_path}\n🎬 Videos: {video_path}")

    image_features = {}
    video_features = {}
    video_keyframes = {}

    image_files = list(image_path.rglob("*.jpg")) + list(image_path.rglob("*.jpeg")) + list(image_path.rglob("*.png"))
    video_files = list(video_path.rglob("*.mp4")) + list(video_path.rglob("*.avi"))

    print(f"✅ Found {len(image_files)} images.")
    print(f"✅ Found {len(video_files)} videos.")

    # Extract features for images
    for idx, img_path in enumerate(image_files):
        print(f"[{idx+1}/{len(image_files)}] 🖼️ Extracting features from image: {img_path.name}")
        feature_vector = extract_combined_features(img_path)
        if feature_vector is not None:
            image_features[img_path.name] = feature_vector

    # Extract features for video keyframes
    for idx, video_path in enumerate(video_files):
        print(f"[{idx+1}/{len(video_files)}] 🎥 Processing video: {video_path.name}")
        keyframe_folder = video_path.parent / f"keyframes_{video_path.stem}"
        keyframe_files = extract_keyframes(video_path, keyframe_folder)
        video_keyframes[video_path.stem] = []

        for frame_path in keyframe_files:
            feature_vector = extract_combined_features(frame_path)
            if feature_vector is not None:
                video_keyframes[video_path.stem].append(feature_vector)

        # Average keyframe features for the video
        if video_keyframes[video_path.stem]:
            video_features[video_path.stem] = np.mean(video_keyframes[video_path.stem], axis=0)

    # Save image features
    if image_features:
        np.savez_compressed(output_image_file, **image_features)
        print(f"✅ Image features saved to {output_image_file}")
    else:
        print(f"⚠️ No image features extracted.")

    # Save video features
    if video_features:
        np.savez_compressed(output_video_file, **video_features)
        print(f"✅ Video features saved to {output_video_file}")
    else:
        print(f"⚠️ No video features extracted.")

    return image_features, video_features  # Optional, useful for reuse

if __name__ == "__main__":
    print("🚀 Extracting features for training dataset...")
    extract_all_features(TRAIN_IMAGE_PATH, TRAIN_VIDEO_PATH, TRAIN_IMAGE_FEATURES_FILE, TRAIN_VIDEO_FEATURES_FILE)

    print("\n🚀 Extracting features for testing dataset...")
    extract_all_features(TEST_IMAGE_PATH, TEST_VIDEO_PATH, TEST_IMAGE_FEATURES_FILE, TEST_VIDEO_FEATURES_FILE)

    print("\n✅ Feature extraction complete!")
