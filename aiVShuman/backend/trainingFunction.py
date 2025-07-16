import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from skimage import io, color
from skimage.transform import resize
from skimage.feature import hog
from skimage.measure import shannon_entropy
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from scipy.stats import skew, kurtosis
from PIL import Image, ExifTags
import joblib
import warnings

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model

warnings.filterwarnings("ignore")

# 📁 Folder paths
AI_FOLDER = 'C:/Users/kurapati sai sreeja/Desktop/aiVShuman40/aiVShuman/backend/ai_folder'
REAL_FOLDER = 'C:/Users/kurapati sai sreeja/Desktop/aiVShuman40/aiVShuman/backend/real_folder'
SUPPORTED_EXTENSIONS = ('.jpg', '.jpeg', '.png')

# 🧠 Load ResNet50 for deep features
base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
resnet_model = Model(inputs=base_model.input, outputs=base_model.output)

# ✅ Extract ResNet deep features
def extract_deep_features(img_path):
    try:
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        features = resnet_model.predict(img_array, verbose=0)
        return features.flatten()
    except Exception as e:
        print(f"[ERROR][ResNet] {img_path}: {e}")
        return None

# ✅ Extract metadata (EXIF)
def extract_metadata_features(image_path):
    try:
        img = Image.open(image_path)
        exif_data = img._getexif()
        if not exif_data:
            return [0] * 5
        tags = {ExifTags.TAGS.get(k): v for k, v in exif_data.items() if k in ExifTags.TAGS}
        return [
            1 if 'Make' in tags else 0,
            1 if 'Model' in tags else 0,
            tags.get('ISOSpeedRatings', 0),
            tags.get('ExposureTime', 0),
            1 if 'GPSInfo' in tags else 0
        ]
    except:
        return [0] * 5

# ✅ Extract handcrafted features (HOG, FFT, histogram, stats, entropy)
def extract_handcrafted_features(image_path):
    try:
        img = io.imread(image_path)
        img = resize(img, (128, 128))

        if img.ndim == 2:
            img = np.stack((img,) * 3, axis=-1)
        if img.shape[2] == 4:
            img = img[:, :, :3]

        gray = color.rgb2gray(img)

        # HOG
        hog_feat = hog(gray, orientations=9, pixels_per_cell=(8, 8),
                       cells_per_block=(2, 2), block_norm='L2-Hys', feature_vector=True)
        hog_feat = hog_feat[:50]  # Limit to 50 dims

        # FFT
        fft = np.fft.fft2(gray)
        fft_shift = np.fft.fftshift(fft)
        fft_mag = np.abs(fft_shift)
        log_fft = np.log(1 + fft_mag)
        fft_mean = np.mean(log_fft)
        fft_std = np.std(log_fft)
        fft_entropy = -np.sum(log_fft * np.log(log_fft + 1e-10))

        # Histogram
        hist, _ = np.histogram(gray, bins=16, range=(0, 1), density=True)

        # Stats
        flat = gray.flatten()
        stats = [np.mean(flat), np.std(flat), skew(flat), kurtosis(flat)]

        # Entropy
        entropy = shannon_entropy(gray)

        return np.concatenate([hog_feat, [entropy, fft_mean, fft_std, fft_entropy], hist, stats])
    except Exception as e:
        print(f"[ERROR][Handcrafted] {image_path}: {e}")
        return None

# ✅ Combine all features
def extract_all_features(image_path):
    handcrafted = extract_handcrafted_features(image_path)
    metadata = extract_metadata_features(image_path)
    deep = extract_deep_features(image_path)
    if handcrafted is not None and deep is not None:
        return np.concatenate([deep, handcrafted, metadata])
    else:
        return None

# ✅ Load dataset
def load_data():
    X, y = [], []
    ai_count = real_count = 0

    for fname in os.listdir(AI_FOLDER):
        if fname.lower().endswith(SUPPORTED_EXTENSIONS):
            fpath = os.path.join(AI_FOLDER, fname)
            feat = extract_all_features(fpath)
            if feat is not None:
                X.append(feat)
                y.append(1)
                ai_count += 1

    for fname in os.listdir(REAL_FOLDER):
        if fname.lower().endswith(SUPPORTED_EXTENSIONS):
            fpath = os.path.join(REAL_FOLDER, fname)
            feat = extract_all_features(fpath)
            if feat is not None:
                X.append(feat)
                y.append(0)
                real_count += 1

    print(f"📊 AI images: {ai_count}, Real images: {real_count}")
    print(f"✅ Total loaded: {len(X)} samples")
    return np.array(X), np.array(y)

# ✅ Train and evaluate model
def train_model():
    print("🔄 Extracting features and loading data...")
    X, y = load_data()

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Train model
    clf = GradientBoostingClassifier(n_estimators=250, learning_rate=0.1, max_depth=4, random_state=42)
    clf.fit(X_train, y_train)

    # Evaluate
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n🎯 Accuracy: {acc * 100:.2f}%")
    print("📄 Classification Report:\n", classification_report(y_test, y_pred, target_names=["Real", "AI"]))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Real", "AI"], yticklabels=["Real", "AI"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()

    # Save model & scaler
    joblib.dump(clf, 'model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    print("✅ Model and scaler saved successfully")

# ✅ Run if main
if __name__ == '__main__':
    train_model()
