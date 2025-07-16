import joblib
import numpy as np
from skimage import io, color
from skimage.transform import resize
from skimage.feature import hog
from skimage.measure import shannon_entropy
from scipy.stats import skew, kurtosis
from PIL import Image, ExifTags
import sys
import os
from aiVShuman.featureExtraction import extract_features, extract_metadata_features

# ✅ Extract metadata features
def extract_metadata_features(image_path):
    try:
        img = Image.open(image_path)
        exif_data = img._getexif()
        if not exif_data:
            return [0] * 5
        tags = {ExifTags.TAGS.get(k): v for k, v in exif_data.items() if k in ExifTags.TAGS}
        camera_make = 1 if 'Make' in tags else 0
        camera_model = 1 if 'Model' in tags else 0
        iso = tags.get('ISOSpeedRatings', 0)
        exposure = tags.get('ExposureTime', 0)
        gps = 1 if 'GPSInfo' in tags else 0
        return [camera_make, camera_model, iso, exposure, gps]
    except:
        return [0] * 5

# ✅ Full feature extraction
def extract_features(image_path):
    try:
        img = io.imread(image_path)
        img = resize(img, (128, 128))

        if img.ndim == 2:
            img = np.stack((img,) * 3, axis=-1)
        if img.shape[2] == 4:
            img = img[:, :, :3]

        gray_img = color.rgb2gray(img)

        # HOG
        fd, _ = hog(gray_img, orientations=9, pixels_per_cell=(8, 8),
                    cells_per_block=(2, 2), visualize=True)

        # FFT
        F = np.fft.fft2(gray_img)
        Fshift = np.fft.fftshift(F)
        magnitude = np.abs(Fshift)
        log_F = np.log(1 + magnitude)

        fft_mean = np.mean(log_F)
        fft_std = np.std(log_F)
        fft_entropy = -np.sum(log_F * np.log(log_F + 1e-10))

        # Histogram
        hist, _ = np.histogram(gray_img, bins=16, range=(0, 1), density=True)

        # Stats
        img_mean = np.mean(gray_img)
        img_std = np.std(gray_img)
        img_skew = skew(gray_img.flatten())
        img_kurt = kurtosis(gray_img.flatten())

        entropy_val = shannon_entropy(gray_img)

        metadata = extract_metadata_features(image_path)

        return np.concatenate([fd, [entropy_val], [fft_mean, fft_std, fft_entropy],
                               hist, [img_mean, img_std, img_skew, img_kurt], metadata])
    except Exception as e:
        print(f"Feature extraction failed: {e}")
        return None

# ✅ Predict using model
def predict(image_path):
    model = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")

    features = extract_features(image_path)
    if features is None:
        print("Feature extraction failed.")
        return

    features_scaled = scaler.transform([features])
    pred = model.predict(features_scaled)[0]
    prob = model.predict_proba(features_scaled)[0]

    # 🟢 0 = Real, 🔴 1 = AI
    if pred == 0:
        result = "Real"
    else:
        result = "AI"

    ai_confidence = prob[1] * 100
    real_confidence = prob[0] * 100

    # 👀 Optional logs for debugging
    print(f"\nPrediction for '{os.path.basename(image_path)}': {result}")
    print(f"Confidence: Real={real_confidence:.2f}%, AI={ai_confidence:.2f}%")

    # ✅ Final output for backend (must be last line)
    print(f"{result},{ai_confidence:.2f},{real_confidence:.2f}")

# ✅ CLI entry point
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python predict.py <image_path>")
    else:
        image_path = sys.argv[1]
        if not os.path.exists(image_path):
            print("File not found.")
        else:
            predict(image_path)
