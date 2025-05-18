import joblib
import numpy as np
from skimage import io, color
from skimage.transform import resize
from scipy.stats import entropy
import sys
import os

def extract_fft_features(image_path):
    try:
        img = io.imread(image_path, as_gray=True)
        img_resized = resize(img, (128, 128), anti_aliasing=True)

        fft_result = np.fft.fft2(img_resized)
        fft_magnitude = np.abs(fft_result)

        mean_val = np.mean(fft_magnitude)
        std_val = np.std(fft_magnitude)
        entropy_val = entropy(np.histogram(fft_magnitude.flatten(), bins=50)[0] + 1e-6)

        hist, _ = np.histogram(fft_magnitude, bins=20)
        hist = hist / np.sum(hist)

        return np.concatenate(([mean_val, std_val, entropy_val], hist))
    except Exception as e:
        print(f"Error extracting features: {e}")
        return None

def predict(image_path):
    # Load the trained model
    model = joblib.load("models/model.pkl")
    
    # Extract features
    features = extract_fft_features(image_path)
    if features is None:
        print("Feature extraction failed. Aborting.")
        return
    
    features = features.reshape(1, -1)  # reshape for prediction
    prediction = model.predict(features)[0]
    
    label = "Real Image" if prediction == 1 else "AI-Generated Image"
    print(f"Prediction: {label}")

# Run only if called from command line
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python predict.py <image_path>")
    else:
        image_path = sys.argv[1]
        if not os.path.exists(image_path):
            print("Image file not found.")
        else:
            predict(image_path)
