import numpy as np
from skimage import io, color
from skimage.transform import resize
import matplotlib.pyplot as plt
from PIL import Image, ExifTags

def extract_metadata_features(image_path):
    try:
        img = Image.open(image_path)
        exif_data = img._getexif()
        if not exif_data:
            return [0] * 5  # No metadata
        tags = {ExifTags.TAGS.get(k): v for k, v in exif_data.items() if k in ExifTags.TAGS}
        
        # Extract some common metadata indicators
        camera_make = 1 if 'Make' in tags else 0
        camera_model = 1 if 'Model' in tags else 0
        iso = tags.get('ISOSpeedRatings', 0)
        exposure = tags.get('ExposureTime', 0)
        gps = 1 if 'GPSInfo' in tags else 0

        return [camera_make, camera_model, iso, exposure, gps]
    except Exception as e:
        print(f"Metadata extraction failed for {image_path}: {e}")
        return [0] * 5  # Default for failed metadata


# Function to extract FFT + Histogram features
def extract_features(image_path, num_bins=20):
    # Load image
    img = io.imread(image_path)

    if img.ndim == 3 and img.shape[2] == 4:
        img = img[..., :3]

    if img.ndim == 3:
        img = color.rgb2gray(img)

    img = resize(img, (256, 256))
    F = np.fft.fft2(img)
    Fshift = np.fft.fftshift(F)
    magnitude = np.abs(Fshift)
    log_F = np.log(1 + magnitude)

    fft_mean = np.mean(log_F)
    fft_std = np.std(log_F)
    fft_entropy = -np.sum(log_F * np.log(log_F + 1e-10))

    hist_vals, _ = np.histogram(log_F.flatten(), bins=num_bins, density=True)
    
    # Metadata
    metadata_features = extract_metadata_features(image_path)

    # Combined features
    return np.concatenate([[fft_mean, fft_std, fft_entropy], hist_vals, metadata_features])


# Example Usage
# Extract features for an example image
image_path = r"C:\Users\kurapati sai sreeja\OneDrive\Desktop\aiVShuman2\aiVShuman\aiVShuman\backend\uploads\ai image3.jpg"
features = extract_features(image_path)
print(features)