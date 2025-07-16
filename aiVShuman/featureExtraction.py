import sys
import numpy as np
from skimage import io, color
from skimage.transform import resize
from skimage.feature import hog
from PIL import Image, ExifTags
from scipy.stats import skew, kurtosis

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
    except Exception as e:
        print(f"Metadata extraction failed: {e}", file=sys.stderr)
        return [0] * 5

def extract_features(image_path, num_bins=20):
    # Load and preprocess image
    img = io.imread(image_path)
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[..., :3]  # remove alpha channel
    if img.ndim == 3:
        img = color.rgb2gray(img)
    img = resize(img, (256, 256))

    # FFT features
    F = np.fft.fft2(img)
    Fshift = np.fft.fftshift(F)
    magnitude = np.abs(Fshift)
    log_F = np.log(1 + magnitude)
    fft_mean = np.mean(log_F)
    fft_std = np.std(log_F)
    fft_entropy = -np.sum(log_F * np.log(log_F + 1e-10))
    fft_skewness = skew(log_F.flatten())
    fft_kurtosis = kurtosis(log_F.flatten())

    # Histogram from FFT
    hist_vals, _ = np.histogram(log_F.flatten(), bins=num_bins, density=True)

    # HOG features
    hog_feats = hog(img, orientations=9, pixels_per_cell=(8, 8),
                    cells_per_block=(2, 2), block_norm='L2-Hys', feature_vector=True)
    hog_feats = hog_feats[:50]  # Take first 50 elements

    # EXIF Metadata
    metadata_features = extract_metadata_features(image_path)

    # Final feature vector
    features = np.concatenate([[fft_mean, fft_std, fft_entropy], hist_vals, hog_feats, metadata_features])
    
    # Output as comma-separated values
    print(','.join(map(str, features)))

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python features.py <image_path>", file=sys.stderr)
        sys.exit(1)

    image_path = sys.argv[1]
    extract_features(image_path)
