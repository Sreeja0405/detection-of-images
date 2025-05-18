import os
import numpy as np
from skimage import io, feature, color
from skimage.transform import resize
from skimage.feature import hog
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from skimage.measure import shannon_entropy
import cv2

# Define the folder paths
ai_folder = 'C:/Users/kurapati sai sreeja/OneDrive/Desktop/aiVShuman10/aiVShuman/aiVShuman/backend/ai_folder'
real_folder = 'C:/Users/kurapati sai sreeja/OneDrive/Desktop/aiVShuman10/aiVShuman/aiVShuman/backend/real_folder'

# Function to extract features from an image
def extract_features(image_path):
    try:
        img = io.imread(image_path)
        img = resize(img, (128, 128))  # Resize image

        # Handle grayscale images by converting to RGB
        if len(img.shape) == 2:
            img = np.stack((img,) * 3, axis=-1)

        # Handle RGBA images by discarding the alpha channel
        if img.shape[2] == 4:
            img = img[:, :, :3]

        # Convert to grayscale
        gray_img = color.rgb2gray(img)

        # Extract HOG features
        fd, _ = hog(gray_img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)

        # Extract entropy
        entropy = shannon_entropy(gray_img)

        # Combine features
        features = np.append(fd, entropy)

        return features
    except Exception as e:
        print(f"Error extracting features from {image_path}: {e}")
        return None


# Load the data from both folders
def load_data():
    X = []  # Features
    y = []  # Labels (0 for real, 1 for AI)

    # Load AI images
    for filename in os.listdir(ai_folder):
        image_path = os.path.join(ai_folder, filename)
        features = extract_features(image_path)
        if features is not None:
            X.append(features)
            y.append(1)  # Label for AI-generated images

    # Load real images
    for filename in os.listdir(real_folder):
        image_path = os.path.join(real_folder, filename)
        features = extract_features(image_path)
        if features is not None:
            X.append(features)
            y.append(0)  # Label for real images

    # Convert lists to numpy arrays
    X = np.array(X)
    y = np.array(y)

    return X, y

# Train the model
def train_model():
    X, y = load_data()

    # Split data into training and testing sets (80% training, 20% testing)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize the classifier (Random Forest)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)

    # Train the classifier
    clf.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = clf.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")

    return clf

# Main function to run training
if __name__ == '__main__':
    model = train_model()
