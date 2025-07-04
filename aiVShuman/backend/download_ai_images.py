import os
import requests
from PIL import Image
from io import BytesIO

# === CONFIG ===
NUM_IMAGES = 500
SAVE_FOLDER = 'C:/Users/kurapati sai sreeja/OneDrive/Desktop/aiVShuman10/aiVShuman/aiVShuman/backend/ai_folder'
URL = 'https://thispersondoesnotexist.com'

# === Ensure folder exists ===
os.makedirs(SAVE_FOLDER, exist_ok=True)

# === Download images ===
for i in range(NUM_IMAGES):
    try:
        response = requests.get(URL, timeout=5)
        if response.status_code == 200:
            img = Image.open(BytesIO(response.content)).convert("RGB")
            img = img.resize((224, 224))  # Resize to model standard
            img.save(os.path.join(SAVE_FOLDER, f"ai_{i+1:04}.jpg"))
            print(f"✅ Saved AI image {i+1}/{NUM_IMAGES}")
        else:
            print(f"⚠️ Failed to download image {i+1}")
    except Exception as e:
        print(f"❌ Error at image {i+1}: {e}")
