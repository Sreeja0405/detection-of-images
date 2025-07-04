import os
import requests

# ==============================
# 🔧 CONFIGURATION
API_KEY = "JrzBcXWInEln3un7MaepsjETfaqcoDyXdZNNjvycp4u8x0ZNvlVxBXjk"
  # 👈 Replace this with your API key
QUERY = "landscape, people, nature, street, city, food"  # Real-world subjects
NUM_IMAGES = 500
SAVE_DIR = "C:/Users/kurapati sai sreeja/OneDrive/Desktop/aiVShuman10/aiVShuman/aiVShuman/backend/real_folder"
PER_PAGE = 80  # Max per request
# ==============================

os.makedirs(SAVE_DIR, exist_ok=True)

def download_real_images():
    print("📷 Downloading real images from Pexels...")
    page = 1
    count = 0

    while count < NUM_IMAGES:
        url = f"https://api.pexels.com/v1/search"
        headers = {"Authorization": API_KEY}
        params = {"query": QUERY, "per_page": PER_PAGE, "page": page}
        response = requests.get(url, headers=headers, params=params)

        if response.status_code != 200:
            print(f"❌ Error {response.status_code}: {response.text}")
            break

        data = response.json()
        photos = data.get("photos", [])
        if not photos:
            print("⚠️ No more images found.")
            break

        for photo in photos:
            if count >= NUM_IMAGES:
                break
            img_url = photo["src"].get("large2x") or photo["src"].get("original")
            try:
                img_data = requests.get(img_url, timeout=10).content
                fname = os.path.join(SAVE_DIR, f"real_{count+1:04}.jpg")
                with open(fname, "wb") as f:
                    f.write(img_data)
                print(f"✅ Saved real image {count+1}/{NUM_IMAGES}")
                count += 1
            except Exception as e:
                print(f"⚠️ Error downloading image #{count+1}: {e}")

        page += 1

    print(f"🎉 Download complete. Total images downloaded: {count}")

if __name__ == "__main__":
    download_real_images()
