import torch
from PIL import Image
from transformers import ViTForImageClassification, ViTImageProcessor

# Load a model fine-tuned for AI detection (example: "Falconsai/nsfw_image_detection")
MODEL_NAME = "Falconsai/nsfw_image_detection"  # Replace with an AI-vs-real model if available
model = ViTForImageClassification.from_pretrained(MODEL_NAME)
processor = ViTImageProcessor.from_pretrained(MODEL_NAME)

def check_if_ai_generated(image_path):
    """Check if an image is AI-generated using a local model."""
    try:
        # Open and preprocess the image
        img = Image.open(image_path).convert("RGB")
        inputs = processor(images=img, return_tensors="pt")
        
        # Run inference
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1)
        
        # Assuming class 1 = AI-generated (adjust based on model)
        ai_prob = probabilities[0][1].item()  # Probability of being AI-generated
        
        # Threshold (adjust as needed)
        is_ai = ai_prob > 0.5
        
        return {
            'ai_probability': ai_prob,
            'is_ai_generated': is_ai,
        }
    
    except Exception as e:
        return {"error": f"An error occurred: {str(e)}"}

# Example usage
if __name__ == "__main__":
    image_path = input("Enter the path to your image: ").strip('"')
    result = check_if_ai_generated(image_path)
    
    if "error" in result:
        print(f"Error: {result['error']}")
    else:
        print(f"\nAI Probability: {result['ai_probability'] * 100:.2f}%")
        print(f"Is AI-Generated? {'Yes' if result['is_ai_generated'] else 'No'}")