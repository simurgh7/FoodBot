import torch
from torchvision import transforms
from PIL import Image
import timm
from transformers import ViTForImageClassification

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the image transformations (must match training transformations)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load the saved model weights
model = ViTForImageClassification.from_pretrained(
        'google/vit-base-patch16-224-in21k',  # Pretrained ViT model
        num_labels=10  # Number of classes (10 in your case)
)
model.load_state_dict(torch.load(r'../weight/vit_16_224.pth', map_location=device))
model.to(device)
model.eval()  # Set the model to evaluation mode
# Function to predict a single image
def predict_image(image_path, model, transform, device, class_names):
    # Load and preprocess the image
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)  # Add batch dimension
    image = image.to(device)
    
    # Perform inference
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        predicted_class = class_names[predicted.item()]
    
    return predicted_class

# Define your class names (ensure these match your dataset's class order)
class_names = ['banh-chung', 'banh-cuon', 'banh-pia', 'chao-long', 'naan',
               'pho', 'pilaf', 'samosa', 'shashlyk', 'shawarma']

# Example usage
image_path = 'image.jpg'
predicted_class = predict_image(image_path, model, transform, device, class_names)
print(f"Predicted class: {predicted_class}")