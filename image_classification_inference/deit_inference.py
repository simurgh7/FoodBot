import torch
from torchvision import transforms
from transformers import DeiTForImageClassification
from PIL import Image
import os

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the image transformations (must match training transformations)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load the pre-trained model and adjust it for your task
model = DeiTForImageClassification.from_pretrained(
    'facebook/deit-base-distilled-patch16-224', 
    num_labels=10  # Match the number of output classes to your dataset
)
# Load saved weights into the model
model.load_state_dict(torch.load(r'../weight/deit-16-224.pth', map_location=device))

# Move model to the appropriate device and set to evaluation mode
model.to(device)
model.eval()

# Function to predict a single image
def predict_image(image_path, model, transform, device, class_names):
    # Load and preprocess the image
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)  # Add batch dimension
    image = image.to(device)
    
    # Perform inference
    with torch.no_grad():
        outputs = model(image)
        logits = outputs.logits  # Extract logits from the model output
        _, predicted = torch.max(logits, 1)  # Get the predicted class index
        predicted_class = class_names[predicted.item()]  # Map to class name
    
    return predicted_class

# Define your class names (ensure these match your dataset's class order)
class_names = ['banh-chung', 'banh-cuon', 'banh-pia', 'chao-long', 'naan',
               'pho', 'pilaf', 'samosa', 'shashlyk', 'shawarma']

# Directory containing images
data_dir = r"../../Data/evasion"

# Process each image
for files in os.listdir(data_dir):
    # Extract ground truth class from filename
    gt_class = files.split('_')[0]
    image_path = os.path.join(data_dir, files)
    
    # Predict the class of the image
    predicted_class = predict_image(image_path, model, transform, device, class_names)
    
    # Check for misclassification and print the result
    if gt_class != predicted_class: 
        print(f"Misclassification: {gt_class} to {predicted_class}")
