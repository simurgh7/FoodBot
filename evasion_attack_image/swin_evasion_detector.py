import torch
from torchvision import transforms
from transformers import SwinForImageClassification
from PIL import Image
from art.estimators.classification import PyTorchClassifier
from art.defences.detector.evasion import BinaryInputDetector
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np
import os

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load pre-trained Swin Transformer model with number of output labels set to 10
model = SwinForImageClassification.from_pretrained(
    'microsoft/swin-base-patch4-window7-224',
    num_labels=10,  # Set the number of output labels for your dataset
    ignore_mismatched_sizes=True  # Ignore the size mismatch for the classifier layer
)

# Load the custom weights (the pre-trained weights)
model.load_state_dict(torch.load(r'../weight/swin-patch4-window7-224.pth', map_location=device))
model.to(device)
model.eval()  # Set the model to evaluation mode

# ART PyTorchClassifier Wrapper
from torch.nn import CrossEntropyLoss
from torch.optim import Adam

# Loss and optimizer for the ART wrapper
loss = CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.001)

# Wrap model with ART PyTorchClassifier
classifier = PyTorchClassifier(
    model=model,
    loss=loss,
    optimizer=optimizer,
    input_shape=(3, 224, 224),
    nb_classes=10,
)

# Image transformation pipeline
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Dataset directories (assuming clean images and adversarial examples are already in separate directories)
clean_data_dir = r"../../Data/inference"  # Folder containing clean images
adversarial_data_dir = r"../../Data/evasion-fgsm"  # Folder containing adversarial images

# Extract features for normal and adversarial samples
def extract_features(data_dir, classifier, transform, device, is_adversarial=False):
    features, labels = [], []
    for files in os.listdir(data_dir):
        image_path = os.path.join(data_dir, files)
        gt_label = files.split('_')[0]  # Assuming file names have ground truth like 'class_name_123.jpg'
        image = Image.open(image_path).convert('RGB')
        image = transform(image).unsqueeze(0).to(device)
        
        # Get feature representation
        with torch.no_grad():
            outputs = classifier.model(image)
            logits = outputs.logits.cpu().numpy()
        
        features.append(logits.flatten())  # Use logits as features
        labels.append(0 if is_adversarial else 1)  # Label: 0 for adversarial, 1 for normal
    
    return np.array(features), np.array(labels)

# Extract features from clean and adversarial data
clean_features, clean_labels = extract_features(clean_data_dir, classifier, transform, device, is_adversarial=False)
adv_features, adv_labels = extract_features(adversarial_data_dir, classifier, transform, device, is_adversarial=True)

# Combine features and labels
features = np.concatenate([clean_features, adv_features])
labels = np.concatenate([clean_labels, adv_labels])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Train an evasion detector using RandomForest
detector_model = RandomForestClassifier()
detector_model.fit(X_train, y_train)

# Evaluate the evasion detector
y_pred = detector_model.predict(X_test)
print("Classification Report for Evasion Detector:")
print(classification_report(y_test, y_pred))

# Wrap the detector with ART BinaryInputDetector
detector = BinaryInputDetector(detector_model)
print("Evasion detector trained and ready.")

# Define function to extract features from an image for prediction
def extract_features_for_prediction(image_path, classifier, transform, device):
    """Extract feature vector from an image for prediction"""
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)  # Add batch dimension
    with torch.no_grad():
        outputs = classifier.model(image)  # Get model outputs
        logits = outputs.logits.cpu().numpy()  # Get logits
    return logits.flatten()  # Flatten the logits to match the feature space

# Function to predict if an image is legitimate or adversarial
def predict_image(image_path, classifier, detector, transform, device):
    """Predict if an image is legitimate or adversarial"""
    # Step 1: Extract features from the image
    features = extract_features_for_prediction(image_path, classifier, transform, device)

    # Step 2: Use the evasion detector (RandomForest) to classify
    prediction = detector.predict([features])  # Classify the image

    # Step 3: Return the prediction result
    if prediction == 1:
        print(f"Image '{image_path}' is legit.")
    

for file in os.listdir(adversarial_data_dir):
    # Example usage: Test the prediction on an image
    image_path = os.path.join(adversarial_data_dir, file)  # Path to the image you want to test
    predict_image(image_path, classifier, detector_model, transform, device)
