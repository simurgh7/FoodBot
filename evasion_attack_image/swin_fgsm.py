import timm
from transformers import AutoImageProcessor
import torch.nn as nn
import torch.optim as optim
from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import FastGradientMethod
from PIL import Image
import numpy as np
import os
from transformers import SwinForImageClassification
import torch

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the model
model_path = r"../weight/swin-patch4-window7-224.pth"
model = SwinForImageClassification.from_pretrained(
    'microsoft/swin-base-patch4-window7-224',
    num_labels=10,  # Set the number of output classes
    ignore_mismatched_sizes=True  # Ignore size mismatch for the classifier layer
)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()  # Set the model to evaluation mode

# Define a wrapper for Hugging Face model
class HuggingFaceModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        # Convert numpy input to tensor if necessary
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, device=device)
        outputs = self.model(pixel_values=x)
        return outputs.logits  # Return raw logits

# Wrap the Hugging Face model
wrapped_model = HuggingFaceModelWrapper(model)

# Define loss and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

# Wrap the model with ART's PyTorchClassifier
classifier = PyTorchClassifier(
    model=wrapped_model,
    loss=loss_fn,
    optimizer=optimizer,
    input_shape=(3, 224, 224),  # Example input shape for Swin models
    nb_classes=10  # Number of output classes for the model
)

# Load the image processor for the Swin model
processor = AutoImageProcessor.from_pretrained("microsoft/swin-base-patch4-window7-224")

# Define directories for clean and adversarial images
data_dir = r"../../Data/inference"
out_dir = r"../../Data/evasion"

# Process each image
for files in os.listdir(data_dir):
    image_path = os.path.join(data_dir, files)

    # Load and preprocess the image
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="np")  # Preprocess for ART
    x = inputs["pixel_values"]  # This will have shape (1, 3, 224, 224)

    # Create the FGSM attack
    attack = FastGradientMethod(estimator=classifier, eps=0.1)

    # Generate adversarial examples
    x_adv = attack.generate(x=x)

    print("Adversarial example shape:", x_adv.shape)

    # Convert the adversarial example back to an image
    x_adv_image = x_adv[0].transpose(1, 2, 0)  # Change shape from (3, 224, 224) to (224, 224, 3)
    x_adv_image = np.clip(x_adv_image * 255, 0, 255).astype(np.uint8)  # De-normalize and clip

    # Save or display the adversarial image
    adv_image = Image.fromarray(x_adv_image)
    out_path = os.path.join(out_dir, files)
    adv_image.save(out_path)
    adv_image.show()
