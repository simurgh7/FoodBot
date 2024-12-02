import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from transformers import DeiTForImageClassification
from torch.optim import Adam
from sklearn import metrics

def main():
    # Define image transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize images to fit the ViT input size
        transforms.ToTensor(),  # Convert images to tensors
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalization for ViT
    ])

    # Define the path to your dataset
    train_dir = r'/media/aanisha/manisha/food/dataset/train'
    val_dir = r'/media/aanisha/manisha/food/dataset/val'
    test_dir = r'/media/aanisha/manisha/food/dataset/test'

    # Use ImageFolder to load images and labels from directory structure
    train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
    val_dataset = datasets.ImageFolder(root=val_dir, transform=transform)
    test_dataset = datasets.ImageFolder(root=test_dir, transform=transform)

    # DataLoader for batch processing
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Load pretrained DeiT model with 10 output classes (your case)
    model = DeiTForImageClassification.from_pretrained(
    'facebook/deit-base-distilled-patch16-224', 
    num_labels=10
    )
    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Optimizer
    optimizer = Adam(model.parameters(), lr=1e-4)

    # Learning rate scheduler (optional)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # Set device (GPU if available, else CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Move model to the device
    model.to(device)

    num_epochs = 10

    for epoch_no in range(num_epochs):
        # Train the model
        train_per_epoch(model, train_loader, criterion, optimizer, scheduler, device, epoch_no)
        # Evaluate after training
        validate_per_epoch(model, val_loader, device, epoch_no, criterion)
    # Test the model
    test(model, test_loader, device)

    torch.save(model.state_dict(), r'../weight/deit-base-distilled-patch16-224.pth')

def train_per_epoch(model, train_loader, criterion, optimizer, scheduler, device, epoch_no):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
        
    for inputs, labels in train_loader:
        # Move to GPU if available
        inputs, labels = inputs.to(device), labels.to(device)
            
        optimizer.zero_grad()
            
        # Forward pass
        outputs = model(inputs).logits
        loss = criterion(outputs, labels)
            
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
            
        running_loss += loss.item()
            
        # Calculate accuracy
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
    # Update the learning rate scheduler
    scheduler.step()
        
    # Print loss and accuracy for this epoch
    print(f'Epoch: {epoch_no+1}, Training Loss: {running_loss/len(train_loader):.4f}, Training Accuracy: {100*correct/total:.2f}%')


def validate_per_epoch(model, val_loader, device, epoch_no, criterion):
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0
    val_loss = 0  # Variable to accumulate the loss
    
    with torch.no_grad():  # Disable gradient calculation for validation
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(inputs).logits  # assuming the model outputs a logits tensor
            
            # Compute the loss (e.g., CrossEntropyLoss)
            loss = criterion(outputs, labels)
            val_loss += loss.item()  # Accumulate the loss
            
            # Get the predictions
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    
    # Print loss and accuracy for this epoch
    print(f'Epoch: {epoch_no+1}, Validation Loss: {val_loss/len(val_loader):.4f}, Validation Accuracy: {100*correct/total:.2f}%')

  

def test(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs).logits
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')
    
    # Optionally, you can calculate precision, recall, F1-score
    print("Classification Report:")
    print(metrics.classification_report(all_labels, all_preds))

if __name__=="__main__":
    main()
