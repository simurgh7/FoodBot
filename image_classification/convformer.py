import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import timm
from torch.optim import Adam
from sklearn import metrics

def main():
    # Define image transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Dataset paths
    train_dir = '/media/aanisha/manisha/food/dataset/train'
    val_dir = '/media/aanisha/manisha/food/dataset/val'
    test_dir = '/media/aanisha/manisha/food/dataset/test'

    # Load datasets
    train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
    val_dataset = datasets.ImageFolder(root=val_dir, transform=transform)
    test_dataset = datasets.ImageFolder(root=test_dir, transform=transform)

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Load ConvFormer model
    model = timm.create_model('convformer_s18', pretrained=True, num_classes=10)
    model.to(device)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=1e-4)

    # Scheduler (optional)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # Training loop
    num_epochs = 10
    for epoch_no in range(num_epochs):
        train_per_epoch(model, train_loader, criterion, optimizer, scheduler, device, epoch_no)
        validate_per_epoch(model, val_loader, device, epoch_no, criterion)
    test(model, test_loader, device)

    # Save the trained model
    torch.save(model.state_dict(), r'/media/aanisha/484C1AEF4C1AD810/Users/mehza/OneDrive - University of South Florida/study(USF)/coursework/2409-CIS6930-Trustworthy-AI-Systems/Projects/weight/convformer_s18.pth')

def train_per_epoch(model, train_loader, criterion, optimizer, scheduler, device, epoch_no):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)  # Adjust for TIMM
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    scheduler.step()
    print(f'Epoch: {epoch_no+1}, Training Loss: {running_loss/len(train_loader):.4f}, Training Accuracy: {100*correct/total:.2f}%')

def validate_per_epoch(model, val_loader, device, epoch_no, criterion):
    model.eval()
    correct = 0
    total = 0
    val_loss = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)  # Adjust for TIMM
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
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
            outputs = model(inputs)  # Adjust for TIMM
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')
    print("Classification Report:")
    print(metrics.classification_report(all_labels, all_preds))

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    main()
