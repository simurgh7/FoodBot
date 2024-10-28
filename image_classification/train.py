import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score
from statenet import StateCNN
import matplotlib.pyplot as plt
import numpy as np
import gc
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"

torch.manual_seed(42)

# Define paths to train and validation data
train_data_dir = r"..\dataset\train\\"
val_data_dir = r"..\dataset\val\\"

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='sum'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    # Data Augmentation
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.RandomHorizontalFlip(), 
        transforms.RandomVerticalFlip(),  
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    # Create datasets for train and validation
    train_dataset = datasets.ImageFolder(train_data_dir, transform=transform)
    val_dataset = datasets.ImageFolder(val_data_dir, transform=transform)
    # Create data loaders for train and validation
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True) 
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    # Initialize the model created
    model = StateCNN(num_classes=5, dropout_rate=0.20, init_option="xavier")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(model)
    total_params = count_parameters(model)
    print("Total Trainable Parameters:", total_params)
    # checkpoint = torch.load("./weight/best_model00.pth")
    # model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    # start_epoch = checkpoint['epoch']
    # best_accuracy = checkpoint['accuracy']
    criterion = FocalLoss()
    optimizer = optim.RMSprop(model.parameters(), lr=0.001, momentum=0.9) 
    milestones = [45] 
    gamma = 0.1  
    scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
    # Train the model
    start_epoch = 0
    num_epochs = 50
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    best_accuracy = 0.0
    accumulation_steps = 2
    l1_lambda = 0.001
    for epoch in range(start_epoch, num_epochs):
        train_accuracy, train_loss = train(model, train_loader, criterion, optimizer, scheduler, device, accumulation_steps, l1_lambda)
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f} %")
        val_accuracy, val_loss = validate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        print(f"Epoch {epoch+1}/{num_epochs}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f} %")
        if (epoch == 0) or (val_accuracy > best_accuracy):
            # Save the model
            checkpoint_path = r'..\\weight\\best_model.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss,
                'accuracy': val_accuracy,
            }, checkpoint_path)
            best_accuracy = val_accuracy

    # Plotting train and validation losses
    epochs = range(1, num_epochs+1)
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig('..\\weight\\loss_plot.png')
    plt.clf()
    plt.plot(epochs, train_accuracies, label='Train Accuracy')
    plt.plot(epochs, val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.savefig('..\\weight\\accuracy_plot.png')

def train(model, train_loader, criterion, optimizer, scheduler, device, accumulation_steps, l1_lambda):
    # Train loop
    model.train()
    loss_log = []
    train_preds = []
    train_labels = []
    loss = 0.0
    for batch_idx, (images, labels) in enumerate(train_loader):
        torch.cuda.empty_cache()
        gc.collect()
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        #print(outputs.shape)
        _, preds = torch.max(F.softmax(outputs, dim=1), 1)
        loss = criterion(outputs, labels)
        l1_reg_loss = 0
        for param in model.parameters():
            l1_reg_loss += torch.norm(param, 1)
        # Add L1 regularization loss to the total loss
        loss += l1_lambda * l1_reg_loss
        loss_log.append(loss.item())
        loss = loss / accumulation_steps 
        loss.backward()
        if (batch_idx + 1) % accumulation_steps == 0: 
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) 
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        torch.cuda.empty_cache()
        gc.collect()
        train_preds.extend(preds.cpu().numpy())
        train_labels.extend(labels.cpu().numpy())
    accuracy = accuracy_score(train_labels, train_preds)
    epoch_loss = sum(loss_log) / len(loss_log)
    return accuracy, epoch_loss

def validate(model, val_loader, criterion, device):
    model.eval()
    val_preds = []
    val_labels = []
    val_loss = 0.0
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(val_loader):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            val_loss += criterion(outputs, labels).item()
            _, preds = torch.max(F.softmax(outputs, dim=1), 1)
            val_preds.extend(preds.cpu().numpy())
            val_labels.extend(labels.cpu().numpy())
    accuracy = accuracy_score(val_labels, val_preds)
    loss = val_loss / len(val_loader)
    return accuracy, loss

if __name__ == "__main__":
    main()
