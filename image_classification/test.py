import os
import json
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image
import cv2
from statenet import StateCNN  # Import StateCNN class from statenet module

classes = {0: 'creamy_paste', 1 : 'diced', 2 : 'floured', 3 : 'grated', 4: 'juiced', 5 : 'jullienne', 6 : 'mixed', 7 : 'other', 8 : 'peeled', 9 : 'sliced', 10 : 'whole'}
def test(test_data_dir, test_img_paths, model, device, transform):
    model.eval()
    predictions = {}
    with torch.no_grad():
        for path in test_img_paths:
            image = Image.open(os.path.join(test_data_dir, path))  # Use Image.open() to open the image
            image = transform(image).unsqueeze(0).to(device)  # Apply transformations and move to device
            outputs = model(image)
            _, preds = torch.max(outputs, 1)
            predicted_labels = preds.cpu().numpy().tolist()
            predictions[path] = classes[predicted_labels[0]]
    return predictions


def accuracy(outfile="output.json",gt="ground_truth.json"):
    with open(outfile, "r") as file:
        first_results = json.load(file)

    # Load the contents of the second JSON file
    with open(gt, "r") as file:
        second_results = json.load(file)

    # Compare the classification results
    same_classifications = []
    different_classifications = []

    # Iterate through the keys (image filenames) in the first JSON
    for filename, label in first_results.items():
        if filename in second_results:
            if first_results[filename] == second_results[filename]:
                same_classifications.append((filename, label))
            else:
                different_classifications.append((filename, label, second_results[filename]))
        else:
            different_classifications.append((filename, label, "Not found in second JSON"))

    # Check for any images in the second JSON not present in the first JSON
    for filename, label in second_results.items():
        if filename not in first_results:
            different_classifications.append(("Not found in first JSON", filename, label))

    # Calculate accuracy score
    total_classifications = len(first_results)
    matching_classifications = len(same_classifications)
    accuracy = (matching_classifications / total_classifications) * 100

    print(f"\nAccuracy Score: {accuracy:.2f}%")

def main():
    # Load the trained model
    model = StateCNN(num_classes=11)
    checkpoint = torch.load(r"./weight/best_model_250.pth")
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Define transformations for the test images
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Create the test dataset using ImageFolder
    test_data_dir = r"./data/test/anonymous/"

    # Use GPU if available, otherwise use CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Get the paths of test images
    test_image_paths = [file_name for file_name in os.listdir(test_data_dir) if file_name.endswith(".jpg") or file_name.endswith(".png")]

    # Get the predicted labels for test images
    predictions = test(test_data_dir, test_image_paths, model, device, transform)

    # Combine image paths with predicted labels
    results = {path: label for path, label in zip(test_image_paths, predictions.values())}

    # Save the predicted labels in a JSON file
    with open("output.json", "w") as json_file:
        json.dump(results, json_file)
    # Compare results
    accuracy("output.json","ground_truth.json")

if __name__ == "__main__":
    main()
