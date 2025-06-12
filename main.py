import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import glob
import pickle
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import seaborn as sns
from tqdm import tqdm
import torch.optim as optim
import torch.nn.functional as F
from collections import Counter
import pandas as pd
from tabulate import tabulate
from sklearn.utils import resample
from sklearn.metrics import precision_recall_curve
import random



# Function to extract frames from a video
def extract_frames(video_path, output_dir, frame_rate=30):
    """
    Extracts frames from a video and saves them in the specified directory.

    Parameters:
        video_path (str): Path to the video file.
        output_dir (str): Directory to save the frames.
        frame_rate (int): Extract 1 frame every `frame_rate` seconds.
    """

    os.makedirs(output_dir, exist_ok=True)

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))  # Video frame rate
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Processing {video_path}: {total_frames} frames at {fps} FPS.")

    # Frame extraction
    frame_count = 0
    saved_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Save one frame every `frame_rate` seconds
        if frame_count % (fps / frame_rate) == 0:
            frame_name = os.path.join(output_dir, f"frame_{saved_count:05d}.jpg")
            cv2.imwrite(frame_name, frame)
            saved_count += 1
        frame_count += 1

    cap.release()
    print(f"Saved {saved_count} frames to {output_dir}")
    


# # Example usage:
# # frame_dir = "path/to/your/video_directory"
# video_file = "Arrest030_x264.mp4"   # Replace with your video file name
# video_path = os.path.join(videosPath, video_file)
# output_dir = os.path.join(framesPath, video_file.split(".")[0])
# print(output_dir)


# Preprocessing the frames for normalization and resizing
def preprocess_frame(frame_path, size=(64, 64)):
    frame = cv2.imread(frame_path)
    if frame is None:
        raise ValueError(f"Image could not be loaded. Check the path: {frame_path}")
    frame = cv2.resize(frame, size)  # Resize to model input size
    frame = frame.astype('float32')/ 255.0  # Normalize pixel values to [0, 1]
    return frame

def batch_process_frames(frames_path):
    """
    Batch process all frames for all videos in the frames directory.
    Args:
        frames_path (str): Path to the directory containing frames for all videos.
        output_dir (str): Directory to save preprocessed frames (optional).
    """
    output_dir = os.path.join("/home/rnamala/project/preprocessed_frames")
    os.makedirs(output_dir, exist_ok=True)

    # Loop through each video folder
    for frame_folder in os.listdir(frames_path):
        frame_dir = os.path.join(frames_path, frame_folder)
        # print(frame_dir)
        # if not os.path.isdir(frame_dir):
        # print(f"Processing video: {frame_folder}")
        
        # Save preprocessed frames to output directory
        frame_output_dir = os.path.join(output_dir , frame_folder)
        # print(frame_output_dir)
        if not os.path.exists(frame_output_dir):
            # print(frame_output_dir)
            os.makedirs(frame_output_dir, exist_ok=True)
            
            # Get all frames for this video
            frames = sorted(glob.glob(f"{frame_dir}/*.jpg"))
            if len(frames) == 0:
                print(f"No frames found in: {frame_dir}")
                continue

            # Preprocess all frames
            preprocessed_frames = [preprocess_frame(f) for f in frames]

            for i, frame in enumerate(preprocessed_frames):
                output_path = os.path.join(frame_output_dir, f"frame_{i:05d}.jpg")
                cv2.imwrite(output_path, (frame * 255).astype("uint8"))  # Save normalized frame back to disk

            print(f"Processed {len(preprocessed_frames)} frames for {frame_folder}")


# create a sequence of frames for spatio-temporal models
def create_sequences(frame_dir, sequence_length=16):
    """
    Create sequences of frames for spatio-temporal models.

    Parameters:
        frame_dir (str): Path to the directory containing extracted frames.
        sequence_length (int): Number of frames per sequence.

    Returns:
        list: A list of sequences, each sequence is a list of normalized frames.
    """
    frames = sorted(glob.glob(f"{frame_dir}/*.jpg"))  # Sort frames
    sequences = []
    for i in range(0, len(frames) - sequence_length + 1, sequence_length):
        sequence = [cv2.imread(f) / 255.0 for f in frames[i:i + sequence_length]]
        sequences.append(np.stack(sequence, axis=0))  # Stack frames into one array
    return sequences



# Example usage for creating sequences
# frame_dir = framesPath + "/Abuse028_x264"
# sequences = create_sequences(frame_dir)
# print(f"Number of sequences: {len(sequences)}")


def batch_process_frames_to_sequences(input_dir, output_dir, sequence_length=16):
    """
    Batch process all frame folders to generate and save sequences.

    Parameters:
        input_dir (str): Path to the directory containing all frame folders.
        output_dir (str): Path to the directory where sequences will be saved.
        sequence_length (int): Number of frames per sequence.
    """
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)  

    for frame_name in os.listdir(input_dir):
        frame_dir = os.path.join(input_dir, frame_name)
        if os.path.isdir(frame_dir):  # Process only directories
            # Save sequences to a pickle file
            output_file = os.path.join(output_dir, f"{frame_name}.pkl")
            if not os.path.exists(output_file):
                print(f"Processing: {frame_name}")

                # Convert frames to sequences
                sequence = create_sequences(frame_dir, sequence_length=sequence_length)
          
                with open(output_file, "wb") as f:
                    pickle.dump(sequence, f)

                print(f"Saved {len(sequences)} sequences for {frame_name} to {output_file}")
                return sequence



def parse_annotation_file(annotation_file):
    """
    Parse annotations to get anomaly ranges for each video.
    """
    annotations = {}
    with open(annotation_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            video_name = parts[0]
            start_frame, end_frame = int(parts[2]), int(parts[3])
            annotations[video_name] = (start_frame, end_frame)
    return annotations

# Load annotations
annotation_file = "video_anomaly_detection_ui/annotations/Temporal_Anomaly_Annotation_for_Testing_Videos.txt"
# print(annotations)


def label_sequences_for_all_videos(sequence_dir, annotations, output_dir, sequence_length=16):
    """
    Correctly label existing sequences based on annotations.

    Parameters:
        sequence_dir (str): Directory containing pre-generated sequence files (.pkl).
        annotations (dict): Dictionary with video names as keys and (start_frame, end_frame) as values.
        output_dir (str): Directory to save labeled sequences with updated labels.
        sequence_length (int): Number of frames per sequence.
    """
    os.makedirs(output_dir, exist_ok=True)

    for sequence_file in os.listdir(sequence_dir):
        if not sequence_file.endswith(".pkl"):
            continue  # Skip non-pickle files

        video_name = os.path.splitext(sequence_file)[0] + ".mp4"  # Match video name in annotations
        print(f"Processing: {video_name}")

        # Load pre-generated sequences
        with open(os.path.join(sequence_dir, sequence_file), "rb") as f:
          sequences = pickle.load(f)  # Load sequences only
          labels = []

        num_sequences = len(sequences)
        sequence_labels = []

        # Generate labels based on annotations
        if video_name in annotations:
            start_frame, end_frame = annotations[video_name]
            print(f"Anomaly in frames {start_frame} to {end_frame}")

            for i in range(num_sequences):
                sequence_start = i * sequence_length
                sequence_end = sequence_start + sequence_length - 1

                # Label as 1 if any frame in the sequence falls within anomaly range
                if max(sequence_start, start_frame) <= min(sequence_end, end_frame):
                    sequence_labels.append(1)
                else:
                    sequence_labels.append(0)
        else:
            # If no anomaly, all labels are 0
            sequence_labels = [0] * num_sequences

        # Save updated sequences and labels
        output_file = os.path.join(output_dir, sequence_file)
        with open(output_file, "wb") as f:
            pickle.dump((sequences, sequence_labels), f)

        print(f"Updated {num_sequences} sequences for {video_name} ")
            #   and sequence of labels are {sequence_labels}")            


def combine_labeled_sequences(labeled_dir):
    """
    Combine labeled sequences from all videos into a single dataset.

    Parameters:
        labeled_dir (str): Directory containing labeled sequence files.

    Returns:
        list, list: Combined sequences and labels.
    """
    all_sequences = []
    all_labels = []

    for file in os.listdir(labeled_dir):
        if file.endswith(".pkl"):
            file_path = os.path.join(labeled_dir, file)
            with open(file_path, "rb") as f:
                sequences, labels = pickle.load(f)
                all_sequences.extend(sequences)
                all_labels.extend(labels)

    return all_sequences, all_labels




class VideoSequenceDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences  # List of sequences
        self.labels = labels        # Corresponding labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = torch.tensor(self.sequences[idx], dtype=torch.float32)  # [sequence_length, height, width, channels]
        # Reorder dimensions to [channels, sequence_length, height, width]
        sequence = sequence.permute(3, 0, 1, 2)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return sequence, label




class STCN(nn.Module):
    def __init__(self, input_channels=3, sequence_length=16, height=128, width=128):
        super(STCN, self).__init__()

        self.conv1 = nn.Conv3d(input_channels, 32, kernel_size=(3, 3, 3), stride=1, padding=1)
        self.pool1 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=2)

        self.conv2 = nn.Conv3d(32, 64, kernel_size=(3, 3, 3), stride=1, padding=1)
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=2)

        self.conv3 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), stride=1, padding=1)
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=2)
        
        self.conv4 = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), stride=1, padding=1)
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=2)

        # Dynamically calculate the flattened size
        with torch.no_grad():
            dummy_input = torch.zeros((1, input_channels, sequence_length, height, width))
            dummy_output = self.pool4(self.conv4(self.pool3(self.conv3(self.pool2(self.conv2(self.pool1(self.conv1(dummy_input))))))))
            # dummy_output = (self.pool3(self.conv3(self.pool2(self.conv2(self.pool1(self.conv1(dummy_input)))))))
            self.flattened_size = dummy_output.numel()

        self.fc1 = nn.Linear(self.flattened_size, 512)
        self.fc2 = nn.Linear(512, 1)
        #self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        # print(f"After Conv1 + Pool1: {x.shape}")  # Debugging

        x = self.pool2(F.relu(self.conv2(x)))
        # print(f"After Conv2 + Pool2: {x.shape}")
        
        x = self.pool3(F.relu(self.conv3(x)))
        # print(f"After Conv3 + Pool3: {x.shape}")
        
        x = self.pool4(F.relu(self.conv4(x)))
        # print(f"After Conv4 + Pool4: {x.shape}")

        # x = x.view(x.size(0), -1)  # Flatten
        
        x = x.reshape(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        #return self.sigmoid(x)
        return x



def train_model(model, train_loader, criterion, optimizer, num_epochs=10, device="cpu"):
    model.to(device)
    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0

        with tqdm(total=len(train_loader), desc=f"Epoch {epoch}/{num_epochs}", unit="batch") as pbar:
            for data, target in train_loader:
                data, target = data.to(device), target.to(device)

                # Flatten target to match model output
                target = target.view(-1)  # From torch.Size([8, 1]) -> torch.Size([8])

                # Forward pass
                outputs = model(data).squeeze()  # Ensure output is torch.Size([8])
                loss = criterion(outputs, target)

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
                pbar.update(1)

        print(f"Epoch {epoch}/{num_epochs} - Training Loss: {running_loss / len(train_loader):.4f}")
        # validate_model(model, val_loader, criterion, device)





def validate_model(model, val_loader, criterion, device):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():  # Disable gradient calculation
        for sequences, labels in val_loader:
            sequences, labels = sequences.to(device), labels.to(device)

            # Forward pass
            outputs = model(sequences).squeeze()  # Ensure output shape matches labels
            labels = labels.view(-1)  # Flatten labels if necessary

            # Calculate loss
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            # Convert probabilities to binary predictions (0 or 1)
            predictions = (outputs > 0.5).float()
            total_correct += (predictions == labels).sum().item()
            total_samples += labels.size(0)

    # Calculate accuracy
    accuracy = total_correct / total_samples
    print(f"Validation Loss: {total_loss / len(val_loader):.4f}, Accuracy: {accuracy:.4f}")


def test_model(model, test_loader, device="cpu"):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device).view(-1)  # Flatten target

            # Forward pass
            outputs = model(data).squeeze()  # Outputs are probabilities
            predictions = (outputs > 0.5).float()  # Convert probabilities to binary predictions

            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(target.cpu().numpy())

            correct += (predictions == target).sum().item()
            total += target.size(0)

    # Calculate metrics
    accuracy = correct / total
    print(f"Test Accuracy: {accuracy:.4f}")


def get_predictions(model, data_loader, device="cpu"):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device).view(-1)
            outputs = model(data).squeeze()
            predictions = (outputs > 0.5).float()  # Binary classification
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(target.cpu().numpy())

    return all_labels, all_preds


def infer_sequences(model, sequences, device):
    model.eval()
    preds = []
    with torch.no_grad():
        for seq in sequences:
            # seq: (T, H, W, C) floats in [0,1]
            x = torch.tensor(seq, dtype=torch.float32) \
                      .permute(3,0,1,2)            # [C,T,H,W]
            x = x.unsqueeze(0).to(device)           # [1,C,T,H,W]
            out = model(x).item()                   # scalar probability
            preds.append(1 if out > 0.5 else 0)
    return preds

def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix", labels=["Normal", "Anomalous"]):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.show()



def compute_metrics(y_true, y_pred):
    """
    Compute key metrics for classification.
    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
    Returns:
        A dictionary of metrics.
    """
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1-Score": f1_score(y_true, y_pred, zero_division=0)
    }
    return metrics

def print_classification_metrics(y_true, y_pred, dataset_name="Dataset"):
    """
    Print and return classification metrics for a dataset.
    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        dataset_name: Name of the dataset (e.g., Training or Test).
    """
    print(f"\nMetrics for {dataset_name}:\n")
    metrics = compute_metrics(y_true, y_pred)
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    print("\nClassification Report:\n")
    print(classification_report(y_true, y_pred, target_names=["Normal", "Anomalous"]))





if __name__ == "__main__":
    
    
    framesPath = 'D:\study\SEM-4\CapstoneProject\frames'
    videosPath = 'D:/project/videos/normal_video_train'
    
    # Directory to save labeled sequences
    output_dir = "D:\study\SEM-4\CapstoneProject\labeled_sequences" 
    
    
    input_dir = "D:\study\SEM-4\CapstoneProject\preprocessed_frames" # Path to all frame directories
    sequence_dir = "D:\study\SEM-4\CapstoneProject\sequences"  # Path to save sequence files
    
    # extract_frames(video_path, output_dir, frame_rate=30)
    for video_file in os.listdir(videosPath):
        video_path = os.path.join(videosPath, video_file)
        video_output = os.path.join(framesPath, video_file.split(".")[0])
        if not os.path.exists(video_output):
            extract_frames(video_path, video_output)
        
    # Print a random frame from a random folder
    try:
        frame_folders = [f for f in os.listdir(framesPath) if os.path.isdir(os.path.join(framesPath, f))]
        # print(frame_folders)
        random_folder = random.choice(frame_folders)
        print(f"Selected folder: {random_folder}")
        frame_path = framesPath + "/" + random_folder
        print(f"Selected folder: {frame_path}")
        # frame_path = framesPath 
        image_files = [f for f in os.listdir(frame_path) if f.lower().__contains__(('frame_'))]
        # print(f"Found {len(image_files)} image files in the specified folder.")
        # # print(image_files)
        if not image_files:
            print("No image files found in the specified folder.")
        random_image = random.choice(image_files)
        frame = cv2.imread(os.path.join(frame_path, random_image))
        # Display the frame
        plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.title("Sample Frame")
        plt.show()        
    except FileNotFoundError:
            print(f"Error: Folder not found: {framesPath}")
            # return None
    except Exception as e:
            print(f"An error occurred: {e}")
            #  return None
    
    # # Directory to save preprocessed frames
    batch_process_frames(framesPath)
    
    
    sequences = batch_process_frames_to_sequences(input_dir, sequence_dir, sequence_length=16)
    
    annotations = parse_annotation_file(annotation_file)
    
    
    # Label sequences based on annotations
    label_sequences_for_all_videos(sequence_dir, annotations, output_dir, sequence_length=16)
    
    
    
    # Combine all labeled sequences
    labeled_dir = "D:\study\SEM-4\CapstoneProject\labeled_sequences"
    sequences, labels = combine_labeled_sequences(labeled_dir)
    print(f"Total sequences: {len(sequences)}")
    print(f"Total labels: {len(labels)}")
    
    

    # Split data: 70% train, 30% temp (validation + test) with stratification
    train_sequences, temp_sequences, train_labels, temp_labels = train_test_split(
        sequences, labels, test_size=0.3, random_state=42, stratify=labels  # Ensure balanced splits
    )


    # Split temp data: 15% validation, 15% test (from 30% temp) with stratification
    val_sequences, test_sequences, val_labels, test_labels = train_test_split(
        temp_sequences, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels  # Ensure balanced splits
    )

    print(f"Training set: {len(train_sequences)} sequences")
    print(f"Validation set: {len(val_sequences)} sequences")
    print(f"Test set: {len(test_sequences)} sequences")



    # Oversample the minority class in the training set
    train_data = list(zip(train_sequences, train_labels))
    majority_class = [d for d in train_data if d[1] == 0]  # Normal class
    minority_class = [d for d in train_data if d[1] == 1]  # Anomalous class


    # Oversample the minority class
    oversampled_minority = resample(minority_class, replace=True, n_samples=len(majority_class), random_state=42)

    # Combine majority and oversampled minority class
    balanced_train_data = majority_class + oversampled_minority

    random.shuffle(balanced_train_data)

    # Separate sequences and labels
    train_sequences, train_labels = zip(*balanced_train_data)
   
    
    # Create datasets
    train_dataset = VideoSequenceDataset(train_sequences, train_labels)
    val_dataset = VideoSequenceDataset(val_sequences, val_labels)
    test_dataset = VideoSequenceDataset(test_sequences, test_labels)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
    
    
    # # Instantiate the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = STCN(input_channels=3, sequence_length=16, height=64, width=64).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)


    # Test a batch of data
    batch = torch.randn(8, 3, 16, 64, 64).to(device)  # Example batch with correct dimensions
    output = model(batch)
    print("Output shape:", output.shape)  # Expected: [8, 1]
    
    
    # Define model
    model = STCN(input_channels=3, sequence_length=16, height=64, width=64)

    # Define optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()


    # Train the model
    train_model(
        model=model,
        train_loader=train_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=10,
        device="cpu"  # Explicitly use CPU
    )
    
    
    validate_model(model, val_loader, criterion, device="cpu")
    
    test_model(model, test_loader, device="cpu")
    
    
    
    # Get predictions and ground truth for training set
    train_labels, train_preds = get_predictions(model, train_loader, device="cpu")

    # Plot the confusion matrix for the training set
    plot_confusion_matrix(train_labels, train_preds, title="Training Confusion Matrix")

    # Compute and print metrics for the training set
    print_classification_metrics(train_labels, train_preds, dataset_name="Training Data")

    # get metrics for training set
    train_metrics = compute_metrics(train_labels, train_preds)



    # Get predictions and ground truth for validation set
    val_labels, val_preds = get_predictions(model, val_loader, device="cpu")

    # Plot the confusion matrix for the validation set
    plot_confusion_matrix(val_labels, val_preds, title="Validation Confusion Matrix")

    # Compute and print metrics for the validation set
    print_classification_metrics(val_labels, val_preds, dataset_name="Validation Data")

    # get metrics for validation set
    val_metrics = compute_metrics(val_labels, val_preds)



    # Get predictions and ground truth for test set
    test_labels, test_preds = get_predictions(model, test_loader, device="cpu")

    # Plot the confusion matrix for the test set
    plot_confusion_matrix(test_labels, test_preds, title="Test Confusion Matrix")

    # Compute and print metrics for the test set
    print_classification_metrics(test_labels, test_preds, dataset_name="Test Data")

    # get metrics for test set
    test_metrics = compute_metrics(test_labels, test_preds)



      