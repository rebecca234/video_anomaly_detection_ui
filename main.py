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




if __name__ == "__main__":
    
    
    framesPath = 'video_anomaly_detection_ui\frames'  
    videosPath = 'video_anomaly_detection_ui/normal_video_train'
    
    # Directory to save labeled sequences
    output_dir = "video_anomaly_detection_ui\labeled_sequences" 
    
    
    input_dir = "video_anomaly_detection_ui\preprocessed_frames" # Path to all frame directories
    sequence_dir = "video_anomaly_detection_ui\sequences"  # Path to save sequence files
    
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
    labeled_dir = "video_anomaly_detection_ui\labeled_sequences"
    sequences, labels = combine_labeled_sequences(labeled_dir)
    print(f"Total sequences: {len(sequences)}")
    print(f"Total labels: {len(labels)}")
    
