import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
import dlib
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
print(torch.cuda.is_available())
print(torch.cuda.memory_allocated())
print(torch.cuda.max_memory_allocated())
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0))

# def load_video(path: str) -> torch.Tensor:
#     cap = cv2.VideoCapture(path)
#     frames = []
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
#         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         frame = frame[190:236, 80:220]
#         frames.append(frame)
#     cap.release()
#     frames = np.array(frames)
#     frames = torch.tensor(frames, dtype=torch.float32)
#     mean = frames.mean()
#     std = frames.std()
#     return (frames - mean) / std

# Initialize dlib's face detector and shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')  # Ensure you have this file

def load_video(path: str, target_height=46, target_width=140) -> torch.Tensor:
    cap = cv2.VideoCapture(path)
    frames = []
    if not cap.isOpened():  # Check if video is loaded properly
        print(f"Error: Could not open video {path}")
        return torch.tensor([])  # Return empty tensor if video cannot be opened
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = detector(gray)
        if len(faces) > 0:
            # Take the first detected face (you can adjust for multiple faces)
            face = faces[0]

            # Get facial landmarks
            landmarks = predictor(gray, face)

            # Get the coordinates of the lip region (mouth landmarks: 48-67)
            lip_points = []
            for i in range(48, 68):
                lip_points.append((landmarks.part(i).x, landmarks.part(i).y))

            # Find the bounding box for the lip region
            lip_rect = cv2.boundingRect(np.array(lip_points))

            # Crop the lip region from the frame
            lip_region = frame[lip_rect[1]:lip_rect[1]+lip_rect[3], lip_rect[0]:lip_rect[0]+lip_rect[2]]
            lip_region = cv2.cvtColor(lip_region, cv2.COLOR_BGR2GRAY)

            # Resize to ensure consistent dimensions
            lip_region_resized = cv2.resize(lip_region, (target_width, target_height))
            frames.append(lip_region_resized)
        else:
            print(f"No face detected in frame of video {path}")
    
    cap.release()

    if len(frames) == 0:
        print(f"Warning: No valid frames extracted from {path}")
        return torch.tensor([])  # Return empty tensor if no frames were extracted

    # Convert frames to a tensor
    frames = np.array(frames)
    frames = torch.tensor(frames, dtype=torch.float32)

    # Normalize frames
    mean = frames.mean()
    std = frames.std()
    return (frames - mean) / std


vocab = [x for x in "abcdefghijklmnopqrstuvwxyz'?!123456789 "]
char_to_num = {char: idx for idx, char in enumerate(vocab)}
num_to_char = {idx: char for char, idx in char_to_num.items()}

def load_alignments(path: str) -> List[int]:
    with open(path, 'r') as f:
        lines = f.readlines()
    tokens = []
    for line in lines:
        parts = line.split()
        if parts[2] != 'sil':
            tokens.extend([' ', parts[2].lower()])
    return [char_to_num[char] for char in ''.join(tokens)]

def load_data(video_path: str, alignment_path: str):
    frames = load_video(video_path)
    alignments = load_alignments(alignment_path)
    return frames, alignments

class LipNetDataset(Dataset):
    def __init__(self, file_paths: List[str], base_dir: str, target_length: int):
        self.file_paths = file_paths
        self.base_dir = base_dir
        self.target_length = target_length

    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        file_name = os.path.splitext(os.path.basename(self.file_paths[idx]))[0]
        video_path = os.path.join(self.base_dir, 'new_output_clips', f'{file_name}.mpg')
        alignment_path = os.path.join(self.base_dir, 'result_align', f'{file_name}.align')
        frames, alignments = load_data(video_path, alignment_path)
        print(f"Video path: {video_path}")
        print(f"Alignment path: {alignment_path}")
        if frames.size(0) == 0:
            raise ValueError(f"Empty frames detected for {file_name}, skipping this sample.")
    
        num_frames = frames.size(0)
        print(f"Before padding, frames shape: {frames.shape}")

        # If number of frames is less than target length, apply padding
        if num_frames < self.target_length:
            padding = self.target_length - num_frames
            # Pad on the temporal axis (first dimension)
            frames = torch.nn.functional.pad(frames, (0, 0, 0, 0, 0, padding))  
        elif num_frames > self.target_length:
            # If frames exceed the target length, truncate them to the target length
            frames = frames[:self.target_length]
    
        print(f"After padding/truncation, frames shape: {frames.shape}")

        frames = frames.unsqueeze(0)  # Add batch dimension
        return frames, torch.tensor(alignments, dtype=torch.long)


   
class LipNetModel(nn.Module):
    def __init__(self, vocab_size):
        super(LipNetModel, self).__init__()
        self.conv3d_1 = nn.Conv3d(1, 128, kernel_size=3, padding=1)
        self.pool3d_1 = nn.MaxPool3d((1, 2, 2))
        self.conv3d_2 = nn.Conv3d(128, 256, kernel_size=3, padding=1)
        self.pool3d_2 = nn.MaxPool3d((1, 2, 2))
        self.conv3d_3 = nn.Conv3d(256, 75, kernel_size=3, padding=1)
        self.pool3d_3 = nn.MaxPool3d((1, 2, 2))
        self.lstm = nn.LSTM(75 * 5 * 17, 128, num_layers=2, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(256, vocab_size + 1)

    def forward(self, x):
        x = self.pool3d_1(torch.relu(self.conv3d_1(x)))
        x = self.pool3d_2(torch.relu(self.conv3d_2(x)))
        x = self.pool3d_3(torch.relu(self.conv3d_3(x)))
        x = x.view(x.size(0), x.size(2), -1)  # Flatten for LSTM
        x, _ = self.lstm(x)
        x = self.fc(x)
        return x

def collate_fn(batch):
    frames = [item[0] for item in batch]
    alignments = [item[1] for item in batch]

    # Check if all frames are of the same shape
    frame_shapes = [frame.shape for frame in frames]
    if len(set(frame_shapes)) > 1:
        raise ValueError(f"Frame sizes are inconsistent: {frame_shapes}")

    frames_padded = torch.stack(frames)
    alignments_padded = nn.utils.rnn.pad_sequence(alignments, batch_first=True)
    return frames_padded, alignments_padded


# Compute the target length for padding/slicing (average or max length)
base_dir = '.\\'
file_paths = [os.path.join(base_dir, 'new_output_clips', f) for f in os.listdir(os.path.join(base_dir, 'new_output_clips')) if f.endswith('.mpg')]
video_lengths = []
for file_path in file_paths:
    cap = cv2.VideoCapture(file_path)
    video_lengths.append(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    cap.release()

# Set target length to the average length of videos
target_length = int(np.mean(video_lengths))
print(f"Target video length: {target_length}")

# Instantiate dataset and dataloader
dataset = LipNetDataset(file_paths, base_dir, target_length)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

# Instantiate model
model = LipNetModel(vocab_size=len(vocab)).to(device)
criterion = nn.CTCLoss(blank=len(vocab))
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Training loop
numEpoches = 10
for epoch in range(numEpoches):
    model.train()
    total_loss = 0
    # Wrap dataloader with tqdm for progress bar
    with tqdm(dataloader, desc=f"Epoch {epoch+1}/{numEpoches}", unit="batch") as pbar:
        for frames, alignments in pbar:
            frames = frames.to(device)
            alignments = alignments.to(device)

            optimizer.zero_grad()
            outputs = model(frames)
            input_lengths = torch.full(size=(outputs.size(0),), fill_value=outputs.size(1), dtype=torch.long).to(device)
            target_lengths = torch.tensor([len(a) for a in alignments], dtype=torch.long).to(device)

            loss = criterion(outputs.log_softmax(2).permute(1, 0, 2), alignments, input_lengths, target_lengths)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # Update the progress bar with the current loss
            pbar.set_postfix(loss=total_loss / len(pbar))

    print(f"Epoch {epoch + 1}, Loss: {total_loss / len(dataloader)}")

# Save the model
os.makedirs('models', exist_ok=True)
torch.save(model.state_dict(), 'models/lipnet.pth')

# Inference example
model.eval()
test_frames, test_alignments = next(iter(dataloader))
test_frames = test_frames.to(device)
with torch.no_grad():
    predictions = model(test_frames)
    decoded_predictions = torch.argmax(predictions, dim=2)

for i, decoded in enumerate(decoded_predictions):
    decoded_text = ''.join([num_to_char[idx.item()] for idx in decoded if idx.item() < len(vocab)])
    print(f"Prediction {i + 1}: {decoded_text}")
