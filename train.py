# All the author(s): S Karun Vikhash, Hareesh S
import cv2
import mediapipe as mp
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from tqdm import tqdm
from typing import List
from torch.utils.data import Dataset, DataLoader

# ---------------------------- Device Configuration ---------------------------- #
# Author(s): S Karun Vikhash

# Empty the cache memory
torch.cuda.empty_cache()

# Determine device (GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device being used: {device}")  # Prints which device is being used (GPU or CPU)

# Check if CUDA is available
print(f"Is CUDA available? {torch.cuda.is_available()}")

# Print the number of GPUs available
print(f"Number of GPUs available: {torch.cuda.device_count()}")

if torch.cuda.is_available():
    # Print the amount of memory currently allocated on the GPU
    print(f"Memory currently allocated on GPU: {torch.cuda.memory_allocated()} bytes")
    # Print the maximum amount of memory allocated on the GPU so far
    print(f"Maximum memory allocated on GPU: {torch.cuda.max_memory_allocated()} bytes")
    # Print the name of the first GPU (if available)
    print(f"Name of the first GPU: {torch.cuda.get_device_name(0)}")

# ---------------------------- Mediapipe Setup ---------------------------- #
Author(s): Hareesh S

# Initialize Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

# -------------------------- Lip Landmark Indexes -------------------------- #
# Author(s): Hareesh S

# Define Lip Landmark Indices (from Mediapipe's 468 points)
LIP_LANDMARKS = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 146, 91, 181, 84, 
                 17, 314, 405, 321, 375, 291]

# ------------------------- Character Mapping -------------------------- #
# Author(s): S Karun Vikhash

# Vocabulary for the characters used in the lip reading model
vocab = [x for x in "abcdefghijklmnopqrstuvwxyz'?!0123456789 "]
char_to_num = {char: idx for idx, char in enumerate(vocab)}
num_to_char = {idx: char for char, idx in char_to_num.items()}

# --------------------------- Data Loaders --------------------------- #
def load_video(path: str, target_height=46, target_width=140) -> torch.Tensor:
    """
    Loads and processes a video into frames, extracting lip regions using Mediapipe.

    Args:
        path (str): The path to the video file.
        target_height (int): The target height of the resized lip region frames.
        target_width (int): The target width of the resized lip region frames.

    Returns:
        torch.Tensor: Normalized frames of the lip region for training.

    Author(s): Hareesh S
    """
    cap = cv2.VideoCapture(path)

    # List to store processed lip region frames
    frames = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to RGB and process with Mediapipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for Mediapipe
        results = face_mesh.process(rgb_frame)

        # Create a blank frame for lip region (if not detected, this will be appended)
        lip_region_resized = np.zeros((target_height, target_width), dtype=np.uint8)
        bbox = None

        # Extract lip landmarks if available
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0]

            # Extract pixel coordinates of the lip landmarks
            lip_points = [(int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0]))
                          for i, landmark in enumerate(landmarks.landmark) if i in LIP_LANDMARKS]
            
            if lip_points:
                # Compute a bounding box around the lip region
                x, y, w, h = cv2.boundingRect(np.array(lip_points))
                bbox = (x, y, w, h)

                # Ensure the bounding box is within the frame bounds
                if x >= 0 and y >= 0 and x + w < frame.shape[1] and y + h < frame.shape[0]:
                    # Crop the lip region from the frame
                    lip_region = frame[y:y+h, x:x+w]

                    # Convert the cropped lip region to grayscale
                    lip_region = cv2.cvtColor(lip_region, cv2.COLOR_BGR2GRAY)

                    # Resize the lip region to the target dimensions
                    lip_region_resized = cv2.resize(lip_region, (target_width, target_height))
        
        frames.append(lip_region_resized)
    
    cap.release()

    # Convert the list of frames to a NumPy array and to a PyTorch tensor with float32 data type
    frames = np.array(frames)
    frames = torch.tensor(frames, dtype=torch.float32)

    # Normalize the tensor (zero mean, unit variance)
    return (frames - frames.mean()) / frames.std()

def load_alignments(path: str) -> List[int]:
    """
    Loads the alignment file and converts tokens to indices using the vocabulary.

    Args:
        path (str): Path to the alignment file.

    Returns:
        List[int]: List of indices corresponding to the tokenized alignments.

    Author(s): S Karun Vikhash
    """
    with open(path, 'r') as f:
        lines = f.readlines()
    tokens = []
    for line in lines:
        parts = line.split()
        if parts[2] != 'sil': # Do not include silent ('sil') from GRID Corpa
            tokens.extend([' ', parts[2].lower()])
    return [char_to_num[char] for char in ''.join(tokens)]

def load_data(video_path: str, alignment_path: str):
    """
    Loads frames and alignments for a video.

    Args:
        video_path (str): Path to the video file.
        alignment_path (str): Path to the alignment file.

    Returns:
        tuple: A tuple containing the frames tensor and alignment indices.

    Author(s): S Karun Vikhash
    """
    frames = load_video(video_path)
    alignments = load_alignments(alignment_path)
    return frames, alignments

# ---------------------------- Dataset Class ---------------------------- #
class LipNetDataset(Dataset):
    """
    Dataset class for lip reading, loading frames and their corresponding alignments.

    Args:
        file_paths (List[str]): List of file paths to video files.
        base_dir (str): Base directory where videos and alignments are stored.
        target_length (int): The target length for padding/truncating video frames.

    Author(s): S Karun Vikhash
    """
    def __init__(self, file_paths: List[str], base_dir: str, target_length: int):
        """
        Author(s): S Karun Vikhash
        """
        self.file_paths = file_paths
        self.base_dir = base_dir
        self.target_length = target_length

    def __len__(self):
        """
        Returns the number of samples in the dataset.

        Author(s): S Karun Vikhash
        """
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        """
        Loads and returns a sample (frames, alignments) from the dataset.

        Args:
            idx (int): Index of the sample.

        Returns:
            tuple: A tuple containing the frames and alignment tensor.

        Author(s): S Karun Vikhash
        """
        # Extract base name (without extension), construct full path to the video file and to the alignment file
        file_name = os.path.splitext(os.path.basename(self.file_paths[idx]))[0]
        video_path = os.path.join(self.base_dir, 'shuffled', 's1', f'{file_name}.mpg')
        alignment_path = os.path.join(self.base_dir, 'shuffled', 'alignments', 's1', f'{file_name}.align')

        # Use the video processor to load and preprocess the video into frames and actual alignments into a list
        frames, alignments = load_data(video_path, alignment_path)

        # If the video has no valid frames, raise an error to skip this sample
        if frames.size(0) == 0:
            raise ValueError(f"Empty frames detected for {file_name}, skipping this sample.")
    
        num_frames = frames.size(0)

        # If the number of frames is less than the required length, pad the tensor
        if num_frames < self.target_length:
            padding = self.target_length - num_frames
            # Pad at the beginning along the time dimension (frames) with zeros
            frames = torch.nn.functional.pad(frames, (0, 0, 0, 0, 0, padding))  
        elif num_frames > self.target_length:
            # If there are more frames than needed, truncate to the target length
            frames = frames[:self.target_length]

        # Add a channel dimension (1) to make it (1, T, H, W) format for CNN input
        frames = frames.unsqueeze(0)
        return frames, torch.tensor(alignments, dtype=torch.long)

# ------------------------ Model Definition ------------------------ #   
class LipNetModel(nn.Module):
    """
    Defines the LipNet model architecture for lip reading.
    The model consists of 3D convolutional layers followed by a bidirectional LSTM
    and a fully connected output layer.

    Args:
        vocab_size (int): The number of classes (characters/words), excluding the blank for CTC.

    Author(s): S Karun Vikhash
    """

    def __init__(self, vocab_size):
        """
        Author(s): S Karun Vikhash
        """
        super(LipNetModel, self).__init__()

        # First 3D convolution: input channels = 1 (grayscale), output channels = 128
        self.conv3d_1 = nn.Conv3d(1, 128, kernel_size=3, padding=1)

        # 3D max pooling over (time=1, height=2, width=2) to reduce spatial size
        self.pool3d_1 = nn.MaxPool3d((1, 2, 2))

        # Second 3D convolution: output channels = 256
        self.conv3d_2 = nn.Conv3d(128, 256, kernel_size=3, padding=1)
        self.pool3d_2 = nn.MaxPool3d((1, 2, 2))  # Further reduce spatial resolution

        # Third 3D convolution: output channels = 75
        self.conv3d_3 = nn.Conv3d(256, 75, kernel_size=3, padding=1)
        self.pool3d_3 = nn.MaxPool3d((1, 2, 2))  # Final spatial downsampling

        # Bidirectional LSTM for sequence modeling
        # Input size: 75 feature maps * 5 (height) * 17 (width) = 6375
        # Output size: 128 units * 2 directions = 256
        self.lstm = nn.LSTM(75 * 5 * 17, 128, num_layers=2, bidirectional=True, batch_first=True)

        # Fully connected layer for final classification (output: vocab_size + 1 for CTC blank)
        self.fc = nn.Linear(256, vocab_size + 1)

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 1, time, height, width)

        Returns:
            torch.Tensor: Output logits of shape (batch_size, time, vocab_size + 1)

        Author(s): S Karun Vikhash
        """
        # Apply 1st conv layer + ReLU + max pool
        x = self.pool3d_1(torch.relu(self.conv3d_1(x)))

        # Apply 2nd conv layer + ReLU + max pool
        x = self.pool3d_2(torch.relu(self.conv3d_2(x)))

        # Apply 3rd conv layer + ReLU + max pool
        x = self.pool3d_3(torch.relu(self.conv3d_3(x)))

        # Reshape from (B, C, T, H, W) -> (B, T, C * H * W) for LSTM
        x = x.view(x.size(0), x.size(2), -1)

        # Pass through bidirectional LSTM
        x, _ = self.lstm(x)

        # Final linear layer to get class logits per time step
        return self.fc(x)

def collate_fn(batch):
    """
    Pads input sequences and aligns target labels for batch processing.

    Args:
        batch (List[Tuple[Tensor, Tensor]]): List of (frames, labels)

    Returns:
        Tuple[Tensor, Tensor]: Padded frames and padded alignments

    Author(s): S Karun Vikhash
    """
    # Extract all frames (input data) and alignments (target labels) from the batch
    frames = [item[0] for item in batch]
    alignments = [item[1] for item in batch]

    # Get the shape of each frame in the batch and check if all the frame shapes are the same
    frame_shapes = [frame.shape for frame in frames]
    if len(set(frame_shapes)) > 1:
        raise ValueError(f"Frame sizes are inconsistent: {frame_shapes}")

    # Stack the frames along a new batch dimension (this adds the batch dimension)
    frames_padded = torch.stack(frames)
    
    # Pad the alignments (target labels) to have the same length for each sequence
    # This is important because sequences in a batch may have different lengths
    alignments_padded = nn.utils.rnn.pad_sequence(alignments, batch_first=True)

    return frames_padded, alignments_padded

# -------------------------- Training Prep -------------------------- #
# Author(s): S Karun Vikhash

# Set the base directory to the current directory
base_dir = '.\\'


file_paths = [
    os.path.join(base_dir, 'shuffled', 's1', f) # Construct the full file path for each file in the directory
    for f in os.listdir(os.path.join(  
        base_dir, 'shuffled', 's1'
        )
    )                                           # List all files in the 'shuffled/s1' directory
    if f.endswith('.mpg')                       # Filter to include only files that end with '.mpg' (video files)                                            
]

# Initialize an empty list to store the lengths of each video
video_lengths = []

# Iterate through each video file to calculate its length (number of frames)
for file_path in file_paths:
    cap = cv2.VideoCapture(file_path)
    video_lengths.append(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))) # Get the total number of frames in the video
    cap.release()

# Compute the average length (number of frames) across all videos to use as the target length for padding
target_length = int(np.mean(video_lengths))
print(f"Target video length: {target_length}")

# Create an instance of the LipNetDataset with the file paths, base directory, and target length
dataset = LipNetDataset(file_paths, base_dir, target_length)

# Initialize the DataLoader to load data from the dataset, with a batch size of 1 and shuffling enabled
dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)

# Initialize the LipNetModel with the vocabulary size (length of the vocab)
model = LipNetModel(vocab_size=len(vocab)).to(device)

# Define the CTC loss criterion, using the length of the vocabulary as the number of output classes
criterion = nn.CTCLoss(blank=len(vocab), zero_infinity=True)

# Initialize the Adam optimizer
optimizer = optim.Adam(model.parameters(), lr=1e-5)

# ---------------------------- Training Loop ---------------------------- #
# Author(s): S Karun Vikhash, Hareesh S

numEpoches = 250
for epoch in range(numEpoches):
    model.train()   # Set the model to training mode (important for layers like dropout, batch norm, etc.)
    total_loss = 0  # Initialize total loss for this epoch
    numNan = 0      # Counter for how many times the loss is NaN during training

    with tqdm(dataloader, desc=f"Epoch {epoch+1}/{numEpoches}", unit="batch") as pbar:
        for frames, alignments in pbar:
            frames = frames.to(device)          # Move frames to the appropriate device (GPU/CPU)
            alignments = alignments.to(device)  # Move alignments (labels) to the appropriate device

            optimizer.zero_grad()               # Zero out any accumulated gradients from previous steps
            outputs = model(frames)

            # Prepare input lengths tensor (each frame's sequence length for the input)
            input_lengths = torch.full((outputs.size(0),), outputs.size(1), dtype=torch.long).to(device)

            # Prepare target lengths tensor (the length of each target sequence/label)
            target_lengths = torch.tensor([len(a) for a in alignments], dtype=torch.long).to(device)

            # Compute the CTC loss. Outputs are log-softmaxed, permuted to match the expected format.
            loss = criterion(outputs.log_softmax(2).permute(1, 0, 2), alignments, input_lengths, target_lengths)

            # Check if the computed loss is NaN, and skip this batch if so
            if torch.isnan(loss).any():
                numNan += 1
                continue

            # Backpropagate the gradients and update the model parameters based on the gradients
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix(loss=total_loss / len(pbar))

    print(f"Epoch {epoch + 1}, Loss: {total_loss / len(dataloader)}, numNan: {numNan}")

# ---------------------------- Save Model ---------------------------- #
# Author(s): S Karun Vikhash
# Save the model's state_dict (weights) to a file
os.makedirs('models', exist_ok=True)
torch.save(model.state_dict(), 'models/lipnet_mp_actualShuffled.pth')

# ---------------------------- Inference ---------------------------- #
# Author(s): S Karun Vikhash

model.eval()

# Get the next batch from the dataloader for testing (using the first batch)
test_frames, test_alignments = next(iter(dataloader))
test_frames = test_frames.to(device)

# Perform inference without computing gradients (to save memory and computation)
with torch.no_grad():
    predictions = model(test_frames)                        # Pass test frames through the model to get predictions (logits)
    decoded_predictions = torch.argmax(predictions, dim=2)  # Get the predicted class for each timestep (across all sequences)

# Iterate over the decoded predictions and convert them to text
for i, decoded in enumerate(decoded_predictions):
    decoded_text = ''.join([
        num_to_char[idx.item()] for idx in decoded 
        if idx.item() < len(vocab)
    ])
    print(f"Prediction {i + 1}: {decoded_text}")
