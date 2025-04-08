# All the author(s): S Karun Vikhash, Hareesh S 
import cv2
import mediapipe as mp
import numpy as np
import os
import re
from typing import List
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Device configuration: 
# Check if a CUDA-enabled GPU is available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Vocabulary: 
# Define the set of characters that the model can predict, including letters, punctuation, digits, and space
vocab = [x for x in "abcdefghijklmnopqrstuvwxyz'?!0123456789 "]  # List of valid characters

# Create a mapping from characters to numerical indices (for input to the model)
char_to_num = {char: idx for idx, char in enumerate(vocab)}

# Create a reverse mapping from numerical indices back to characters (for decoding model outputs)
num_to_char = {idx: char for char, idx in char_to_num.items()}

class VideoProcessorBackend:
    """
    Handles video processing and extraction of the lip region from video frames using Mediapipe FaceMesh.

    Attributes:
        target_height (int): The height to which the extracted lip region should be resized.
        target_width (int): The width to which the extracted lip region should be resized.
        LIP_LANDMARKS (List[int]): Indices of facial landmarks that correspond to the lips.
        face_mesh (mp.solutions.face_mesh.FaceMesh): Mediapipe FaceMesh model for detecting facial landmarks.
    Author(s): Hareesh S, S Karun Vikhash
    """

    def __init__(self, target_height=46, target_width=140):
        """
        Initializes the VideoProcessorBackend with given target dimensions and sets up the FaceMesh model.
        
        Args:
            target_height (int): Height of the output lip region frame.
            target_width (int): Width of the output lip region frame.

        Author(s): Hareesh S
        """
        self.target_height = target_height
        self.target_width = target_width

        # Define the landmark indices corresponding to the lips from Mediapipe's 468-point face mesh
        self.LIP_LANDMARKS = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 146,
                              91, 181, 84, 17, 314, 405, 321, 375, 291]
        
        # Initialize Mediapipe's FaceMesh model
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,    # Enables processing of a video stream instead of static images
            max_num_faces=1,            # Only detect one face per frame
            refine_landmarks=True       # Improves landmark accuracy (especially around eyes and lips)
        )

    def load_video(self, path: str) -> torch.Tensor:
        """
        Loads a video from the given file path, extracts the lip region from each frame,
        resizes it to the target dimensions, and returns the sequence of frames as a normalized tensor.
        
        Args:
            path (str): Path to the input video file.
        
        Returns:
            torch.Tensor: A normalized tensor of shape (num_frames, target_height, target_width) containing
                          the grayscale lip region for each frame.

        Author(s): Hareesh S 
        """
        cap = cv2.VideoCapture(path)
        
        # List to store processed lip region frames
        frames = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for Mediapipe
            results = self.face_mesh.process(rgb_frame)

            # Create a blank frame for lip region (if not detected, this will be appended)
            lip_region_resized = np.zeros((self.target_height, self.target_width), dtype=np.uint8)

            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0]  # Get landmarks of the first detected face

                # Extract pixel coordinates of the lip landmarks
                lip_points = [(int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0]))
                              for i, landmark in enumerate(landmarks.landmark) if i in self.LIP_LANDMARKS]
                
                if lip_points:
                    # Compute a bounding box around the lip region
                    x, y, w, h = cv2.boundingRect(np.array(lip_points))

                    # Ensure the bounding box is within the frame bounds
                    if x >= 0 and y >= 0 and x + w < frame.shape[1] and y + h < frame.shape[0]:
                        # Crop the lip region from the frame
                        lip_region = frame[y:y+h, x:x+w]

                        # Convert the cropped lip region to grayscale
                        lip_region = cv2.cvtColor(lip_region, cv2.COLOR_BGR2GRAY)

                        # Resize the lip region to the target dimensions
                        lip_region_resized = cv2.resize(lip_region, (self.target_width, self.target_height))
            
            frames.append(lip_region_resized)
        
        cap.release()

        # Convert the list of frames to a NumPy array and to a PyTorch tensor with float32 data type
        frames = np.array(frames)
        frames = torch.tensor(frames, dtype=torch.float32)

        # Normalize the tensor (zero mean, unit variance)
        return (frames - frames.mean()) / frames.std()

class LipNetDataset(Dataset):
    """
    Custom dataset for LipNet. Loads videos, processes them into frames,
    and ensures a fixed number of frames per sample for model input.
    
    Attributes:
        file_paths (List[str]): List of video file paths (or identifiers).
        base_dir (str): Directory where video files are stored.
        target_length (int): Desired number of frames per video sample.
        video_processor (VideoProcessorBackend): Processor to extract and preprocess lip regions from videos.
        
    Author(s): S Karun Vikhash
    """

    def __init__(self, file_paths: List[str], base_dir: str, target_length: int, video_processor: VideoProcessorBackend):
        """
        Initializes the dataset with file paths and video processor.

        Args:
            file_paths (List[str]): List of video identifiers or filenames.
            base_dir (str): Path to the directory containing video files.
            target_length (int): Number of frames each sample should have.
            video_processor (VideoProcessorBackend): Preprocessor for extracting lip regions.

        Author(s): S Karun Vikhash
        """
        self.file_paths = file_paths            # Store the list of file identifiers or paths
        self.base_dir = base_dir                # Path where video files are located
        self.target_length = target_length      # Fixed number of frames per sample (for model compatibility)
        self.video_processor = video_processor  # Lip region extractor and preprocessor

    def __len__(self):
        """
        Returns the number of video samples in the dataset.

        Author(s): S Karun Vikhash
        """
        return len(self.file_paths)  # Dataset size is the number of file paths provided

    def __getitem__(self, idx):
        """
        Loads and processes a single video sample.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            torch.Tensor: Preprocessed video tensor of shape (1, target_length, height, width)

        Author(s): S Karun Vikhash
        """
        # Extract base name (without extension), construct full path to the video file
        file_name = os.path.splitext(os.path.basename(self.file_paths[idx]))[0]
        video_path = os.path.join(self.base_dir, f'{file_name}.mpg')

        # Use the video processor to load and preprocess the video into frames
        frames = self.video_processor.load_video(video_path)

        # If the video has no valid frames, raise an error to skip this sample
        if frames.size(0) == 0:
            raise ValueError(f"Empty frames detected for {file_name}, skipping this sample.")

        # If the number of frames is less than the required length, pad the tensor
        if frames.size(0) < self.target_length:
            padding_size = self.target_length - frames.size(0)
            # Pad at the beginning along the time dimension (frames) with zeros
            frames = torch.nn.functional.pad(frames, (0, 0, 0, 0, 0, padding_size))
        else:
            # If there are more frames than needed, truncate to the target length
            frames = frames[:self.target_length]

        # Add a channel dimension (1) to make it (1, T, H, W) format for CNN input
        return frames.unsqueeze(0)

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

class LipReadingAPI:
    """
    Main API class for performing lip-reading predictions using a trained LipNet model.

    Attributes:
        device (str): The device to run the model on (e.g., 'cpu' or 'cuda').
        base_dir (str): Directory where video files are stored.
        video_processor (VideoProcessorBackend): Video processor for extracting lip regions.
        model (LipNetModel): The LipNet model for performing lip-reading predictions.
        
    Author(s): S Karun Vikhash
    """

    def __init__(self, model_path, base_dir):
        """
        Initializes the LipReadingAPI with model loading, device selection, and video processor setup.

        Args:
            model_path (str): Path to the pre-trained model.
            base_dir (str): Directory where video files are stored.

        Author(s): S Karun Vikhash
        """
        # 'device' refers to either 'cpu' or 'cuda'
        self.device = device

        # Store the base directory for video files
        self.base_dir = base_dir
        
        self.video_processor = VideoProcessorBackend()

        # Initialize the LipNet model and load the trained weights
        self.model = LipNetModel(vocab_size=len(vocab)).to(device)
        self.model.load_state_dict(torch.load(model_path, map_location=device))  # Load the model's state dict
        self.model.eval()  # Set the model to evaluation mode (turn off dropout, etc.)

    def ctc_decode(self, predictions):
        """
        Decodes the model output using CTC (Connectionist Temporal Classification) decoding.
        
        Args:
            predictions (torch.Tensor): The output tensor from the model.

        Returns:
            str: The decoded text string.

        Author(s): S Karun Vikhash
        """
        decoded = []  # List to hold decoded characters
        prev_char = None  # Variable to keep track of the previous character

        # Iterate over the predicted indices from the model's output
        for idx in predictions:
            char = num_to_char.get(idx, "")  # Convert index to character using the num_to_char map
            # Only add the character if it's different from the previous one and not a space
            if char != prev_char and char != " ":
                decoded.append(char)
            prev_char = char  # Update previous character

        # Join the list of decoded characters into a string and return it
        return "".join(decoded)

    def predict(self, video_path: str, target_length=79):
        """
        Processes a single video file, makes lip-reading predictions, and returns the predicted text.

        Args:
            video_path (str): Path to the video file to process.
            target_length (int): Desired number of frames to process from the video.

        Returns:
            str: The predicted text from the lip reading model.

        Author(s): S Karun Vikhash
        """
        file_name = os.path.splitext(os.path.basename(video_path))[0]

        # Create a dataset for the given video file, passing it to the video processor
        dataset = LipNetDataset([file_name], self.base_dir, target_length, self.video_processor)
        
        # Create a DataLoader to batch the dataset
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

        # Disable gradient computation (since the model is in inference mode)
        with torch.no_grad():
            for frames in dataloader:
                # If no frames are extracted (empty tensor), return an empty string
                if frames.numel() == 0:
                    return ""

                frames = frames.to(self.device)  # Move frames to the correct device (CPU or GPU)

                # Pass frames through the LipNet model to get predictions
                outputs = self.model(frames)

                # Decode the model's output (get the most probable character for each time step)
                decoded_predictions = torch.argmax(outputs, dim=2)

                # Convert the predicted indices to characters
                predicted_text = ''.join([
                    num_to_char[idx.item()] for idx in decoded_predictions[0]
                    if idx.item() < len(vocab)
                ])

                # Clean up the decoded text (replace multiple spaces with a single space and strip)
                predicted_text = re.sub(r'\s+', ' ', predicted_text).strip()

                print("[DEBUG] Raw decoded text:", predicted_text)
                return predicted_text


if __name__ == "__main__":
    base_dir = ".\\videoSegments"  # Directory containing the video segments
    video_path = ".\\videoSegments\\segment_0_final.mpg"  # Path to the specific video segment to process
    model_path = "models/lipnet_mp_shuffled.pth"

    # Use the LipReadingAPI to predict the text from the video
    lip_api = LipReadingAPI(model_path, base_dir)
    prediction = lip_api.predict(video_path)
    print("Prediction:", prediction)
