import torch
import torch.nn as nn
import cv2
import os
import numpy as np
import mediapipe as mp
from flask import Flask, request, jsonify
from flask_cors import CORS
import threading

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load LipNet Model
class LipNetModel(nn.Module):
    def __init__(self, vocab_size):
        super(LipNetModel, self).__init__()
        self.conv3d_1 = nn.Conv3d(1, 128, kernel_size=3, padding=1)
        self.pool3d_1 = nn.MaxPool3d((1, 2, 2))
        self.conv3d_2 = nn.Conv3d(128, 256, kernel_size=3, padding=1)
        self.pool3d_2 = nn.MaxPool3d((1, 2, 2))
        self.conv3d_3 = nn.Conv3d(256, 75, kernel_size=3, padding=1)
        self.pool3d_3 = nn.MaxPool3d((1, 2, 2))
        self.lstm = nn.LSTM(8100, 128, num_layers=2, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(256, 40)  # 40 vocab chars + blank for CTC

    def forward(self, x):
        x = self.pool3d_1(torch.relu(self.conv3d_1(x)))
        x = self.pool3d_2(torch.relu(self.conv3d_2(x)))
        x = self.pool3d_3(torch.relu(self.conv3d_3(x)))
        x = x.view(x.size(0), x.size(2), -1)
        x, _ = self.lstm(x)
        x = self.fc(x)
        return x

# Load the trained model
model = LipNetModel(vocab_size=40)
model.load_state_dict(torch.load('E:\\DLib - Extension\\backend\\lipnet_mp_shuffled.pth', map_location=torch.device('cpu')))
model.eval()

# Initialize Mediapipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

LIP_LANDMARKS = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 146, 91, 181, 84, 
                 17, 314, 405, 321, 375, 291]

frame_buffer = []
buffer_lock = threading.Lock()

char_map = "abcdefghijklmnopqrstuvwxyz'?!0123456789 "
num_to_char = {i: char for i, char in enumerate(char_map)}

def extract_lip_region(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0]
        lip_points = [(int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0]))
                      for i, landmark in enumerate(landmarks.landmark) if i in LIP_LANDMARKS]
        if lip_points:
            x, y, w, h = cv2.boundingRect(np.array(lip_points))
            lip_region = frame[y:y+h, x:x+w]
            lip_region = cv2.cvtColor(lip_region, cv2.COLOR_BGR2GRAY)
            resized_lip = cv2.resize(lip_region, (140, 46))
            return resized_lip
    return None

def predict_text(frames):
    frames = np.array(frames)
    frames = torch.tensor(frames, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # (1, 1, T, H, W)
    with torch.no_grad():
        predictions = model(frames)
        decoded_predictions = torch.argmax(predictions, dim=2).squeeze(0)
    return ''.join([num_to_char[idx.item()] for idx in decoded_predictions if idx.item() < len(char_map)])

@app.route('/process_frame', methods=['POST'])
def process_frame():
    try:
        if 'frame' not in request.files:
            return jsonify({'error': 'No frame received'}), 400

        file = request.files['frame']
        np_img = np.frombuffer(file.read(), np.uint8)
        frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
        
        if frame is None:
            return jsonify({'error': 'Invalid frame'}), 400

        with buffer_lock:
            frame_buffer.append(frame)
            if len(frame_buffer) >= 79:
                frames_to_process = [extract_lip_region(f) for f in frame_buffer[-75:]]
                frame_buffer.clear()
                frames_to_process = [f for f in frames_to_process if f is not None]
                if len(frames_to_process) == 79:
                    predicted_text = predict_text(frames_to_process)
                    return jsonify({'text': predicted_text})
                else:
                    return jsonify({'error': 'Not enough valid frames'}), 400

        return jsonify({'message': 'Frame stored'}), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='localhost', port=5000, debug=True, threaded=True)