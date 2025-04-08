import os
import cv2
import numpy as np
import subprocess
from flask import Flask, request

app = Flask(__name__)

# Create directory if it doesn't exist
SAVE_DIR = "Sample"
os.makedirs(SAVE_DIR, exist_ok=True)

frame_count = 0  # Track the number of received frames

@app.route('/process_frame', methods=['POST'])
def process_frame():
    global frame_count

    file = request.files.get('frame')
    
    if file:
        npimg = np.frombuffer(file.read(), np.uint8)
        frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        if frame is not None and frame_count < 75:  # Ensure frame is valid
            frame_path = os.path.join(SAVE_DIR, f"frame_{frame_count:03d}.jpg")
            cv2.imwrite(frame_path, frame)  # Save frame as an image
            frame_count += 1
            print(f"Saved: {frame_path}")

            # If 75 frames are collected, create the video
            if frame_count == 75:
                create_video()
        else:
            print("Frame is None or limit reached")

    else:
        print("No frame received in request")

    return "Frame received", 200

def create_video():
    print("Creating video from frames...")
    output_video_path = os.path.join(SAVE_DIR, "output.mpg")
    
    # Run FFmpeg command to create video
    cmd = [
        "ffmpeg", "-y", "-framerate", "25", "-i", 
        os.path.join(SAVE_DIR, "frame_%03d.jpg"), "-c:v", "mpeg1video", output_video_path
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print(f"Video created: {output_video_path}")
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg error: {e}")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)