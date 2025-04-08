import socket
import os
import cv2
import numpy as np

SAVE_DIR = "Sample"
os.makedirs(SAVE_DIR, exist_ok=True)

frame_count = 0
MAX_FRAMES = 10

# Create UDP socket
UDP_IP = "0.0.0.0"
UDP_PORT = 5005
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))

print(f"Listening for UDP packets on {UDP_IP}:{UDP_PORT}...")

while frame_count < MAX_FRAMES:
    data, addr = sock.recvfrom(65536)  # Max UDP packet size
    if data:
        npimg = np.frombuffer(data, dtype=np.uint8)
        frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        if frame is not None:
            frame_path = os.path.join(SAVE_DIR, f"frame_{frame_count:03d}.jpg")
            cv2.imwrite(frame_path, frame)
            print(f"Saved: {frame_path}")
            frame_count += 1

#if frame_count == MAX_FRAMES:
#    print("75 frames received. Building video...")
#    output_path = os.path.join(SAVE_DIR, "output.mpg")
#    cmd = [
#        "ffmpeg", "-y", "-framerate", "25", "-i",
#        os.path.join(SAVE_DIR, "frame_%03d.jpg"), "-c:v", "mpeg1video", output_path
#    ]
#    os.system(" ".join(cmd))
#    print("Video created.")