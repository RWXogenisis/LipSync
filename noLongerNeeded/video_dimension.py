import cv2
import os

def get_video_details(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, None, None
    
    fps = cap.get(cv2.CAP_PROP_FPS)  # Frames per second
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Total number of frames
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # Video width
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # Video height
    cap.release()
    
    if fps > 0:
        return round(frame_count / fps, 2), width, height  # Duration in seconds, width, height
    return None, width, height

def process_videos_in_folder(folder_path):
    video_details = {}
    
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".mpg"):
            video_path = os.path.join(folder_path, filename)
            length, width, height = get_video_details(video_path)
            if length is not None:
                video_details[filename] = (length, width, height)
            else:
                video_details[filename] = "Could not process"
    
    return video_details

if __name__ == "__main__":
    folder_path = "/Users/aswinkumarv/Desktop/video_length/s13/video/mpg_6000"  # Set your folder path here
    if os.path.isdir(folder_path):
        results = process_videos_in_folder(folder_path)
        for video, details in results.items():
            if isinstance(details, tuple):
                length, width, height = details
                print(f"{video}: {length} seconds, Dimensions: {width}x{height}")
            else:
                print(f"{video}: {details}")
    else:
        print("Invalid folder path!")
