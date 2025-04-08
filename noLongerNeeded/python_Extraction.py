import cv2
import dlib
import numpy as np

def extract_lip_region(frame, detector, predictor, target_height=46, target_width=140):
    """
    Extracts the lip region from the frame using dlib's face detector and landmark predictor.
    Returns the lip region resized to the target height and width.
    """
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale image
    faces = detector(gray)

    if len(faces) > 0:
        # We assume the first detected face is the one we want to process
        face = faces[0]

        # Get the landmarks
        landmarks = predictor(gray, face)

        # Extract the lip region (landmarks for lips are from point 48 to 67)
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

        return lip_region_resized
    return None

def display_lip_region(video_path):
    # Load dlib's face detector and shape predictor
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')  # Make sure to download this file

    # Open the video
    cap = cv2.VideoCapture(video_path)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Extract the lip region from the current frame
        lip_region = extract_lip_region(frame, detector, predictor)

        if lip_region is not None:
            # Display the lip region
            cv2.imshow("Lip Region", lip_region)

        # Press 'q' to quit the video window
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_path = "E:\DLib - Extension\WhatsApp Video 2025-02-13 at 11.34.04 PM.mp4"  # Replace with your video file path
    display_lip_region(video_path)