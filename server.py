# All the author(s): Hareesh S, S Karun Vikhash
from concurrent.futures import ThreadPoolExecutor
from flask import Flask, request, jsonify
from flask_cors import CORS
from LipReadingAPI import LipReadingAPI
import os
import threading
from SpellCorrecter import Preprocessor, SpellCorrector
from VideoProcessor import VideoProcessor
from waitress import serve

# ---------------------------- Initialization ---------------------------- #
# Author(s): Hareesh S

# Cache dictionary: {video_path: predicted_caption}
prediction_cache = {}
cache_lock = threading.Lock()
processed_count = 0

# Limit threads to logical processors (or a bit less to avoid 100% load)
MAX_WORKERS = 10
executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)

# Paths
BASE_DIR = "./videoSegments"
MODEL_PATH = "models/lipnet_mp_shuffled.pth"

# Initialize only once
videoProcessor = VideoProcessor()
preprocessor = Preprocessor()
corrector = SpellCorrector(preprocessor)

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True, allow_headers=["Content-Type"], methods=["GET", "POST"])

youtube_url = None  # Store the latest received URL
current_timestamp = 0  # Store the latest timestamp
video_segments = []  # Store sorted segment filenames
segment_index = 0  # Track which segment to send next

# Add a flag to avoid repeated thread spawns
prediction_started = False

# ---------------------------- Main Server Logic ---------------------------- #
# Author(s): Hareesh S, S Karun Vikhash

def loadVideoSegments():
    """
    Loads and sorts video segment filenames from the output folder.

    This function updates the global `video_segments` list with filenames that
    match the naming pattern `segment_<index>_final.mpg`, sorted by index.

    Returns:
        None

    Author(s): Hareesh S
    """
    global video_segments
    folder_path = BASE_DIR
    if not os.path.exists(folder_path):
        return []

    # Load video segments and sort by numerical index
    video_segments = sorted(
        [f for f in os.listdir(folder_path) if f.startswith("segment_") and f.endswith("_final.mpg")],
        key=lambda x: int(x.split("_")[1])  # Extract numerical index from filename
    )

# Load segments initially
loadVideoSegments()

def process_segment(video_filename):
    """
    Processes a single video segment to predict text using the lip-reading model.

    This function runs the lip-reading model on a given video segment and caches
    the predicted caption. It also ensures that no video segment is processed
    more than once by using a cache and lock.

    Args:
        video_filename (str): The filename of the video segment to process.

    Returns:
        None

    Author(s): Hareesh S
    """
    global processed_count
    video_filename = os.path.join(BASE_DIR, video_filename)
    with cache_lock:
        # Skip if already processed
        if video_filename in prediction_cache:
            return

    try:
        # Run the model
        local_lip_api = LipReadingAPI(MODEL_PATH, BASE_DIR)
        predicted_text = local_lip_api.predict(video_filename)

        # Update cache and counter
        with cache_lock:
            corrected_text = corrector.correct_sentence(predicted_text)
            prediction_cache[video_filename] = corrected_text
            processed_count += 1

        print(f"[{processed_count}/{len(video_segments)}] Processed: {video_filename}")

    except Exception as e:
        print(f"[ERROR] Failed to process {video_filename}: {e}")

@app.route("/update_timestamp", methods=["POST"])
def updateTimestamp():
    """
    Updates the current timestamp and returns the corresponding video segment data.

    This endpoint is used to send a timestamp from the client and get the predicted
    caption for the corresponding video segment.

    Args:
        None

    Returns:
        JSON: Success status, timestamp, and segment data (or error message).

    Author(s): Hareesh S, S Karun Vikhash
    """
    global current_timestamp, segment_index, video_segments
    data = request.json

    if "timestamp" in data:
        current_timestamp = data["timestamp"]
        segment_index = int(current_timestamp // 3)  # 3 seconds per segment
        print(len(video_segments), segment_index)

        print(f"[SERVER] Updated Timestamp: {current_timestamp}s")

        if segment_index < len(video_segments):
            video_filename = video_segments[segment_index]
            video_path = os.path.join(BASE_DIR, video_filename)

            with cache_lock:
                predicted_text = prediction_cache.get(video_path)

            if predicted_text is None:
                predicted_text = "[Processing or not available yet]"

            segment_data = {video_filename: (predicted_text + f" @ {current_timestamp}s")}
        else:
            segment_data = {"message": "No more segments available."}

        return jsonify({
            "success": True,
            "timestamp": current_timestamp,
            "segment_data": segment_data
        })

    return jsonify({"success": False, "message": "Invalid data."})

@app.route("/get_timestamp", methods=["GET"])
def getTimestamp():
    """
    Returns the current timestamp value stored on the server.

    This endpoint is used to keep track of synchronization between client and server.

    Args:
        None

    Returns:
        JSON: Current timestamp.

    Author(s): Hareesh S
    """
    return jsonify({"timestamp": current_timestamp})

@app.route("/receive_url", methods=["POST"])
def receiveUrl():
    """
    Receives a YouTube video URL and starts the video processing pipeline.

    This endpoint is triggered when a new YouTube video URL is received. It processes
    the video, extracts segments, and starts prediction if the URL is new.

    Args:
        None

    Returns:
        JSON: Success status and message.

    Author(s): Hareesh S
    """
    global youtube_url, current_timestamp, video_segments, segment_index
    data = request.json

    if "youtubeURL" in data:
        global processed_count
        global prediction_started
        new_url = data["youtubeURL"]
        # Check if it's a new URL
        if new_url != youtube_url:
            # Reset state variables
            youtube_url = new_url
            current_timestamp = 0
            video_segments = []
            segment_index = 0
            prediction_cache.clear()
            processed_count = 0
            prediction_started = False

        # Process video
        status, result = videoProcessor.process_video(new_url)
        if status:
            # Reload video segments after processing
            loadVideoSegments()
            if not prediction_started:
                prediction_started = True
                # Start processing each video segment in a separate thread
                for video_filename in video_segments:
                    executor.submit(process_segment, video_filename)
            return jsonify({"success": True, "message": "URL received and processing started."})
        else:
            return jsonify({"success": False, "message": result})
    return jsonify({"success": False, "message": "Invalid data."})

@app.route("/processing_status", methods=["GET"])
def processingStatus():
    """
    Returns the current processing status of video segments.

    This endpoint provides information on whether all segments have been processed
    or if some are still being processed.

    Args:
        None

    Returns:
        JSON: Processing status and counts.

    Author(s): Hareesh S
    """
    with cache_lock:
        cached_files = list(prediction_cache.keys())
    is_complete = len(cached_files) == len(video_segments)

    return jsonify({
        "status": "complete" if is_complete else "processing",
        "processed_count": len(cached_files),
        "total_segments": len(video_segments)
    })

@app.route("/get_url", methods=["GET"])
def getUrl():
    """
    Returns the last YouTube URL received and processed by the server.

    If no URL has been received, returns an error message.

    Args:
        None

    Returns:
        JSON: YouTube URL or error message.

    Author(s): Hareesh S
    """
    if youtube_url:
        return jsonify({"youtubeURL": youtube_url})
    return jsonify({"error": "No URL received yet."})

if __name__ == "__main__":
    # Start the Flask app with Waitress server
    serve(app, port=5001)
