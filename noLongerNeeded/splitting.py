import os
import subprocess
import pandas as pd
from collections import defaultdict

# Define input and output directories
input_video_dir = '/Users/aswinkumarv/Downloads/videoshuffling/data/s1'  # Folder with input videos
input_align_dir = '/Users/aswinkumarv/Downloads/videoshuffling/data/alignments/s1'  # Folder with input align files
video_output_base = 'result_video_files'
align_output_base = 'result_align_files'

# Ensure output directories exist
os.makedirs(video_output_base, exist_ok=True)
os.makedirs(align_output_base, exist_ok=True)

# Function to get video duration using ffprobe
def get_video_duration(video_path):
    command = [
        'ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
        'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', video_path
    ]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return float(result.stdout.strip())  # Convert result to float (seconds)

# Function to extract clips from the video (now in .mpg format)
def extract_clip(input_video, output_video, start_time, duration):
    command = [
        'ffmpeg', '-i', input_video, '-ss', str(start_time), '-t', str(duration), 
        '-c:v', 'mpeg2video', '-q:v', '2', '-c:a', 'mp2', '-b:a', '192k', '-y', output_video
    ]
    subprocess.run(command)

# Function to create an align file for a given word
def create_align_file(word, original_duration_ms, output_align_file):
    with open(output_align_file, 'w') as f:
        f.write(f"0 {int(original_duration_ms)} {word}\n")  # Store the original word without index
    print(f"Created align file: {output_align_file} -> 0 {int(original_duration_ms)} {word}")

# Process each video and its corresponding align file
for video_file in sorted(os.listdir(input_video_dir)):  # Ensure files are processed in order
    if video_file.endswith('.mpg'):
        video_path = os.path.join(input_video_dir, video_file)
        
        # Find the corresponding align file (same name but .align extension)
        align_filename = os.path.splitext(video_file)[0] + '.align'
        align_path = os.path.join(input_align_dir, align_filename)
        
        if not os.path.exists(align_path):
            print(f" No align file found for {video_file}, skipping...")
            continue

        print(f" Processing: {video_file} with {align_filename}")

        # Create subfolders for this video
        video_output_dir = os.path.join(video_output_base, os.path.splitext(video_file)[0])
        align_output_dir = os.path.join(align_output_base, os.path.splitext(video_file)[0])
        os.makedirs(video_output_dir, exist_ok=True)
        os.makedirs(align_output_dir, exist_ok=True)

        # Read the align file into a pandas DataFrame
        df = pd.read_csv(align_path, delim_whitespace=True, names=['start', 'end', 'word'])

        # Get the actual video length in seconds
        video_duration = get_video_duration(video_path)

        # Calculate scale factor (to convert timestamps to real video time)
        scale_factor = video_duration / 74500  # Converts milliseconds to seconds

        # Track occurrences of words
        word_count = defaultdict(int)

        # Iterate through each word and extract the corresponding video segment
        for _, row in df.iterrows():
            start_timestamp = row['start']
            end_timestamp = row['end']
            word = row['word']
            
            # Convert timestamps to real video time
            start_time = start_timestamp * scale_factor
            clip_duration = (end_timestamp - start_timestamp) * scale_factor  # In seconds

            # Convert to milliseconds for align file
            original_duration_ms = (end_timestamp - start_timestamp)

            # Handle multiple occurrences of the same word
            word_count[word] += 1
            word_index = word_count[word]  # Get occurrence number
            unique_filename = f"{word}_{word_index}" if word_count[word] > 1 else word

            # Define output file paths (now using .mpg)
            output_video = os.path.join(video_output_dir, f'{unique_filename}_clip.mpg')
            output_align_file = os.path.join(align_output_dir, f'{unique_filename}.align')

            # Extract the video segment for the specific word
            extract_clip(video_path, output_video, start_time, clip_duration)

            # Create the align file (WITHOUT _1, _2 in the word)
            create_align_file(word, original_duration_ms, output_align_file)

        print(f"Completed processing {video_file}")

print(" All videos and align files have been successfully processed!")
