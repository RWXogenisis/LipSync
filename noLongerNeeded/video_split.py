import os
import re
import random
from moviepy.editor import VideoFileClip

def convert_timestamp_to_seconds(timestamp):
    h, m, s = map(float, timestamp.replace(',', '.').split(':'))
    return h * 3600 + m * 60 + s

def parse_transcript(file_path, duration_target, split_longer=False):
    segments = []
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
        matches = re.findall(r"(\d{2}:\d{2}:\d{2}\.\d{3}) --> (\d{2}:\d{2}:\d{2}\.\d{3})\n(.+?) \((\d+\.\d{2})s\)", content, re.DOTALL)
        
        for start, end, text, duration in matches:
            duration = float(duration)
            start_sec = convert_timestamp_to_seconds(start)
            
            if split_longer and duration >= 2 * duration_target:
                segments.append((start_sec, start_sec + duration_target, text[:len(text)//2], duration_target))
                segments.append((start_sec + duration_target, start_sec + 2 * duration_target, text[len(text)//2:], duration_target))
            else:
                segments.append((start_sec, start_sec + duration_target, text, duration))
    return segments

def save_align_file(text, duration, index, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"clip_{index+1}.align")
    
    words = re.findall(r"\b\w+\b", text)  # Extract words only, removing special characters and symbols
    
    if 3.60 <= duration <= 3.99 and len(words) > 2:
        words = words[:-2]  # Remove last two words if duration is between 3.60 and 3.99 seconds
    elif 3.40 <= duration <= 3.59 and len(words) > 1:
        words = words[:-1]  # Remove last word if duration is between 3.40 and 3.59 seconds
    
    timestamps = sorted(random.sample(range(74500), len(words)))
    
    with open(output_path, 'w', encoding='utf-8') as file:
        for word, ts in zip(words, timestamps):
            file.write(f"{word} {ts}\n")
    print(f"Saved align file: {output_path}")

def split_video(video_file, segments, output_video_dir, output_align_dir):
    os.makedirs(output_video_dir, exist_ok=True)
    video = VideoFileClip(video_file)
    
    for idx, (start, end, text, duration) in enumerate(segments):
        clip = video.subclip(start, end)
        output_path = os.path.join(output_video_dir, f"clip_{idx+1}_{start:.2f}-{end:.2f}.mpg")
        clip.write_videofile(output_path, codec="mpeg1video")
        print(f"Saved video: {output_path}")
        
        save_align_file(text, duration, idx, output_align_dir)

# File paths (update accordingly)
video_file_path = "Jeff Dean & Noam Shazeer – 25 years at Google： from PageRank to AGI [v0gjI__RyCY].mp4"
transcript_file_path_3s = "jeff_video_length/3_second_sentences/sentences.txt"
transcript_file_path_6s = "jeff_video_length/6_second_sentences/sentences.txt"

# Extract segments
segments_3s = parse_transcript(transcript_file_path_3s, 3.00)
segments_6s = parse_transcript(transcript_file_path_6s, 3.00, split_longer=True)  # Split 6s into two 3s clips

# Merge both for processing
all_segments = segments_3s + segments_6s

# Split video and generate align files
split_video(video_file_path, all_segments, "output_clips", "result_align")

print("Video processing and alignment generation completed.")
