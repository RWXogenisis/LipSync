import webvtt
import os
from moviepy.editor import VideoFileClip

def extract_sentences_with_timestamps(vtt_file, output_txt_file, video_file):
    video = VideoFileClip(video_file)
    with open(output_txt_file, 'w', encoding='utf-8') as out_file:
        for caption in webvtt.read(vtt_file):
            start_time = caption.start
            end_time = caption.end
            duration = convert_timestamp_to_seconds(end_time) - convert_timestamp_to_seconds(start_time)
            text = caption.text.replace('\n', ' ')  # Remove new lines within captions
            out_file.write(f"{start_time} --> {end_time}\n{text} ({duration:.2f}s)\n\n")
            print(f"{start_time} --> {end_time}\n{text} ({duration:.2f}s)\n")

def convert_timestamp_to_seconds(timestamp):
    h, m, s = map(float, timestamp.replace(',', '.').split(':'))
    return h * 3600 + m * 60 + s

# File paths (update these paths accordingly)
vtt_file_path = "Teaching Software Development _ @HiteshCodeLab _ Beyond Coding Podcast #55.vtt"
video_file_path = "Teaching Software Development ｜ @HiteshCodeLab ｜ Beyond Coding Podcast #55 [oGXhLcFXDak].mp4"
output_text_file_path = "transcript_with_timestamps_2.txt"

# Run the extraction function
extract_sentences_with_timestamps(vtt_file_path, output_text_file_path, video_file_path)

print(f"Transcript with timestamps saved to: {output_text_file_path}")
