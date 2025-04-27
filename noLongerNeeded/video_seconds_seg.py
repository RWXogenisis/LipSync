import webvtt
import os
from moviepy.editor import VideoFileClip

def extract_sentences_with_timestamps(vtt_file, video_file):
    video = VideoFileClip(video_file)
    output_directory = "/Users/aswinkumarv/Desktop/video_length/jeff_video_length"
    os.makedirs(output_directory, exist_ok=True)
    file_handles = {}
    duration_counts = {}
    
    for caption in webvtt.read(vtt_file):
        start_time = caption.start
        end_time = caption.end
        duration = convert_timestamp_to_seconds(end_time) - convert_timestamp_to_seconds(start_time)
        text = caption.text.replace('\n', ' ')  # Remove new lines within captions
        output_line = f"{start_time} --> {end_time}\n{text} ({duration:.2f}s)\n\n"
        
        print(output_line)
        
        duration_category = int(duration)  # Group by whole number seconds
        duration_folder = os.path.join(output_directory, f"{duration_category}_second_sentences")
        os.makedirs(duration_folder, exist_ok=True)
        duration_file = os.path.join(duration_folder, f"sentences.txt")
        
        if duration_category not in file_handles:
            file_handles[duration_category] = open(duration_file, 'a', encoding='utf-8')
        file_handles[duration_category].write(output_line)
        
        duration_counts[duration_category] = duration_counts.get(duration_category, 0) + 1
    
    for handle in file_handles.values():
        handle.close()
    
    count_file = os.path.join(output_directory, "sentence_counts_2.txt")
    with open(count_file, 'w', encoding='utf-8') as count_out:
        for duration, count in sorted(duration_counts.items()):
            count_out.write(f"{duration}-second sentences: {count}\n")
    
    print("Sentence count summary saved to", count_file)

def convert_timestamp_to_seconds(timestamp):
    h, m, s = map(float, timestamp.replace(',', '.').split(':'))
    return h * 3600 + m * 60 + s

# File paths (update these paths accordingly)
vtt_file_path = "Jeff Dean & Noam Shazeer – 25 years at Google_ from PageRank to AGI.vtt"
video_file_path = "Jeff Dean & Noam Shazeer – 25 years at Google： from PageRank to AGI [v0gjI__RyCY].mp4"

# Run the extraction function
extract_sentences_with_timestamps(vtt_file_path, video_file_path)

print("Segregated transcripts saved based on sentence duration ranges.")
