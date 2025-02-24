import os
import random
import subprocess
import pandas as pd

# Define base paths
video_output_base = 'result_video_files'
align_output_base = 'result_align_files'
final_video_folder = 'final_videos'
final_align_folder = 'final_alignments'

# Create output folders if they don't exist
os.makedirs(final_video_folder, exist_ok=True)
os.makedirs(final_align_folder, exist_ok=True)

# Process each directory separately
for subdir in os.listdir(video_output_base):
    subdir_path = os.path.join(video_output_base, subdir)
    if not os.path.isdir(subdir_path):
        continue
    
    final_video = os.path.join(final_video_folder, f'final_output_{subdir}.mpg')
    final_align = os.path.join(final_align_folder, f'final_output_{subdir}.align')
    
    # Collect video clips
    video_clips = [os.path.join(subdir_path, f) for f in os.listdir(subdir_path) if f.endswith('.mpg')]
    
    # Sort & Shuffle clips
    sil_clip = [clip for clip in video_clips if clip.endswith('sil_clip.mpg')]
    sil_2_clip = [clip for clip in video_clips if clip.endswith('sil_2_clip.mpg')]
    other_clips = [clip for clip in video_clips if clip not in sil_clip + sil_2_clip]
    random.shuffle(other_clips)
    
    final_clip_order = sil_clip + other_clips + sil_2_clip
    
    # Create concat file
    concat_file = f'concat_list_{subdir}.txt'
    with open(concat_file, 'w') as f:
        for clip in final_clip_order:
            f.write(f"file '{clip}'\n")
    
    # Run FFmpeg
    concat_command = [
        'ffmpeg', '-f', 'concat', '-safe', '0', '-i', concat_file, 
        '-c', 'copy', '-y', final_video
    ]
    subprocess.run(concat_command)
    print(f"Final video saved: {final_video}")
    
    # Process align files
    align_files = []
    for clip in final_clip_order:
        align_file = os.path.join(align_output_base, subdir, os.path.basename(clip).replace('_clip.mpg', '.align'))
        if os.path.exists(align_file):
            align_files.append(align_file)
    
    final_align_data = []
    current_time = 0
    
    for align_file in align_files:
        df = pd.read_csv(align_file, delim_whitespace=True, names=['start', 'end', 'word'])
        df['start'] += current_time
        df['end'] += current_time
        final_align_data.append(df)
        current_time = df['end'].max()
    
    if final_align_data:
        final_df = pd.concat(final_align_data, ignore_index=True)
        final_df.to_csv(final_align, sep=' ', index=False, header=False)
        print(f"Final align file saved: {final_align}")
    else:
        print(f"No align files found for {subdir}, skipping align merge.")
