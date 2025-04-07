import cv2
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor

class VideoProcessor:
    """Handles the processing of videos including segmenting, converting, resizing, and adjusting frame rates."""
    
    def __init__(self, output_folder="videoSegments", fps=26.33):
        """
        Initializes the VideoProcessor with a specified output folder and frames per second.
        
        Args:
            output_folder (str): Folder where processed video segments will be saved.
            fps (float): Frames per second to use when writing video segments.
        """
        self.output_folder = output_folder
        self.fps = fps
        os.makedirs(self.output_folder, exist_ok=True)
    
    def _clear_output_folder(self):
        """
        Deletes all files in the output folder before starting a new video processing task.
        """
        for file in os.listdir(self.output_folder):
            file_path = os.path.join(self.output_folder, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
        print(f"Cleared all files in {self.output_folder}")

    def _get_video_url(self, youtube_url):
        """
        Uses yt-dlp to fetch the direct URL of the best quality MP4 video stream.
        
        Args:
            youtube_url (str): URL of the YouTube video.
        
        Returns:
            str: Direct URL to the video stream or None if there was an error.
        """
        try:
            # Run yt-dlp to get the video URL
            result = subprocess.run(
                ["yt-dlp", "-g", "-f", "best[ext=mp4]", youtube_url], 
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )
            return result.stdout.strip()
        except Exception as e:
            print(f"Error fetching video URL: {e}")
            return None

    def _convert_to_mpg(self, input_path, output_path):
        """
        Converts a video file to MPEG-2 format using ffmpeg.
        
        Args:
            input_path (str): Path to the input video file.
            output_path (str): Path to the output MPEG-2 file.
        
        Returns:
            bool: True if conversion succeeded, False otherwise.
        """
        # FFmpeg command to convert the video to MPEG-2 format
        command = [
            "ffmpeg", "-i", input_path, "-c:v", "mpeg2video", "-q:v", "2",
            "-c:a", "mp2", "-b:a", "192k", output_path
        ]
        return subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE).returncode == 0

    def _check_integrity(self, video_path):
        """
        Verifies that a video can be read frame by frame and contains valid frames.
        
        Args:
            video_path (str): Path to the video file.
        
        Returns:
            bool: True if video integrity is valid, False otherwise.
        """
        cap = cv2.VideoCapture(video_path)  # Open the video file
        if not cap.isOpened():
            return False  # Return False if the video could not be opened
        processed_frames = 0
        while True:
            ret, frame = cap.read()  # Read each frame
            if not ret:
                break
            if frame is None or frame.size == 0:  # Check for invalid frames
                cap.release()
                return False
            processed_frames += 1
        cap.release()
        return processed_frames > 0  # Return True if there are valid frames

    def _resize(self, input_path, output_path, width=360, height=288):
        """
        Resizes a video to a specified resolution using ffmpeg.
        
        Args:
            input_path (str): Path to the input video.
            output_path (str): Path to the resized output video.
            width (int): Target width.
            height (int): Target height.
        
        Returns:
            bool: True if resizing was successful, False otherwise.
        """
        # FFmpeg command to resize the video
        command = [
            "ffmpeg", "-i", input_path, "-vf", f"scale={width}:{height}", "-c:v", "mpeg2video", "-q:v", "2",
            "-c:a", "mp2", "-b:a", "192k", output_path
        ]
        return subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE).returncode == 0

    def _change_fps(self, input_path, output_path, target_fps=26.33):
        """
        Changes the frame rate of a video using ffmpeg.
        
        Args:
            input_path (str): Path to the input video.
            output_path (str): Path to the output video with adjusted FPS.
            target_fps (float): Desired frame rate.
        
        Returns:
            bool: True if FPS change succeeded, False otherwise.
        """
        # FFmpeg command to change the FPS of the video
        command = [
            "ffmpeg", "-i", input_path, "-r", str(target_fps), "-c:v", "mpeg2video", "-q:v", "2",
            "-c:a", "mp2", "-b:a", "192k", output_path
        ]
        return subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE).returncode == 0

    def _process_segment(self, segment_index):
        """
        Processes a video segment by converting, checking integrity, resizing,
        adjusting FPS, and cleaning up intermediate files.
        
        Args:
            segment_index (int): Index of the segment to process.
        
        Returns:
            str: Message indicating the result of the processing.
        """
        # Define paths for input and output video files
        inputSegment = f"{self.output_folder}/segment_{segment_index}.mp4"
        mpgSegment = f"{self.output_folder}/segment_{segment_index}.mpg"
        resizedSegment = f"{self.output_folder}/segment_{segment_index}_resized.mpg"
        finalSegment = f"{self.output_folder}/segment_{segment_index}_final.mpg"

        if not self._convert_to_mpg(inputSegment, mpgSegment):
            return f"Conversion failed for {inputSegment}"

        if not self._check_integrity(mpgSegment):
            return f"Integrity check failed for {mpgSegment}"

        if not self._resize(mpgSegment, resizedSegment):
            return f"Resizing failed for {mpgSegment}"

        if not self._change_fps(resizedSegment, finalSegment):
            return f"FPS conversion failed for {resizedSegment}"

        if not self._check_integrity(finalSegment):
            return f"Final integrity check failed for {finalSegment}"

        # Remove intermediate files
        os.remove(inputSegment)
        os.remove(mpgSegment)
        os.remove(resizedSegment)
        
        return f"Processed segment saved at {finalSegment}"

    def _trim_to_multiple_of_three(self, input_path, output_path):
        """
        Trims a video to the nearest lower multiple of 3 seconds.
        
        Args:
            input_path (str): Path or URL to the input video.
            output_path (str): Path to save the trimmed video.
        
        Returns:
            str: Path to the trimmed video if successful, None otherwise.
        """
        try:
            # Get the duration of the video
            result = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of",
                "default=noprint_wrappers=1:nokey=1", input_path],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )
            duration = float(result.stdout.strip())
            trimmed_duration = int(duration // 3) * 3
            if trimmed_duration <= 0:
                return None

            # Trim the video to the desired length
            command = [
                "ffmpeg", "-y", "-i", input_path, "-t", str(trimmed_duration),
                "-c", "copy", output_path
            ]
            subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            return output_path
        except Exception as e:
            print(f"Error trimming video: {e}")
            return None

    def process_video(self, video_path):
        """
        Processes a video by splitting it into segments every 3 seconds,
        and applying conversion, resizing, and FPS adjustment on each segment.
        
        Args:
            video_path (str): YouTube URL or local video file path.
        
        Returns:
            tuple: (success (bool), message (str))
        """
        self._clear_output_folder()  # Clear the output folder before starting

        processed_url = self._get_video_url(video_path)  # Get the direct video URL from YouTube
        trimmed_video_path = os.path.join(self.output_folder, "trimmed_input.mp4")

        if not self._trim_to_multiple_of_three(processed_url, trimmed_video_path):
            return False, "Failed to trim video to a valid length."

        cap = cv2.VideoCapture(trimmed_video_path)  # Open the trimmed video for processing
        
        if not cap.isOpened():
            print("Error: Could not open video.")
            return False, "Could not open video."
        
        frame_interval = int(self.fps * 3)  # Every 3 seconds
        frame_count = 0
        segment_count = 0
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Video codec for mp4
        segment_writer = None
        
        # Using ThreadPoolExecutor for parallel processing of video segments
        with ThreadPoolExecutor() as executor:
            futures = []
            
            while True:
                ret, frame = cap.read()  # Read each frame
                if not ret:
                    break
                
                if frame_count % frame_interval == 0:  # Every 3 seconds, start a new segment
                    if segment_writer:
                        segment_writer.release()  # Release the previous segment writer
                        futures.append(executor.submit(self._process_segment, segment_count - 1))  # Process the previous segment
                    
                    segment_filename = f"{self.output_folder}/segment_{segment_count}.mp4"
                    height, width, _ = frame.shape
                    segment_writer = cv2.VideoWriter(segment_filename, fourcc, self.fps, (width, height))  # Create a new writer for the next segment
                    segment_count += 1
                
                if segment_writer:
                    segment_writer.write(frame)  # Write the frame to the current segment
                
                frame_count += 1
            
            # Release the last segment writer and submit its processing
            if segment_writer:
                segment_writer.release()
                futures.append(executor.submit(self._process_segment, segment_count - 1))

            # Wait for all segment processing to finish
            for future in futures:
                print(future.result())
        
        cap.release()
        os.remove(trimmed_video_path)  # Clean up the trimmed video file
        print(f"Processing complete! {segment_count} segments processed.")
        return True, f"{segment_count} segments processed successfully."
