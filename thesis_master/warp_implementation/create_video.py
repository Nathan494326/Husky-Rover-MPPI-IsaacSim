import os
import cv2
import glob

def create_video_from_frames(frame_folder="frame_folder", output_video="trajectory.mp4", fps=10):
    """
    Creates a video from saved frames.

    Args:
        frame_folder (str): Path to the folder containing frames.
        output_video (str): Output video file name.
        fps (int): Frames per second.
    """
    # List all files in the directory and sort them by creation time (or modification time)
    frame_files = [os.path.join(frame_folder, f) for f in os.listdir(frame_folder) if f.endswith('.png')]
    
    # Sort files by creation or modification time
    frame_files.sort(key=lambda x: os.path.getctime(x))  # Sort by creation time (alternative: use getmtime for modification time)
    
    if not frame_files:
        print("No frames found!")
        return
    
    # Read first image to get dimensions
    frame = cv2.imread(frame_files[0])
    height, width, layers = frame.shape
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    
    # Add frames
    for frame_file in frame_files:
        video.write(cv2.imread(frame_file))
    
    video.release()
    print(f"Video saved as {output_video}")

# Run this after the loop
create_video_from_frames()