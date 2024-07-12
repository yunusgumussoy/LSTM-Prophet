# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 14:54:05 2024

@author: Yunus
"""

# pip install opencv-python
# pip install opencv-python-headless

import cv2
import os

# Video file path
video_dizin = "C:/Users/yunus/Downloads/1.mp4"

# Output directory on the desktop
masaustu = os.path.join(os.path.expanduser("~"), "Desktop")
dizin = os.path.join(masaustu, "Resimler")

# Create the output directory if it does not exist
if not os.path.exists(dizin):
    os.makedirs(dizin)

resim_dosya = dizin

# Open the video file
cap = cv2.VideoCapture(video_dizin)

# Check if video opened successfully
if not cap.isOpened():
    print(f"Error: Could not open video file {video_dizin}")
else:
    # Get frames per second (fps) and total frame count
    fps = cap.get(cv2.CAP_PROP_FPS)
    toplam = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"FPS: {fps}")
    print(f"Total Frames: {toplam}")

    # Iterate over frames and save images
    for i in range(0, toplam, int(fps)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        j, k = cap.read()

        if j:
            resim = os.path.join(resim_dosya, f"resim_{i}.jpg")
            cv2.imwrite(resim, k)
            print(f"Frame {i} saved as {resim}")
        else:
            print(f"Error: Could not read frame {i}")

    # Release the video capture object
    cap.release()

print("Processing completed.")
