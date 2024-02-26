Face Detection and Rotation Correction with MTCNN and OpenCV:
This Python script utilizes OpenCV and MTCNN libraries to achieve face detection and rotation correction based on the relative position of the eyes.

Description:

The script captures video frames from your webcam.
It employs MTCNN to detect faces in each frame.
It focuses on the face with the highest confidence score.
The script then calculates the angle of rotation based on the vertical difference between the left and right eyes.
Finally, it corrects the frame's orientation by rotating it clockwise or counterclockwise depending on the eye tilt.

Requirements:

Python 3.4

OpenCV library (pip install opencv-python)

MTCNN library (pip install mtcnn)

Usage:

-Save the script as face_detection_rotation.py.

-Install the required libraries.

-Run the script from your terminal:
