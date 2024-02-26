import cv2
from mtcnn import MTCNN
import math
import numpy as np

cap = cv2.VideoCapture(0)
detector = MTCNN()

while True:
    ret, frame = cap.read()
    face_imgs = detector.detect_faces(frame)
    max_confidence = 0
    max_face = None
    key_points = None
    for face_img in face_imgs:
        if face_img.get("confidence", 0) > max_confidence:
            max_confidence = face_img["confidence"]
            max_face = face_img["box"]
            key_points = face_img["keypoints"]

    if max_face is None:
        continue
    x, y, w, h = max_face
    frame = frame[y: y+h, x:x+w, :]
    x1, y1 = key_points["right_eye"]
    x2, y2 = key_points["left_eye"]
    a = abs(y1 - y2)
    b = abs(x2 - x1)
    c = math.sqrt(a * a + b * b)
    cos_alpha = (b * b + c * c - a * a) / (2 * b * c)
    alpha = np.arccos(cos_alpha)
    alpha = (alpha * 180) / math.pi
    if y1 - y2 > 0:
        # Rotate the frame "Counterclockwise"
        M = cv2.getRotationMatrix2D((frame.shape[1] / 2, frame.shape[0] / 2), alpha, 0.6)
        rotated_frame = cv2.warpAffine(frame, M, (frame.shape[1], frame.shape[0]))

    elif y1 - y2 < 0:
        # Rotate the frame "Clockwise"
        M = cv2.getRotationMatrix2D((frame.shape[1] / 2, frame.shape[0] / 2), -alpha, 0.6)
        rotated_frame = cv2.warpAffine(frame, M, (frame.shape[1], frame.shape[0]))

    else:
        rotated_frame = frame

    cv2.imshow("Rotated Frame", rotated_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

