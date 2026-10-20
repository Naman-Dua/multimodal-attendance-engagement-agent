import cv2
import numpy as np
import os

class FaceDetector:
    def __init__(self):
        # Use OpenCV's built-in Haarcascade for face detection
        # This is more robust across Python versions than MediaPipe
        self.face_cascade = None
        if hasattr(cv2, 'CascadeClassifier') and hasattr(cv2, 'data') and hasattr(cv2.data, 'haarcascades'):
            haarcascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            try:
                self.face_cascade = cv2.CascadeClassifier(haarcascade_path)
                if self.face_cascade.empty():
                    print("Error: Could not load face cascade xml")
            except Exception as e:
                print(f"Warning loading cascade: {e}")

    def process_frame(self, frame):
        # Convert to grayscale for detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        detections = []
        if self.face_cascade and not self.face_cascade.empty():
            # Detect faces
            # scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
            for (x, y, w, h) in faces:
                # Format to match previous output: bbox (x, y, w, h), score (1.0 placeholder)
                detections.append({"bbox": (int(x), int(y), int(w), int(h)), "score": 1.0})
        else:
            # Fallback face locator if CascadeClassifier is unavailable in OpenCV 5.0
            h, w = frame.shape[:2]
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            # Standard human skin tone mask range in HSV
            mask = cv2.inRange(hsv, (0, 30, 60), (25, 255, 255))
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > (w * h * 0.015):  # Candidate face region (>1.5% of frame area)
                    bx, by, bw, bh = cv2.boundingRect(cnt)
                    aspect_ratio = float(bh) / max(1, bw)
                    if 0.9 <= aspect_ratio <= 1.8:  # Head/face aspect ratio
                        detections.append({"bbox": (int(bx), int(by), int(bw), int(bh)), "score": 0.85})
                        break  # Match primary presenter / student in view

        return detections
