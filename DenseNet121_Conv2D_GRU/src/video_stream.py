import cv2

class VideoStream:
    
    def __init__(self, filepath):
        self.cap = cv2.VideoCapture(filepath)
        if not self.cap.isOpened():
            raise Exception('Video stream doesn\'t open!')
    
    def __enter__(self):
        return self.cap
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cap.release()