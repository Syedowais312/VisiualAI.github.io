import cv2
import numpy as np
import pyttsx3
import threading
import time
from ultralytics import YOLO  # YOLOv8 library

# Initializing  the TTS engine
engine = pyttsx3.init()

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')  # Use 'yolov8s.pt' for slightly better performance with minimal overhead

# Open the default camera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Set  resolution for the camera displace
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Frame skipping 
skip_frames = 2
frame_count = 0

# Camera parameters 
FOCAL_LENGTH = 615  

# Known real-world widths for common objects (in m)
KNOWN_WIDTHS = {
    "chair": 0.5,
    "dog": 0.6,
    "cat": 0.3,
    "plant": 0.4,
    "phone": 0.08,
    "person": 0.5, 
}

# Function for asynchronous tts
def speak(text):
    engine.say(text)
    engine.runAndWait()

# Function to estimate distance
def estimate_distance(known_width, focal_length, pixel_width):
    if pixel_width > 0:  
        return (known_width * focal_length) / pixel_width
    return -1  

while True:
    start_time = time.time()  

    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    
    frame_count += 1
    if frame_count % skip_frames != 0:
        continue

    # Resize frame for YOLO processing 
    resized_frame = cv2.resize(frame, (640, 640))

    # Perform object detection with YOLOv8
    results = model(resized_frame, conf=0.5)  # Set a confidence threshold of 0.5


    for result in results[0].boxes:  # Access the results properly for YOLOv8
        x1, y1, x2, y2 = result.xyxy[0]  # Bounding box coordinates (x1, y1, x2, y2)
        confidence = result.conf[0]  # Confidence score
        class_id = int(result.cls[0])  # Class id
        class_name = model.names[class_id]  

        
        if class_name in KNOWN_WIDTHS:
            # Calculate bounding box width (in pixels)
            pixel_width = x2 - x1

            # Estimate distance using the known width for this class
            distance = estimate_distance(KNOWN_WIDTHS[class_name], FOCAL_LENGTH, pixel_width)

            # Draw bounding box and label on the frame
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            label = f"{class_name}: {confidence:.2f}, Distance: {distance:.2f} m"
            cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Use TTS to announce the detected object and its distance
            if distance > 0:  
                speech_text = f"Detected {class_name} at a distance of approximately {distance:.2f} meters"
                threading.Thread(target=speak, args=(speech_text,)).start()

    # Display the results
    cv2.imshow('YOLOv8 Live Camera Feed', frame)

    # Calculate and print frame processing time for profiling
    print(f"Frame processing time: {time.time() - start_time:.2f} seconds")

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
