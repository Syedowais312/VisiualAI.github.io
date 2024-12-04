Key Features

   !. Object Detection with YOLOv8:
        The script uses the YOLOv8 object detection model (yolov8n.pt for the lightweight version) to detect objects in real-time from the camera feed.
        Bounding boxes, class labels, and confidence scores are drawn for each detected object.

   2. Distance Estimation:
        For specific objects, the script estimates the distance from the camera using the pinhole camera model:
        Distance=Known Object Width×Focal LengthPixel Width of the Object
        Distance=Pixel Width of the ObjectKnown Object Width×Focal Length​
        Known real-world widths of objects like chairs, dogs, and cats are defined in a dictionary (KNOWN_WIDTHS).

   3. Text-to-Speech (TTS):
        A pyttsx3 engine is used for speech synthesis to announce detected objects and their estimated distances.
        The announcements are handled in a separate thread to prevent delays in the main detection loop.
      Workflow of the Script

  Workflow of the Script

   1. Initialization:
        The YOLOv8 model is loaded.
        The camera feed is opened, and settings like resolution are adjusted.

   2. Frame Processing:
        Each captured frame is resized and passed to YOLOv8 for inference.
        Detected objects are processed to calculate distances for known object classes.
        Bounding boxes and labels are drawn on the frame.

   3.  Speech Announcement:
        If the distance is valid, a TTS announcement is made asynchronously for the detected object and its distance.

   4. Performance Profiling:
        Frame processing time is calculated and displayed for performance monitoring.

   5. Termination:
        The camera and OpenCV windows are properly released when the user exits.
      [NOTE:-to terminate the code press 'Q' ,'q']

Requirements

To run this script, you need:

   1. Python Libraries:
        opencv-python
        numpy
        pyttsx3
        ultralytics (for YOLOv8)
        Other dependencies for YOLOv8, such as PyTorch.
    2. YOLOv8 Weights:
        Ensure the yolov8n.pt model file is downloaded.

 Use Cases
 
   1. Real-time object detection for accessibility (e.g., announcing objects for visually impaired users).
   2. Robotics and automation systems requiring distance estimation.
   3. Educational purposes for learning object detection and computer vision concepts.
