
import cv2
import numpy as np
import pytesseract
import pyttsx3
import os

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Load YOLO
net = cv2.dnn.readNet("C:/Miniproject/yolov3.weights", "C:/Miniproject/yolov3.cfg.txt")
with open("C:/Miniproject/coco.names.txt", "r") as f:
    classes = [line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

def detect_objects(frame):
    height, width, channels = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
    
    class_ids = []
    confidences = []
    boxes = []
    detected_objects = []  # List to store detected objects

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            detected_objects.append(label)  # Add label to the detected objects list
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, label, (x, y + 30), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
    
    return frame, detected_objects

def extract_text(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray)
    return text

def speak(text):
    engine.say(text)
    engine.runAndWait()

# Load the camera
cap = cv2.VideoCapture(0)

# Check if camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)  # Ensure window is created properly

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break
    
    frame, detected_objects = detect_objects(frame)
    text = extract_text(frame)
    
    if detected_objects:
        objects_text = "Detected objects: " + ", ".join(detected_objects)
        print(objects_text)
        speak(objects_text)
    
    if text.strip():
        text_output = f"Detected text: {text.strip()}"
        print(text_output)
        speak(text_output)
    
    cv2.imshow("Frame", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
        break

cap.release()
cv2.destroyAllWindows()