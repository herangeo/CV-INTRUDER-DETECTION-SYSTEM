import cv2
import numpy as np
import pygame
import time
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import threading

class HazardDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Advanced Hazardous Object Detection")
        self.root.configure(bg="#2C3E50")  # Modern background color
        
        # Initialize pygame for playing MP3 alerts
        pygame.mixer.init()

        # Load YOLO
        self.MODEL_WEIGHTS = "yolov3.weights"
        self.MODEL_CONFIG = "yolov3.cfg"
        self.LABELS = "coco.names"

        # Load network and labels
        self.net = cv2.dnn.readNet(self.MODEL_WEIGHTS, self.MODEL_CONFIG)
        layer_names = self.net.getLayerNames()

        # Get output layer indices (adjusted for newer OpenCV versions)
        self.output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]

        # Load class labels
        with open(self.LABELS, "r") as f:
            self.labels = [line.strip() for line in f.readlines()]

        # Enhanced detection thresholds
        self.detection_thresholds = {
            "default": 0.3,    # Default confidence threshold
            "knife": 0.5,       # Higher threshold for knife detection
            "gun": 0.6,         # Higher threshold for gun detection
            "fire": 0.4,        # Adjusted threshold for fire
            "person": 0.5       # Threshold for person detection
        }

        # Expanded hazardous objects and sound alerts
        self.hazardous_objects = {
            "gun": "high.mp3",       # High alert for guns
            "fire": "high.mp3",      # High alert for fire
            "knife": "high.mp3",     # High alert for knives
            "person": "high.mp3"     # High alert for person (intruder)
        }

        # Comprehensive list of knife and dangerous object variations
        self.knife_variations = [
            "knife", 
            "scissors", 
            "blade", 
            "cutter", 
            "dagger", 
            "cutlery",
            "sword"
        ]

        # General hazardous objects for low-level alerts
        self.general_hazardous_objects = [
            "bottle", 
            "lighter", 
            "axe", 
            "fork", 
            "scissors", 
            "blade"
        ]

        # Set up webcam (real-time video capture)
        self.cap = cv2.VideoCapture(0)  # Use 0 for default webcam
        
        # Alert management
        self.alert_triggered = False
        self.previous_alert = ""

        # Create GUI elements
        self.create_gui_elements()

        # Start detection thread
        self.is_running = True
        self.detection_thread = threading.Thread(target=self.detect_objects)
        self.detection_thread.daemon = True
        self.detection_thread.start()

    def create_gui_elements(self):
        """Create and set up GUI components"""
        # Video display label
        self.video_label = tk.Label(self.root)
        self.video_label.pack(padx=10, pady=10)

        # Status label with enhanced styling
        self.status_label = tk.Label(
            self.root, 
            text="Detection Status: Ready", 
            font=("Helvetica", 14), 
            fg="white", 
            bg="#2C3E50"
        )
        self.status_label.pack(pady=5)

        # Stop detection button
        self.stop_button = tk.Button(
            self.root, 
            text="Stop Detection", 
            command=self.stop_detection, 
            font=("Helvetica", 12), 
            bg="#E74C3C", 
            fg="white"
        )
        self.stop_button.pack(pady=10)

    def stop_detection(self):
        """Stop the detection process and close the application"""
        self.is_running = False
        self.cap.release()
        cv2.destroyAllWindows()
        self.root.quit()

    def play_sound(self, sound_file):
        """Enhanced sound playing function"""
        try:
            pygame.mixer.music.load(sound_file)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
        except Exception as e:
            print(f"Error playing sound {sound_file}: {e}")

    def detect_objects(self):
        """Advanced object detection method"""
        while self.is_running:
            # Capture frame-by-frame
            ret, frame = self.cap.read()
            if not ret:
                break

            height, width, channels = frame.shape

            # Improved image preprocessing for YOLO
            blob = cv2.dnn.blobFromImage(
                frame, 
                1/255.0,  # Normalize pixel values
                (416, 416), 
                swapRB=True,  # Swap Red and Blue channels
                crop=False
            )
            self.net.setInput(blob)
            outs = self.net.forward(self.output_layers)

            # Detection data structures
            class_ids = []
            confidences = []
            boxes = []
            alert_message = ""
            play_alert = None

            # Process detections
            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    label = self.labels[class_id]

                    # Dynamic confidence threshold
                    threshold = self.detection_thresholds.get(label, self.detection_thresholds["default"])
                    
                    # Special handling for knife and similar objects
                    if label in self.knife_variations:
                        threshold = self.detection_thresholds.get("knife", threshold)

                    # Confidence-based detection
                    if confidence > threshold:
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)

                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)

                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)

            # Advanced Non-Maximum Suppression
            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.3, 0.2)

            # Draw detections and process alerts
            for i in range(len(boxes)):
                if i in indexes:
                    x, y, w, h = boxes[i]
                    label = str(self.labels[class_ids[i]])
                    confidence = str(round(confidences[i], 2))

                    # Threat level color coding
                    if (label in self.hazardous_objects or 
                        label in self.knife_variations or 
                        label in self.general_hazardous_objects):
                        color = (0, 0, 255)  # Red for threats
                    else:
                        color = (0, 255, 0)  # Green for non-threats

                    # Draw bounding box and label
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(frame, f"{label} {confidence}", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                    # Alert logic for hazardous objects
                    if (label in self.knife_variations or 
                        label in self.hazardous_objects):
                        alert_message = f"High Threat: {label.capitalize()} detected!"
                        play_alert = "high.mp3"
                    elif label in self.general_hazardous_objects:
                        alert_message = f"Low Threat: {label.capitalize()} detected!"
                        play_alert = "low.mp3"

            # Update status label
            if alert_message:
                self.status_label.config(
                    text=alert_message, 
                    fg="yellow" if "Low" in alert_message else "red"
                )
                # Play alert sound if different from previous
                if play_alert and play_alert != self.previous_alert:
                    threading.Thread(target=self.play_sound, args=(play_alert,)).start()
                    self.previous_alert = play_alert
            else:
                self.status_label.config(text="Detection Status: No Threats", fg="green")

            # Convert frame for Tkinter display
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)
            pil_img = pil_img.resize((640, 480), Image.LANCZOS)
            img_tk = ImageTk.PhotoImage(image=pil_img)
            
            # Update video label
            self.video_label.config(image=img_tk)
            self.video_label.image = img_tk

            # Reduce CPU usage
            time.sleep(0.025)

def main():
    root = tk.Tk()
    root.geometry("680x650")  # Set window size
    app = HazardDetectionApp(root)
    root.protocol("WM_DELETE_WINDOW", app.stop_detection)
    root.mainloop()

if __name__ == "__main__":
    main()