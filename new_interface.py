import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import threading
import time
from collections import deque

# Load the trained model
model = load_model('attention_detection_model.h5')

# Constants
IMG_HEIGHT, IMG_WIDTH = 224, 224
CAPTURE_INTERVAL = 1  # seconds
WINDOW_SIZE = 5  # number of recent predictions to consider
MAJORITY_THRESHOLD = 0.6  # percentage needed for majority

class AttentionMonitor:
    def __init__(self, window, video_source=0):
        self.window = window
        self.window.title("Attention Monitor")
        self.video_source = video_source

        self.vid = cv2.VideoCapture(self.video_source)
        self.canvas = tk.Canvas(window, width=IMG_WIDTH, height=IMG_HEIGHT)
        self.canvas.pack()

        self.btn_start = ttk.Button(window, text="Start Monitoring", command=self.start_monitoring)
        self.btn_start.pack(padx=10, pady=10)

        self.btn_stop = ttk.Button(window, text="Stop Monitoring", command=self.stop_monitoring)
        self.btn_stop.pack(padx=10, pady=10)
        self.btn_stop.config(state='disabled')

        self.label_status = ttk.Label(window, text="Status: Not Monitoring")
        self.label_status.pack(padx=10, pady=10)

        self.label_result = ttk.Label(window, text="Overall Result: N/A")
        self.label_result.pack(padx=10, pady=10)

        self.is_monitoring = False
        self.recent_predictions = deque(maxlen=WINDOW_SIZE)
        self.overall_result = None

        self.update()
        self.window.mainloop()

    def start_monitoring(self):
        self.is_monitoring = True
        self.btn_start.config(state='disabled')
        self.btn_stop.config(state='normal')
        self.label_status.config(text="Status: Monitoring")
        threading.Thread(target=self.monitor_attention, daemon=True).start()

    def stop_monitoring(self):
        self.is_monitoring = False
        self.btn_start.config(state='normal')
        self.btn_stop.config(state='disabled')
        self.label_status.config(text="Status: Not Monitoring")
        self.label_result.config(text="Overall Result: N/A")
        self.recent_predictions.clear()
        self.overall_result = None

    def monitor_attention(self):
        while self.is_monitoring:
            ret, frame = self.vid.read()
            if ret:
                resized_frame = cv2.resize(frame, (IMG_WIDTH, IMG_HEIGHT))
                result = self.predict_image(resized_frame)
                self.update_result(result)
            time.sleep(CAPTURE_INTERVAL)

    def predict_image(self, frame):
        img_array = np.expand_dims(frame, axis=0) / 255.0
        prediction = model.predict(img_array)
        return "Attentive" if prediction[0][0] > 0.5 else "Inattentive"

    def update_result(self, result):
        self.recent_predictions.append(result)
        
        attentive_count = self.recent_predictions.count("Attentive")
        total_predictions = len(self.recent_predictions)
        
        if total_predictions > 0:
            attentive_ratio = attentive_count / total_predictions
            
            if attentive_ratio >= MAJORITY_THRESHOLD:
                new_result = "Inattentive"
            elif (1 - attentive_ratio) >= MAJORITY_THRESHOLD:
                new_result = "Attentive"
            else:
                new_result = "Mixed"
            
            if new_result != self.overall_result:
                self.overall_result = new_result
                self.label_result.config(text=f"Overall Result: {self.overall_result}")

    def update(self):
        ret, frame = self.vid.read()
        if ret:
            frame = cv2.resize(frame, (IMG_WIDTH, IMG_HEIGHT))
            self.photo = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
        self.window.after(10, self.update)

if __name__ == "__main__":
    root = tk.Tk()
    AttentionMonitor(root)