import cv2
import numpy as np
from ultralytics import YOLO
import torch
from collections import defaultdict
import matplotlib.pyplot as plt
from IPython.display import Video, display, HTML

import os

class VehicleCounter:
    def __init__(self, model_path='yolov8n.pt'):
        """
        Initialize Vehicle Counter with YOLOv8 model
        """
        print(f"Loading YOLO model: {model_path}")
        self.model = YOLO(model_path)
        
        # Vehicle class IDs in COCO dataset
        self.vehicle_classes = [2, 3, 5, 7]  # car, motorcycle, bus, truck
        self.class_names = {2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}
        
        # Tracking variables
        self.track_history = defaultdict(lambda: [])
        self.vehicle_counts = {'in': 0, 'out': 0}
        self.counted_ids = set()
        
        # Line positions for counting
        self.counting_line_in = None
        self.counting_line_out = None
        
        print("Vehicle Counter initialized successfully!")
        
    def setup_counting_lines(self, frame_height, frame_width):
        """
        Setup counting lines based on video dimensions
        """
        self.counting_line_in = int(frame_height * 0.4)
        self.counting_line_out = int(frame_height * 0.6)
        print(f"Counting lines set at: IN={self.counting_line_in}, OUT={self.counting_line_out}")
        
    def draw_counting_lines(self, frame):
        """
        Draw counting lines on frame
        """
        height, width = frame.shape[:2]
        
        # Draw incoming line (red)
        cv2.line(frame, (0, self.counting_line_in), (width, self.counting_line_in), 
                (0, 0, 255), 3)
        cv2.putText(frame, "IN LINE", (10, self.counting_line_in - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Draw outgoing line (blue)
        cv2.line(frame, (0, self.counting_line_out), (width, self.counting_line_out), 
                (255, 0, 0), 3)
        cv2.putText(frame, "OUT LINE", (10, self.counting_line_out + 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    
    def check_line_crossing(self, track_id, current_center):
        """
        Check if vehicle crossed counting lines
        """
        if track_id in self.track_history:
            track = self.track_history[track_id]
            if len(track) > 1:
                prev_y = track[-2][1]
                curr_y = current_center[1]
                
                # Check for incoming vehicles
                if (prev_y < self.counting_line_in and curr_y >= self.counting_line_in and 
                    f"{track_id}_in" not in self.counted_ids):
                    self.vehicle_counts['in'] += 1
                    self.counted_ids.add(f"{track_id}_in")
                    return 'in'
                
                # Check for outgoing vehicles
                elif (prev_y > self.counting_line_out and curr_y <= self.counting_line_out and 
                      f"{track_id}_out" not in self.counted_ids):
                    self.vehicle_counts['out'] += 1
                    self.counted_ids.add(f"{track_id}_out")
                    return 'out'
        
        return None
    
    def draw_counts(self, frame):
        """
        Draw vehicle counts on frame
        """
        # Background rectangle for counts
        cv2.rectangle(frame, (10, 10), (300, 120), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (300, 120), (255, 255, 255), 3)
        
        # Draw counts with larger font
        cv2.putText(frame, f"Vehicles IN: {self.vehicle_counts['in']}", 
                   (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"Vehicles OUT: {self.vehicle_counts['out']}", 
                   (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(frame, f"Total: {self.vehicle_counts['in'] + self.vehicle_counts['out']}", 
                   (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    def process_video(self, input_path, output_path='output_video.mp4'):
        """
        Process video for vehicle detection and counting
        """
        print(f"Processing video: {input_path}")
        
        # Open video
        cap = cv2.VideoCapture(input_path)
        
        if not cap.isOpened():
            print("Error: Could not open video file")
            return False
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video properties: {width}x{height}, {fps} FPS, {total_frames} frames")
        
        # Setup counting lines
        self.setup_counting_lines(height, width)
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        
        print("Starting video processing...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Run YOLO inference with tracking
            results = self.model.track(frame, persist=True, verbose=False)
            
            # Process detections
            if results[0].boxes is not None and results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                track_ids = results[0].boxes.id.cpu().numpy().astype(int)
                classes = results[0].boxes.cls.cpu().numpy().astype(int)
                confidences = results[0].boxes.conf.cpu().numpy()
                
                for box, track_id, cls, conf in zip(boxes, track_ids, classes, confidences):
                    # Filter for vehicle classes only
                    if cls in self.vehicle_classes and conf > 0.5:
                        x1, y1, x2, y2 = box.astype(int)
                        
                        # Calculate center point
                        center_x = int((x1 + x2) / 2)
                        center_y = int((y1 + y2) / 2)
                        center = (center_x, center_y)
                        
                        # Update tracking history
                        self.track_history[track_id].append(center)
                        
                        # Keep only last 30 points for each track
                        if len(self.track_history[track_id]) > 30:
                            self.track_history[track_id] = self.track_history[track_id][-30:]
                        
                        # Check for line crossing
                        direction = self.check_line_crossing(track_id, center)
                        
                        # Draw bounding box
                        color = (0, 255, 0)  # Green for normal
                        thickness = 2
                        if direction == 'in':
                            color = (0, 0, 255)  # Red for incoming
                            thickness = 4
                        elif direction == 'out':
                            color = (255, 0, 0)  # Blue for outgoing
                            thickness = 4
                        
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
                        
                        # Draw label
                        label = f"{self.class_names.get(cls, 'vehicle')} ID:{track_id}"
                        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                        cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                                    (x1 + label_size[0], y1), color, -1)
                        cv2.putText(frame, label, (x1, y1 - 10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        
                        # Draw tracking trail
                        points = self.track_history[track_id]
                        if len(points) > 1:
                            for i in range(1, len(points)):
                                thickness = max(1, int(2 * (i / len(points))))
                                cv2.line(frame, points[i-1], points[i], (0, 255, 255), thickness)
            
            # Draw counting lines
            self.draw_counting_lines(frame)
            
            # Draw counts
            self.draw_counts(frame)
            
            # Write frame to output video
            out.write(frame)
            
            # Print progress every 30 frames
            if frame_count % 30 == 0 or frame_count == total_frames:
                progress = (frame_count / total_frames) * 100
                print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames}) - IN: {self.vehicle_counts['in']}, OUT: {self.vehicle_counts['out']}")
        
        # Cleanup
        cap.release()
        out.release()
        
        print(f"\n🎉 Processing completed!")
        print(f"📊 Final Results:")
        print(f"   • Vehicles IN: {self.vehicle_counts['in']}")
        print(f"   • Vehicles OUT: {self.vehicle_counts['out']}")
        print(f"   • Total vehicles: {self.vehicle_counts['in'] + self.vehicle_counts['out']}")
        print(f"   • Output saved: {output_path}")
        
        return True