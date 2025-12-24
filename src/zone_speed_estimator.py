"""
Advanced Speed Estimation with Line Crossing
=============================================
This module adds zone-based speed estimation using virtual detection lines.

This approach is more accurate than frame-by-frame displacement because:
1. You define two lines at known real-world distance apart
2. The system measures the time each vehicle takes to travel between lines
3. Speed = known_distance / travel_time

Author: Mohamed Hendawy
Date: December 2024
"""

import cv2
import numpy as np
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import time
import json

from ultralytics import YOLO
import supervision as sv


@dataclass
class LineConfig:
    """Configuration for a detection line."""
    start: Tuple[int, int]
    end: Tuple[int, int]
    name: str = "line"


@dataclass 
class VehicleCrossing:
    """Records when a vehicle crosses detection lines."""
    track_id: int
    class_name: str
    line1_time: Optional[float] = None
    line2_time: Optional[float] = None
    speed_kmh: Optional[float] = None
    
    def calculate_speed(self, distance_meters: float) -> Optional[float]:
        """Calculate speed based on crossing times and known distance."""
        if self.line1_time and self.line2_time:
            time_diff = abs(self.line2_time - self.line1_time)
            if time_diff > 0:
                speed_mps = distance_meters / time_diff
                self.speed_kmh = speed_mps * 3.6
                return self.speed_kmh
        return None


class ZoneSpeedEstimator:
    """
    Speed estimation using two detection lines at known distance.
    
    This is more accurate than frame displacement because it uses
    real-world distance measurements between two reference lines.
    
    Usage:
        1. Define two horizontal lines in the video frame
        2. Measure the real-world distance between the lines
        3. The system tracks when vehicles cross each line
        4. Speed = distance / crossing_time
    """
    
    VEHICLE_CLASSES = {
        2: "car",
        3: "motorcycle", 
        5: "bus",
        7: "truck"
    }
    
    def __init__(
        self,
        model_path: str = "yolo11n.pt",
        line1: Tuple[Tuple[int, int], Tuple[int, int]] = None,
        line2: Tuple[Tuple[int, int], Tuple[int, int]] = None,
        distance_between_lines_meters: float = 10.0,
        confidence_threshold: float = 0.3,
    ):
        """
        Initialize zone-based speed estimator.
        
        Args:
            model_path: Path to YOLO model
            line1: First detection line ((x1,y1), (x2,y2))
            line2: Second detection line ((x1,y1), (x2,y2))
            distance_between_lines_meters: Real-world distance between lines
            confidence_threshold: Detection confidence threshold
        """
        print(f"Loading YOLO model: {model_path}")
        self.model = YOLO(model_path)
        
        self.line1 = line1
        self.line2 = line2
        self.distance_meters = distance_between_lines_meters
        self.confidence_threshold = confidence_threshold
        
        # Tracker
        self.tracker = sv.ByteTrack(
            track_activation_threshold=confidence_threshold,
            minimum_matching_threshold=0.8,
            lost_track_buffer=30,
        )
        
        # Line crossing detectors (will be set up when lines are defined)
        self.line_zone1 = None
        self.line_zone2 = None
        
        # Previous positions for crossing detection
        self.prev_positions: Dict[int, Tuple[float, float]] = {}
        
        # Crossing records
        self.crossings: Dict[int, VehicleCrossing] = {}
        
        # Completed speed measurements
        self.completed_measurements: List[VehicleCrossing] = []
        
        # Annotators
        self.box_annotator = sv.BoxAnnotator(thickness=2)
        self.label_annotator = sv.LabelAnnotator(text_scale=0.5, text_padding=5)
        self.trace_annotator = sv.TraceAnnotator(thickness=2, trace_length=50)
    
    def setup_lines(self, frame_width: int, frame_height: int):
        """Setup detection lines based on frame dimensions if not specified."""
        if self.line1 is None:
            # Default: horizontal lines at 40% and 60% of frame height
            y1 = int(frame_height * 0.4)
            y2 = int(frame_height * 0.6)
            self.line1 = ((0, y1), (frame_width, y1))
            self.line2 = ((0, y2), (frame_width, y2))
        
        # Create line zones
        self.line_zone1 = sv.LineZone(
            start=sv.Point(*self.line1[0]),
            end=sv.Point(*self.line1[1])
        )
        self.line_zone2 = sv.LineZone(
            start=sv.Point(*self.line2[0]),
            end=sv.Point(*self.line2[1])
        )
        
        print(f"Detection lines configured:")
        print(f"  Line 1: {self.line1}")
        print(f"  Line 2: {self.line2}")
        print(f"  Distance between lines: {self.distance_meters}m")
    
    def check_line_crossing(
        self,
        track_id: int,
        current_pos: Tuple[float, float],
        timestamp: float,
        class_name: str
    ) -> Optional[str]:
        """
        Check if a vehicle crossed any detection line.
        
        Returns:
            "line1", "line2", or None
        """
        if track_id not in self.prev_positions:
            self.prev_positions[track_id] = current_pos
            return None
        
        prev_pos = self.prev_positions[track_id]
        self.prev_positions[track_id] = current_pos
        
        # Check line 1 crossing
        line1_y = self.line1[0][1]  # Y coordinate of horizontal line 1
        if (prev_pos[1] < line1_y <= current_pos[1]) or (prev_pos[1] > line1_y >= current_pos[1]):
            if track_id not in self.crossings:
                self.crossings[track_id] = VehicleCrossing(
                    track_id=track_id,
                    class_name=class_name
                )
            if self.crossings[track_id].line1_time is None:
                self.crossings[track_id].line1_time = timestamp
                return "line1"
        
        # Check line 2 crossing
        line2_y = self.line2[0][1]
        if (prev_pos[1] < line2_y <= current_pos[1]) or (prev_pos[1] > line2_y >= current_pos[1]):
            if track_id not in self.crossings:
                self.crossings[track_id] = VehicleCrossing(
                    track_id=track_id,
                    class_name=class_name
                )
            if self.crossings[track_id].line2_time is None:
                self.crossings[track_id].line2_time = timestamp
                
                # Calculate speed if both lines crossed
                speed = self.crossings[track_id].calculate_speed(self.distance_meters)
                if speed:
                    self.completed_measurements.append(self.crossings[track_id])
                return "line2"
        
        return None
    
    def process_frame(
        self,
        frame: np.ndarray,
        timestamp: float
    ) -> Tuple[np.ndarray, List[Dict]]:
        """Process a single frame."""
        
        # Setup lines if first frame
        if self.line_zone1 is None:
            self.setup_lines(frame.shape[1], frame.shape[0])
        
        # Detect vehicles
        results = self.model(
            frame,
            conf=self.confidence_threshold,
            classes=list(self.VEHICLE_CLASSES.keys()),
            verbose=False
        )[0]
        
        detections = sv.Detections.from_ultralytics(results)
        tracked = self.tracker.update_with_detections(detections)
        
        # Process detections and check crossings
        labels = []
        vehicle_info = []
        
        if len(tracked) > 0:
            for i in range(len(tracked)):
                bbox = tracked.xyxy[i]
                bottom_center = ((bbox[0] + bbox[2]) / 2, bbox[3])
                track_id = tracked.tracker_id[i]
                class_id = tracked.class_id[i]
                class_name = self.VEHICLE_CLASSES.get(class_id, "vehicle")
                
                # Check line crossings
                crossed = self.check_line_crossing(
                    track_id, bottom_center, timestamp, class_name
                )
                
                # Get speed if available
                speed_str = ""
                if track_id in self.crossings:
                    crossing = self.crossings[track_id]
                    if crossing.speed_kmh:
                        speed_str = f" {crossing.speed_kmh:.1f} km/h"
                
                label = f"#{track_id} {class_name}{speed_str}"
                labels.append(label)
                
                vehicle_info.append({
                    "track_id": int(track_id),
                    "class": class_name,
                    "crossed": crossed,
                    "speed_kmh": self.crossings[track_id].speed_kmh if track_id in self.crossings else None
                })
        
        # Annotate frame
        annotated = frame.copy()
        
        # Draw detection lines
        cv2.line(annotated, self.line1[0], self.line1[1], (0, 255, 0), 3)
        cv2.line(annotated, self.line2[0], self.line2[1], (0, 0, 255), 3)
        cv2.putText(annotated, "LINE 1", (10, self.line1[0][1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(annotated, "LINE 2", (10, self.line2[0][1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Draw tracking
        annotated = self.trace_annotator.annotate(annotated, tracked)
        annotated = self.box_annotator.annotate(annotated, tracked)
        annotated = self.label_annotator.annotate(annotated, tracked, labels)
        
        # Show speed measurements
        y_offset = 90
        for crossing in self.completed_measurements[-5:]:  # Last 5 measurements
            text = f"Vehicle #{crossing.track_id}: {crossing.speed_kmh:.1f} km/h"
            cv2.putText(annotated, text, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            y_offset += 25
        
        return annotated, vehicle_info
    
    def process_video(
        self,
        video_path: str,
        output_path: Optional[str] = None,
        display: bool = True,
        max_frames: Optional[int] = None,
        resize_width: int = 1280
    ) -> Dict:
        """Process a video file."""
        print(f"\nProcessing video: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Resize calculations
        scale = resize_width / width if resize_width < width else 1.0
        new_w = int(width * scale)
        new_h = int(height * scale)
        
        print(f"Video info: {width}x{height} @ {fps:.1f} FPS")
        print(f"Processing at: {new_w}x{new_h}")
        
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (new_w, new_h))
        
        frame_count = 0
        start_time = time.time()
        
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret or (max_frames and frame_count >= max_frames):
                    break
                
                if scale != 1.0:
                    frame = cv2.resize(frame, (new_w, new_h))
                
                timestamp = frame_count / fps
                annotated, info = self.process_frame(frame, timestamp)
                
                # Add info overlay
                cv2.putText(annotated, f"Frame: {frame_count}/{total_frames}",
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(annotated, f"Speed measurements: {len(self.completed_measurements)}",
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                if writer:
                    writer.write(annotated)
                
                if display:
                    cv2.imshow("Zone Speed Estimation", annotated)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                frame_count += 1
                if frame_count % 100 == 0:
                    print(f"Progress: {100*frame_count/total_frames:.1f}%")
        
        finally:
            cap.release()
            if writer:
                writer.release()
            cv2.destroyAllWindows()
        
        total_time = time.time() - start_time
        
        stats = {
            "frames_processed": frame_count,
            "processing_time": round(total_time, 2),
            "fps": round(frame_count / total_time, 1),
            "speed_measurements": len(self.completed_measurements),
            "measurements": [
                {
                    "track_id": c.track_id,
                    "class": c.class_name,
                    "speed_kmh": round(c.speed_kmh, 1) if c.speed_kmh else None
                }
                for c in self.completed_measurements
            ]
        }
        
        print(f"\n{'='*50}")
        print("Processing Complete!")
        print(f"Speed measurements: {len(self.completed_measurements)}")
        for c in self.completed_measurements:
            print(f"  Vehicle #{c.track_id} ({c.class_name}): {c.speed_kmh:.1f} km/h")
        
        return stats


def main():
    """Demo with zone-based speed estimation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Zone-based Speed Estimation")
    parser.add_argument("--video", "-v", type=str, default="data/videos/traffic_sample.mp4")
    parser.add_argument("--output", "-o", type=str, default=None)
    parser.add_argument("--distance", "-d", type=float, default=10.0,
                       help="Distance between detection lines in meters")
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--max-frames", type=int, default=None)
    
    args = parser.parse_args()
    
    estimator = ZoneSpeedEstimator(
        distance_between_lines_meters=args.distance
    )
    
    stats = estimator.process_video(
        video_path=args.video,
        output_path=args.output,
        display=not args.no_display,
        max_frames=args.max_frames
    )
    
    # Save results
    if args.output:
        results_path = args.output.replace('.mp4', '_results.json')
        with open(results_path, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"Results saved to: {results_path}")


if __name__ == "__main__":
    main()
