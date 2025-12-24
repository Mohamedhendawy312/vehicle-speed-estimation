"""
Line-Crossing Speed Estimation System
======================================
Most accurate method for vehicle speed estimation - no calibration guessing!

How it works:
1. Define two detection lines at a KNOWN real-world distance
2. Track when each vehicle crosses each line
3. Speed = distance / time

This is exactly how police speed cameras and traffic monitoring systems work!

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
class LineCrossingRecord:
    """Records crossing times for a vehicle."""
    track_id: int
    class_name: str
    line1_time: Optional[float] = None
    line2_time: Optional[float] = None
    
    def has_both_crossings(self) -> bool:
        return self.line1_time is not None and self.line2_time is not None
    
    def get_travel_time(self) -> Optional[float]:
        if self.has_both_crossings():
            return abs(self.line2_time - self.line1_time)
        return None
    
    def calculate_speed(self, distance_meters: float) -> Optional[float]:
        """Calculate speed given the known distance between lines."""
        travel_time = self.get_travel_time()
        if travel_time and travel_time > 0:
            speed_mps = distance_meters / travel_time
            return speed_mps * 3.6  # Convert to km/h
        return None


class LineCrossingSpeedEstimator:
    """
    Speed estimation using the line-crossing method.
    
    Supports two orientations:
    - "horizontal": Lines are horizontal, traffic moves top-to-bottom (or bottom-to-top)
    - "vertical": Lines are vertical, traffic moves left-to-right (or right-to-left)
    """
    
    VEHICLE_CLASSES = {
        2: "car",
        3: "motorcycle", 
        5: "bus",
        7: "truck"
    }
    
    def __init__(
        self,
        model_path: str = "yolo11m.pt",
        line1_percent: float = 0.30,
        line2_percent: float = 0.70,
        distance_meters: float = 10.0,
        confidence_threshold: float = 0.4,
        orientation: str = "vertical",  # "vertical" for L-R traffic, "horizontal" for T-B
    ):
        print(f"Loading YOLO model: {model_path}")
        self.model = YOLO(model_path)
        
        self.line1_percent = line1_percent
        self.line2_percent = line2_percent
        self.distance_meters = distance_meters
        self.confidence_threshold = confidence_threshold
        self.orientation = orientation
        
        # Line positions (set on first frame)
        self.line1_pos = None
        self.line2_pos = None
        self.frame_width = None
        self.frame_height = None
        
        # Tracking
        self.tracker = sv.ByteTrack(
            track_activation_threshold=confidence_threshold,
            minimum_matching_threshold=0.8,
            lost_track_buffer=30,
            minimum_consecutive_frames=3
        )
        
        # Previous positions
        self.prev_positions: Dict[int, float] = {}
        
        # Records
        self.records: Dict[int, LineCrossingRecord] = {}
        self.completed_speeds: List[Dict] = []
        
        # Annotators
        self.box_annotator = sv.BoxAnnotator(thickness=2)
        self.label_annotator = sv.LabelAnnotator(text_scale=0.5, text_padding=5)
        self.trace_annotator = sv.TraceAnnotator(thickness=2, trace_length=50)
    
    def setup_lines(self, frame_height: int, frame_width: int):
        """Setup detection line positions based on frame size and orientation."""
        self.frame_height = frame_height
        self.frame_width = frame_width
        
        if self.orientation == "vertical":
            # Vertical lines for left-to-right traffic
            self.line1_pos = int(frame_width * self.line1_percent)
            self.line2_pos = int(frame_width * self.line2_percent)
            print(f"\nLine-Crossing Setup (VERTICAL lines - L↔R traffic):")
            print(f"  Line 1 (GREEN): x = {self.line1_pos} ({self.line1_percent*100:.0f}% from left)")
            print(f"  Line 2 (RED):   x = {self.line2_pos} ({self.line2_percent*100:.0f}% from left)")
        else:
            # Horizontal lines for top-to-bottom traffic
            self.line1_pos = int(frame_height * self.line1_percent)
            self.line2_pos = int(frame_height * self.line2_percent)
            print(f"\nLine-Crossing Setup (HORIZONTAL lines - T↔B traffic):")
            print(f"  Line 1 (GREEN): y = {self.line1_pos} ({self.line1_percent*100:.0f}% from top)")
            print(f"  Line 2 (RED):   y = {self.line2_pos} ({self.line2_percent*100:.0f}% from top)")
        
        print(f"  Known distance between lines: {self.distance_meters} meters")
    
    def get_crossing_coordinate(self, bbox: np.ndarray) -> float:
        """Get the coordinate to check for line crossing based on orientation."""
        center_x = (bbox[0] + bbox[2]) / 2
        center_y = (bbox[1] + bbox[3]) / 2
        
        if self.orientation == "vertical":
            return center_x  # Check X coordinate for vertical lines
        else:
            return bbox[3]  # Check bottom Y for horizontal lines
    
    def check_line_crossing(
        self,
        track_id: int,
        current_pos: float,
        timestamp: float,
        class_name: str
    ) -> Optional[str]:
        """Check if a vehicle crossed either detection line."""
        
        if track_id not in self.records:
            self.records[track_id] = LineCrossingRecord(
                track_id=track_id,
                class_name=class_name
            )
        
        record = self.records[track_id]
        prev_pos = self.prev_positions.get(track_id)
        self.prev_positions[track_id] = current_pos
        
        if prev_pos is None:
            return None
        
        crossed = None
        
        # Check LINE 1 crossing (both directions)
        if record.line1_time is None:
            if (prev_pos < self.line1_pos <= current_pos) or (prev_pos > self.line1_pos >= current_pos):
                record.line1_time = timestamp
                crossed = "line1"
        
        # Check LINE 2 crossing (both directions)
        if record.line2_time is None:
            if (prev_pos < self.line2_pos <= current_pos) or (prev_pos > self.line2_pos >= current_pos):
                record.line2_time = timestamp
                crossed = "line2"
        
        # Calculate speed when both lines crossed
        if record.has_both_crossings() and track_id not in [s['track_id'] for s in self.completed_speeds]:
            speed = record.calculate_speed(self.distance_meters)
            if speed and 5 < speed < 300:  # Sanity check: 5-300 km/h
                measurement = {
                    'track_id': int(track_id),
                    'class': class_name,
                    'speed_kmh': round(float(speed), 1),
                    'travel_time_sec': round(float(record.get_travel_time()), 3),
                }
                self.completed_speeds.append(measurement)
                print(f"  ✓ Vehicle #{track_id} ({class_name}): {speed:.1f} km/h "
                      f"(crossed in {record.get_travel_time():.2f}s)")
        
        return crossed
    
    def draw_lines(self, frame: np.ndarray):
        """Draw detection lines on the frame."""
        if self.orientation == "vertical":
            # Vertical lines
            cv2.line(frame, (self.line1_pos, 0), (self.line1_pos, self.frame_height), 
                     (0, 255, 0), 3)
            cv2.line(frame, (self.line2_pos, 0), (self.line2_pos, self.frame_height), 
                     (0, 0, 255), 3)
            cv2.putText(frame, "LINE 1", (self.line1_pos + 5, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"LINE 2 ({self.distance_meters}m)", 
                        (self.line2_pos + 5, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            # Horizontal lines
            cv2.line(frame, (0, self.line1_pos), (self.frame_width, self.line1_pos), 
                     (0, 255, 0), 3)
            cv2.line(frame, (0, self.line2_pos), (self.frame_width, self.line2_pos), 
                     (0, 0, 255), 3)
            cv2.putText(frame, "LINE 1", (10, self.line1_pos - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"LINE 2 ({self.distance_meters}m)", 
                        (10, self.line2_pos - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    def process_frame(self, frame: np.ndarray, timestamp: float) -> Tuple[np.ndarray, List[Dict]]:
        """Process a single frame."""
        
        if self.line1_pos is None:
            self.setup_lines(frame.shape[0], frame.shape[1])
        
        # Detect and track
        results = self.model(
            frame,
            conf=self.confidence_threshold,
            classes=list(self.VEHICLE_CLASSES.keys()),
            verbose=False
        )[0]
        
        detections = sv.Detections.from_ultralytics(results)
        tracked = self.tracker.update_with_detections(detections)
        
        labels = []
        frame_info = []
        
        if len(tracked) > 0:
            for i in range(len(tracked)):
                bbox = tracked.xyxy[i]
                crossing_coord = self.get_crossing_coordinate(bbox)
                
                track_id = tracked.tracker_id[i]
                class_id = tracked.class_id[i]
                class_name = self.VEHICLE_CLASSES.get(class_id, "vehicle")
                
                crossed = self.check_line_crossing(track_id, crossing_coord, timestamp, class_name)
                
                # Create label
                record = self.records.get(track_id)
                if record and record.has_both_crossings():
                    speed = record.calculate_speed(self.distance_meters)
                    if speed:
                        label = f"#{track_id} {speed:.0f} km/h"
                    else:
                        label = f"#{track_id} {class_name}"
                elif record and record.line1_time is not None:
                    label = f"#{track_id} timing..."
                else:
                    label = f"#{track_id} {class_name}"
                
                labels.append(label)
                frame_info.append({'track_id': int(track_id), 'class': class_name, 'crossed': crossed})
        
        # Annotate
        annotated = frame.copy()
        self.draw_lines(annotated)
        annotated = self.trace_annotator.annotate(annotated, tracked)
        annotated = self.box_annotator.annotate(annotated, tracked)
        annotated = self.label_annotator.annotate(annotated, tracked, labels)
        
        # Speed panel
        self._draw_speed_panel(annotated)
        
        return annotated, frame_info
    
    def _draw_speed_panel(self, frame: np.ndarray):
        """Draw speed measurements panel."""
        panel_height = min(200, 30 + len(self.completed_speeds) * 25)
        cv2.rectangle(frame, (10, 60), (350, 60 + panel_height), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 60), (350, 60 + panel_height), (255, 255, 255), 1)
        
        cv2.putText(frame, f"SPEEDS ({len(self.completed_speeds)})", 
                    (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        y_offset = 110
        for m in self.completed_speeds[-5:]:
            text = f"#{m['track_id']} {m['class']}: {m['speed_kmh']} km/h"
            cv2.putText(frame, text, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 22
    
    def process_video(
        self,
        video_path: str,
        output_path: Optional[str] = None,
        display: bool = True,
        max_frames: Optional[int] = None,
        resize_width: int = 1280
    ) -> Dict:
        """Process a video file."""
        print(f"\n{'='*60}")
        print(f"LINE-CROSSING SPEED ESTIMATION ({self.orientation.upper()} lines)")
        print(f"{'='*60}")
        print(f"Video: {video_path}")
        print(f"Distance between lines: {self.distance_meters} meters")
        
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if resize_width and resize_width < width:
            scale = resize_width / width
            new_w, new_h = resize_width, int(height * scale)
        else:
            new_w, new_h = width, height
            scale = 1.0
        
        print(f"Video: {width}x{height} @ {fps:.1f} FPS → {new_w}x{new_h}")
        
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
                
                cv2.putText(annotated, f"Frame: {frame_count}/{total_frames}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                if writer:
                    writer.write(annotated)
                
                if display:
                    cv2.imshow("Line-Crossing Speed", annotated)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                frame_count += 1
                if frame_count % 100 == 0:
                    print(f"Progress: {100*frame_count/total_frames:.1f}% | Speeds: {len(self.completed_speeds)}")
        
        finally:
            cap.release()
            if writer:
                writer.release()
            cv2.destroyAllWindows()
        
        total_time = time.time() - start_time
        
        if self.completed_speeds:
            speeds = [s['speed_kmh'] for s in self.completed_speeds]
            avg_speed = sum(speeds) / len(speeds)
        else:
            avg_speed = 0
        
        stats = {
            "frames_processed": frame_count,
            "processing_fps": round(frame_count / total_time if total_time > 0 else 0, 1),
            "vehicles_measured": len(self.completed_speeds),
            "average_speed_kmh": round(avg_speed, 1),
            "distance_meters": self.distance_meters,
            "orientation": self.orientation,
            "measurements": self.completed_speeds
        }
        
        print(f"\n{'='*60}")
        print(f"RESULTS: {len(self.completed_speeds)} vehicles measured")
        print(f"Average speed: {avg_speed:.1f} km/h")
        print(f"{'='*60}")
        
        if output_path:
            print(f"Output: {output_path}")
            with open(output_path.replace('.mp4', '_results.json'), 'w') as f:
                json.dump(stats, f, indent=2)
        
        return stats


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Line-Crossing Speed Estimation")
    parser.add_argument("--video", "-v", type=str, default="data/videos/highway_traffic.mp4")
    parser.add_argument("--output", "-o", type=str, default=None)
    parser.add_argument("--model", "-m", type=str, default="yolo11m.pt")
    parser.add_argument("--distance", "-d", type=float, default=10.0)
    parser.add_argument("--line1", type=float, default=0.30)
    parser.add_argument("--line2", type=float, default=0.70)
    parser.add_argument("--orientation", type=str, default="vertical",
                       choices=["vertical", "horizontal"])
    parser.add_argument("--confidence", "-c", type=float, default=0.4)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--resize", type=int, default=1280)
    
    args = parser.parse_args()
    
    estimator = LineCrossingSpeedEstimator(
        model_path=args.model,
        line1_percent=args.line1,
        line2_percent=args.line2,
        distance_meters=args.distance,
        confidence_threshold=args.confidence,
        orientation=args.orientation
    )
    
    stats = estimator.process_video(
        video_path=args.video,
        output_path=args.output,
        display=not args.no_display,
        max_frames=args.max_frames,
        resize_width=args.resize
    )
    
    return stats


if __name__ == "__main__":
    main()
