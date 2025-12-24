"""
Enhanced Speed Estimator with SAHI for Small Object Detection
==============================================================
This module adds SAHI (Slicing Aided Hyper Inference) support for detecting
small vehicles in high-resolution aerial/drone footage.

SAHI works by:
1. Slicing the image into overlapping patches
2. Running detection on each patch
3. Merging results with NMS

This dramatically improves detection of small objects that YOLO misses
when they occupy only a few pixels in a large image.

Author: Mohamed Hendawy
Date: December 2024
"""

import cv2
import numpy as np
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import time

from ultralytics import YOLO
import supervision as sv

# SAHI imports
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction


@dataclass
class VehicleTrack:
    """Stores tracking information for a single vehicle."""
    track_id: int
    positions: List[Tuple[float, float]] = field(default_factory=list)
    timestamps: List[float] = field(default_factory=list)
    speeds: List[float] = field(default_factory=list)
    class_name: str = "vehicle"
    
    def add_position(self, x: float, y: float, timestamp: float):
        self.positions.append((x, y))
        self.timestamps.append(timestamp)
    
    def get_average_speed(self, window: int = 5) -> float:
        if len(self.speeds) == 0:
            return 0.0
        recent_speeds = self.speeds[-window:]
        return sum(recent_speeds) / len(recent_speeds)


class SAHISpeedEstimator:
    """
    Speed estimator with SAHI support for small object detection.
    
    Use this when:
    - Vehicles appear very small in the frame (< 50 pixels)
    - Using high-resolution aerial/drone footage
    - Standard YOLO detection is missing objects
    
    Note: SAHI is slower than direct YOLO inference due to multiple
    patch predictions. For normal traffic camera footage, use the
    standard SpeedEstimator for better performance.
    """
    
    VEHICLE_CLASSES = {
        2: "car",
        3: "motorcycle", 
        5: "bus",
        7: "truck"
    }
    
    # COCO names for SAHI filtering
    VEHICLE_NAMES = ["car", "motorcycle", "bus", "truck"]
    
    def __init__(
        self,
        model_path: str = "yolo11m.pt",
        pixels_per_meter: float = 8.0,
        confidence_threshold: float = 0.25,
        slice_height: int = 640,
        slice_width: int = 640,
        overlap_ratio: float = 0.2,
        speed_smoothing_window: int = 5,
    ):
        """
        Initialize SAHI-enhanced speed estimator.
        
        Args:
            model_path: Path to YOLO model weights
            pixels_per_meter: Calibration value
            confidence_threshold: Detection confidence threshold
            slice_height: Height of each slice for SAHI
            slice_width: Width of each slice for SAHI
            overlap_ratio: Overlap between slices (0.0-0.5)
            speed_smoothing_window: Frames to average speed over
        """
        print(f"Loading YOLO model with SAHI: {model_path}")
        
        # Load model for SAHI
        self.detection_model = AutoDetectionModel.from_pretrained(
            model_type="yolov8",  # SAHI uses yolov8 type for ultralytics models
            model_path=model_path,
            confidence_threshold=confidence_threshold,
            device="cuda:0" if self._cuda_available() else "cpu"
        )
        
        # SAHI parameters
        self.slice_height = slice_height
        self.slice_width = slice_width
        self.overlap_ratio = overlap_ratio
        self.confidence_threshold = confidence_threshold
        
        # Speed estimation parameters
        self.pixels_per_meter = pixels_per_meter
        self.speed_smoothing_window = speed_smoothing_window
        
        # Tracking
        self.tracker = sv.ByteTrack(
            track_activation_threshold=confidence_threshold,
            minimum_matching_threshold=0.8,
            lost_track_buffer=30,
            minimum_consecutive_frames=3
        )
        
        # Track storage
        self.tracks: Dict[int, VehicleTrack] = {}
        
        # Visualization
        self.box_annotator = sv.BoxAnnotator(thickness=2)
        self.label_annotator = sv.LabelAnnotator(text_scale=0.5, text_padding=5)
        self.trace_annotator = sv.TraceAnnotator(
            thickness=2,
            trace_length=30,
            position=sv.Position.BOTTOM_CENTER
        )
    
    def _cuda_available(self) -> bool:
        """Check if CUDA is available."""
        try:
            import torch
            return torch.cuda.is_available()
        except:
            return False
    
    def detect_vehicles_sahi(self, frame: np.ndarray) -> sv.Detections:
        """
        Detect vehicles using SAHI for improved small object detection.
        
        Args:
            frame: BGR image as numpy array
            
        Returns:
            Detections object with bounding boxes, confidences, and class IDs
        """
        # Get sliced prediction
        result = get_sliced_prediction(
            frame,
            self.detection_model,
            slice_height=self.slice_height,
            slice_width=self.slice_width,
            overlap_height_ratio=self.overlap_ratio,
            overlap_width_ratio=self.overlap_ratio,
            verbose=0
        )
        
        # Filter for vehicle classes and convert to supervision format
        boxes = []
        confidences = []
        class_ids = []
        
        for pred in result.object_prediction_list:
            category_name = pred.category.name.lower()
            if category_name in self.VEHICLE_NAMES:
                bbox = pred.bbox
                boxes.append([bbox.minx, bbox.miny, bbox.maxx, bbox.maxy])
                confidences.append(pred.score.value)
                
                # Map category name to COCO class ID
                class_id_map = {"car": 2, "motorcycle": 3, "bus": 5, "truck": 7}
                class_ids.append(class_id_map.get(category_name, 2))
        
        if len(boxes) == 0:
            return sv.Detections.empty()
        
        return sv.Detections(
            xyxy=np.array(boxes),
            confidence=np.array(confidences),
            class_id=np.array(class_ids)
        )
    
    def track_vehicles(self, detections: sv.Detections) -> sv.Detections:
        """Apply ByteTrack to associate detections across frames."""
        return self.tracker.update_with_detections(detections)
    
    def calculate_speed(
        self,
        track_id: int,
        current_pos: Tuple[float, float],
        current_time: float,
        class_id: int
    ) -> float:
        """Calculate speed for a tracked vehicle."""
        if track_id not in self.tracks:
            self.tracks[track_id] = VehicleTrack(
                track_id=track_id,
                class_name=self.VEHICLE_CLASSES.get(class_id, "vehicle")
            )
        
        track = self.tracks[track_id]
        speed_kmh = 0.0
        
        if len(track.positions) > 0:
            prev_pos = track.positions[-1]
            prev_time = track.timestamps[-1]
            
            dx = current_pos[0] - prev_pos[0]
            dy = current_pos[1] - prev_pos[1]
            pixel_displacement = np.sqrt(dx**2 + dy**2)
            dt = current_time - prev_time
            
            if dt > 0:
                meters_displacement = pixel_displacement / self.pixels_per_meter
                speed_mps = meters_displacement / dt
                speed_kmh = speed_mps * 3.6
                speed_kmh = min(speed_kmh, 200.0)
                track.speeds.append(speed_kmh)
        
        track.add_position(current_pos[0], current_pos[1], current_time)
        return track.get_average_speed(self.speed_smoothing_window)
    
    def process_frame(
        self,
        frame: np.ndarray,
        timestamp: float
    ) -> Tuple[np.ndarray, List[Dict]]:
        """Process a single frame with SAHI detection."""
        # Detect vehicles with SAHI
        detections = self.detect_vehicles_sahi(frame)
        
        # Track vehicles
        tracked_detections = self.track_vehicles(detections)
        
        # Calculate speeds and create labels
        labels = []
        vehicle_info = []
        
        if len(tracked_detections) > 0:
            for i in range(len(tracked_detections)):
                bbox = tracked_detections.xyxy[i]
                bottom_center = ((bbox[0] + bbox[2]) / 2, bbox[3])
                
                track_id = tracked_detections.tracker_id[i]
                class_id = tracked_detections.class_id[i]
                confidence = tracked_detections.confidence[i]
                
                speed = self.calculate_speed(
                    track_id, 
                    bottom_center, 
                    timestamp, 
                    class_id
                )
                
                class_name = self.VEHICLE_CLASSES.get(class_id, "vehicle")
                label = f"#{track_id} {class_name} {speed:.1f} km/h"
                labels.append(label)
                
                vehicle_info.append({
                    "track_id": int(track_id),
                    "class": class_name,
                    "speed_kmh": round(speed, 1),
                    "confidence": round(float(confidence), 2),
                    "bbox": bbox.tolist()
                })
        
        # Annotate frame
        annotated_frame = frame.copy()
        annotated_frame = self.trace_annotator.annotate(
            scene=annotated_frame,
            detections=tracked_detections
        )
        annotated_frame = self.box_annotator.annotate(
            scene=annotated_frame,
            detections=tracked_detections
        )
        annotated_frame = self.label_annotator.annotate(
            scene=annotated_frame,
            detections=tracked_detections,
            labels=labels
        )
        
        return annotated_frame, vehicle_info
    
    def process_video(
        self,
        video_path: str,
        output_path: Optional[str] = None,
        display: bool = True,
        max_frames: Optional[int] = None,
        resize_width: Optional[int] = None  # Don't resize for SAHI - we need full resolution
    ) -> Dict:
        """
        Process a video file with SAHI detection.
        
        Note: For SAHI, we don't resize the frame to maintain detection quality
        on small objects. This is slower but more accurate for aerial footage.
        """
        print(f"\nProcessing video with SAHI: {video_path}")
        print(f"Slice size: {self.slice_width}x{self.slice_height}, Overlap: {self.overlap_ratio}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Video info: {width}x{height} @ {fps:.1f} FPS, {total_frames} frames")
        
        # For display/output, we may want to resize
        output_width = resize_width if resize_width else width
        output_height = int(height * (output_width / width)) if resize_width else height
        
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (output_width, output_height))
        
        frame_count = 0
        start_time = time.time()
        all_vehicle_info = []
        
        print("\nStarting SAHI processing... (Press 'q' to quit)")
        print("Note: SAHI is slower than standard YOLO for better small object detection")
        
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                if max_frames and frame_count >= max_frames:
                    break
                
                timestamp = frame_count / fps
                
                # Process at full resolution for SAHI
                annotated_frame, vehicle_info = self.process_frame(frame, timestamp)
                
                # Resize for output if needed
                if resize_width and resize_width != width:
                    annotated_frame = cv2.resize(annotated_frame, (output_width, output_height))
                
                # Draw frame info
                cv2.putText(
                    annotated_frame,
                    f"Frame: {frame_count}/{total_frames} | Vehicles: {len(vehicle_info)} | SAHI Mode",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )
                
                elapsed = time.time() - start_time
                processing_fps = frame_count / elapsed if elapsed > 0 else 0
                cv2.putText(
                    annotated_frame,
                    f"Processing FPS: {processing_fps:.1f}",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )
                
                all_vehicle_info.extend(vehicle_info)
                
                if writer:
                    writer.write(annotated_frame)
                
                if display:
                    # Resize for display
                    display_frame = cv2.resize(annotated_frame, (1280, 720)) if width > 1280 else annotated_frame
                    cv2.imshow("SAHI Vehicle Speed Estimation", display_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("\nUser interrupted processing")
                        break
                
                frame_count += 1
                
                if frame_count % 50 == 0:
                    progress = (frame_count / total_frames) * 100
                    print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames} frames) - {len(self.tracks)} tracks")
        
        finally:
            cap.release()
            if writer:
                writer.release()
            cv2.destroyAllWindows()
        
        total_time = time.time() - start_time
        avg_fps = frame_count / total_time if total_time > 0 else 0
        
        stats = {
            "frames_processed": frame_count,
            "total_time_seconds": round(total_time, 2),
            "average_fps": round(avg_fps, 1),
            "unique_vehicles": len(self.tracks),
            "total_detections": len(all_vehicle_info),
            "mode": "SAHI",
            "slice_size": f"{self.slice_width}x{self.slice_height}"
        }
        
        print(f"\n{'='*50}")
        print("SAHI Processing Complete!")
        print(f"{'='*50}")
        print(f"Frames processed: {stats['frames_processed']}")
        print(f"Processing time: {stats['total_time_seconds']:.1f} seconds")
        print(f"Average FPS: {stats['average_fps']:.1f}")
        print(f"Unique vehicles tracked: {stats['unique_vehicles']}")
        
        if output_path:
            print(f"Output saved to: {output_path}")
        
        return stats


def main():
    """Main function for SAHI-enhanced speed estimation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="SAHI-Enhanced Vehicle Speed Estimation")
    parser.add_argument("--video", "-v", type=str, default="data/videos/traffic_sample.mp4")
    parser.add_argument("--output", "-o", type=str, default=None)
    parser.add_argument("--model", "-m", type=str, default="yolo11m.pt")
    parser.add_argument("--ppm", type=float, default=8.0)
    parser.add_argument("--confidence", "-c", type=float, default=0.25)
    parser.add_argument("--slice-size", type=int, default=640)
    parser.add_argument("--overlap", type=float, default=0.2)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--resize", type=int, default=None)
    
    args = parser.parse_args()
    
    estimator = SAHISpeedEstimator(
        model_path=args.model,
        pixels_per_meter=args.ppm,
        confidence_threshold=args.confidence,
        slice_height=args.slice_size,
        slice_width=args.slice_size,
        overlap_ratio=args.overlap
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
