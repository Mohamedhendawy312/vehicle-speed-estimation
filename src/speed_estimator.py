"""
Vehicle Speed Estimation System
================================
Real-time vehicle detection, tracking, and speed estimation using YOLO11 and OpenCV.

This system uses:
- YOLO11 for vehicle detection
- ByteTrack (via supervision) for multi-object tracking
- Geometric speed estimation based on frame-to-frame displacement

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


@dataclass
class VehicleTrack:
    """Stores tracking information for a single vehicle."""
    track_id: int
    positions: List[Tuple[float, float]] = field(default_factory=list)
    timestamps: List[float] = field(default_factory=list)
    speeds: List[float] = field(default_factory=list)
    class_name: str = "vehicle"
    
    def add_position(self, x: float, y: float, timestamp: float):
        """Add a new position to the track."""
        self.positions.append((x, y))
        self.timestamps.append(timestamp)
    
    def get_average_speed(self, window: int = 5) -> float:
        """Get the average speed over the last `window` measurements."""
        if len(self.speeds) == 0:
            return 0.0
        recent_speeds = self.speeds[-window:]
        return sum(recent_speeds) / len(recent_speeds)


class SpeedEstimator:
    """
    Estimates vehicle speeds from video using detection, tracking, and geometric calculations.
    
    The speed estimation works by:
    1. Detecting vehicles in each frame using YOLO11
    2. Tracking vehicles across frames using ByteTrack
    3. Calculating displacement between frames in pixels
    4. Converting pixel displacement to real-world speed using calibration
    
    Calibration Note:
    -----------------
    For accurate speed estimation, you need to calibrate the pixels_per_meter value
    based on your camera setup. This can be done by:
    - Measuring a known distance in the scene (e.g., lane width ~3.7m)
    - Counting the pixels that distance spans in your video
    - pixels_per_meter = pixel_distance / real_distance_in_meters
    """
    
    # COCO class IDs for vehicles
    VEHICLE_CLASSES = {
        2: "car",
        3: "motorcycle", 
        5: "bus",
        7: "truck"
    }
    
    def __init__(
        self,
        model_path: str = "yolo11n.pt",
        pixels_per_meter: float = 8.0,  # Calibration: pixels per meter (adjust based on camera)
        confidence_threshold: float = 0.3,
        iou_threshold: float = 0.5,
        speed_smoothing_window: int = 5,
    ):
        """
        Initialize the speed estimator.
        
        Args:
            model_path: Path to YOLO model weights
            pixels_per_meter: Calibration value - pixels per real-world meter
            confidence_threshold: Minimum detection confidence
            iou_threshold: IoU threshold for NMS
            speed_smoothing_window: Number of frames to average speed over
        """
        print(f"Loading YOLO model: {model_path}")
        self.model = YOLO(model_path)
        
        # Detection parameters
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        
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
        
    def detect_vehicles(self, frame: np.ndarray) -> sv.Detections:
        """
        Detect vehicles in a frame using YOLO11.
        
        Args:
            frame: BGR image as numpy array
            
        Returns:
            Detections object containing bounding boxes, confidences, and class IDs
        """
        results = self.model(
            frame,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            classes=list(self.VEHICLE_CLASSES.keys()),
            verbose=False
        )[0]
        
        detections = sv.Detections.from_ultralytics(results)
        return detections
    
    def track_vehicles(self, detections: sv.Detections) -> sv.Detections:
        """
        Apply ByteTrack to associate detections across frames.
        
        Args:
            detections: Current frame detections
            
        Returns:
            Detections with track IDs assigned
        """
        return self.tracker.update_with_detections(detections)
    
    def calculate_speed(
        self,
        track_id: int,
        current_pos: Tuple[float, float],
        current_time: float,
        class_id: int
    ) -> float:
        """
        Calculate speed for a tracked vehicle.
        
        Args:
            track_id: Unique track identifier
            current_pos: Current (x, y) position (bottom center of bbox)
            current_time: Current timestamp in seconds
            class_id: COCO class ID
            
        Returns:
            Estimated speed in km/h
        """
        # Create track if new
        if track_id not in self.tracks:
            self.tracks[track_id] = VehicleTrack(
                track_id=track_id,
                class_name=self.VEHICLE_CLASSES.get(class_id, "vehicle")
            )
        
        track = self.tracks[track_id]
        
        # Calculate speed if we have previous position
        speed_kmh = 0.0
        if len(track.positions) > 0:
            prev_pos = track.positions[-1]
            prev_time = track.timestamps[-1]
            
            # Calculate displacement in pixels
            dx = current_pos[0] - prev_pos[0]
            dy = current_pos[1] - prev_pos[1]
            pixel_displacement = np.sqrt(dx**2 + dy**2)
            
            # Time difference
            dt = current_time - prev_time
            
            if dt > 0:
                # Convert to real-world speed
                meters_displacement = pixel_displacement / self.pixels_per_meter
                speed_mps = meters_displacement / dt  # meters per second
                speed_kmh = speed_mps * 3.6  # convert to km/h
                
                # Sanity check: cap unrealistic speeds
                speed_kmh = min(speed_kmh, 200.0)  # Max 200 km/h
                
                track.speeds.append(speed_kmh)
        
        # Update track
        track.add_position(current_pos[0], current_pos[1], current_time)
        
        # Return smoothed speed
        return track.get_average_speed(self.speed_smoothing_window)
    
    def process_frame(
        self,
        frame: np.ndarray,
        timestamp: float
    ) -> Tuple[np.ndarray, List[Dict]]:
        """
        Process a single frame: detect, track, estimate speed, and annotate.
        
        Args:
            frame: BGR image
            timestamp: Frame timestamp in seconds
            
        Returns:
            Tuple of (annotated_frame, list of vehicle info dicts)
        """
        # Detect vehicles
        detections = self.detect_vehicles(frame)
        
        # Track vehicles
        tracked_detections = self.track_vehicles(detections)
        
        # Calculate speeds and create labels
        labels = []
        vehicle_info = []
        
        if len(tracked_detections) > 0:
            for i in range(len(tracked_detections)):
                # Get bounding box bottom center
                bbox = tracked_detections.xyxy[i]
                bottom_center = ((bbox[0] + bbox[2]) / 2, bbox[3])
                
                # Get track ID and class
                track_id = tracked_detections.tracker_id[i]
                class_id = tracked_detections.class_id[i]
                confidence = tracked_detections.confidence[i]
                
                # Calculate speed
                speed = self.calculate_speed(
                    track_id, 
                    bottom_center, 
                    timestamp, 
                    class_id
                )
                
                # Create label
                class_name = self.VEHICLE_CLASSES.get(class_id, "vehicle")
                label = f"#{track_id} {class_name} {speed:.1f} km/h"
                labels.append(label)
                
                # Store vehicle info
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
        resize_width: Optional[int] = 1280
    ) -> Dict:
        """
        Process an entire video file.
        
        Args:
            video_path: Path to input video
            output_path: Path to save output video (optional)
            display: Whether to display video while processing
            max_frames: Maximum frames to process (optional)
            resize_width: Resize frame width for processing (optional)
            
        Returns:
            Dictionary with processing statistics
        """
        print(f"\nProcessing video: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Video info: {width}x{height} @ {fps:.1f} FPS, {total_frames} frames")
        
        # Calculate resize dimensions
        if resize_width and resize_width < width:
            scale = resize_width / width
            new_width = resize_width
            new_height = int(height * scale)
        else:
            new_width, new_height = width, height
            scale = 1.0
        
        # Adjust pixels_per_meter for resize
        effective_ppm = self.pixels_per_meter * scale
        
        # Setup video writer
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (new_width, new_height))
        
        # Processing stats
        frame_count = 0
        start_time = time.time()
        all_vehicle_info = []
        
        print("\nStarting processing... (Press 'q' to quit)")
        
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                if max_frames and frame_count >= max_frames:
                    break
                
                # Resize if needed
                if scale != 1.0:
                    frame = cv2.resize(frame, (new_width, new_height))
                
                # Calculate timestamp
                timestamp = frame_count / fps
                
                # Temporarily adjust pixels_per_meter
                original_ppm = self.pixels_per_meter
                self.pixels_per_meter = effective_ppm
                
                # Process frame
                annotated_frame, vehicle_info = self.process_frame(frame, timestamp)
                
                # Restore original ppm
                self.pixels_per_meter = original_ppm
                
                # Draw frame info
                cv2.putText(
                    annotated_frame,
                    f"Frame: {frame_count}/{total_frames} | Vehicles: {len(vehicle_info)}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )
                
                # Calculate and display FPS
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
                
                # Write to output
                if writer:
                    writer.write(annotated_frame)
                
                # Display
                if display:
                    cv2.imshow("Vehicle Speed Estimation", annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("\nUser interrupted processing")
                        break
                
                frame_count += 1
                
                # Progress update
                if frame_count % 100 == 0:
                    progress = (frame_count / total_frames) * 100
                    print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames} frames)")
        
        finally:
            cap.release()
            if writer:
                writer.release()
            cv2.destroyAllWindows()
        
        # Calculate statistics
        total_time = time.time() - start_time
        avg_fps = frame_count / total_time if total_time > 0 else 0
        
        stats = {
            "frames_processed": frame_count,
            "total_time_seconds": round(total_time, 2),
            "average_fps": round(avg_fps, 1),
            "unique_vehicles": len(self.tracks),
            "total_detections": len(all_vehicle_info),
        }
        
        print(f"\n{'='*50}")
        print("Processing Complete!")
        print(f"{'='*50}")
        print(f"Frames processed: {stats['frames_processed']}")
        print(f"Processing time: {stats['total_time_seconds']:.1f} seconds")
        print(f"Average FPS: {stats['average_fps']:.1f}")
        print(f"Unique vehicles tracked: {stats['unique_vehicles']}")
        
        if output_path:
            print(f"Output saved to: {output_path}")
        
        return stats


def main():
    """Main function to run the speed estimation demo."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Vehicle Speed Estimation System")
    parser.add_argument(
        "--video", "-v",
        type=str,
        default="data/videos/traffic_sample.mp4",
        help="Path to input video"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Path to output video (optional)"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="yolo11n.pt",
        help="YOLO model to use"
    )
    parser.add_argument(
        "--ppm",
        type=float,
        default=8.0,
        help="Pixels per meter calibration value"
    )
    parser.add_argument(
        "--confidence", "-c",
        type=float,
        default=0.3,
        help="Detection confidence threshold"
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum frames to process"
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Disable display window"
    )
    parser.add_argument(
        "--resize",
        type=int,
        default=1280,
        help="Resize width for processing"
    )
    
    args = parser.parse_args()
    
    # Initialize estimator
    estimator = SpeedEstimator(
        model_path=args.model,
        pixels_per_meter=args.ppm,
        confidence_threshold=args.confidence
    )
    
    # Process video
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
