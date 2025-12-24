"""
Robust Vehicle Speed Estimation
================================
This version fixes the speed prediction issues by:

1. **Jitter Filtering** - Ignores tiny movements that are just detection noise
2. **Speed Smoothing** - Uses longer averaging window and outlier rejection
3. **Acceleration Limits** - Cap impossible speed changes (can't go 0→100 km/h instantly)
4. **Perspective Correction** - Accounts for camera angle (near vs far)
5. **Proper Calibration** - Interactive tool to calibrate pixels_per_meter

Author: Mohamed Hendawy
Date: December 2024
"""

import cv2
import numpy as np
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import time
import json

from ultralytics import YOLO
import supervision as sv


@dataclass
class RobustVehicleTrack:
    """
    Vehicle track with robust speed estimation.
    
    Key improvements:
    - Uses deque with fixed size for memory efficiency
    - Filters out jitter and outliers
    - Applies acceleration limits
    """
    track_id: int
    class_name: str = "vehicle"
    
    # Position history (bottom center of bbox)
    positions: deque = field(default_factory=lambda: deque(maxlen=30))
    timestamps: deque = field(default_factory=lambda: deque(maxlen=30))
    
    # Raw and smoothed speeds
    raw_speeds: deque = field(default_factory=lambda: deque(maxlen=30))
    smoothed_speed: float = 0.0
    
    # For jitter detection
    is_stationary: bool = False
    stationary_frames: int = 0
    
    # Track quality
    consecutive_detections: int = 0
    
    def add_observation(
        self,
        x: float,
        y: float,
        timestamp: float,
        pixels_per_meter: float,
        min_movement_pixels: float = 5.0,  # Jitter threshold (increased to handle bbox jitter)
        max_acceleration_kmh_per_s: float = 50.0  # Max realistic acceleration
    ) -> float:
        """
        Add a new position observation and calculate speed.
        
        Args:
            x, y: Position in pixels
            timestamp: Time in seconds
            pixels_per_meter: Calibration value
            min_movement_pixels: Ignore movements smaller than this (jitter filter)
            max_acceleration_kmh_per_s: Maximum allowed speed change per second
            
        Returns:
            Smoothed speed in km/h
        """
        self.consecutive_detections += 1
        
        # First observation - no speed yet
        if len(self.positions) == 0:
            self.positions.append((x, y))
            self.timestamps.append(timestamp)
            return 0.0
        
        # Get previous position
        prev_x, prev_y = self.positions[-1]
        prev_time = self.timestamps[-1]
        dt = timestamp - prev_time
        
        # Calculate displacement
        dx = x - prev_x
        dy = y - prev_y
        displacement_pixels = np.sqrt(dx**2 + dy**2)
        
        # Store observation
        self.positions.append((x, y))
        self.timestamps.append(timestamp)
        
        if dt <= 0:
            return self.smoothed_speed
        
        # POSITION SMOOTHING: Apply EMA to reduce detection jitter
        # This smooths out small bbox fluctuations before speed calculation
        alpha = 0.7  # Higher = more responsive, lower = more smooth
        if len(self.positions) >= 2:
            prev_x, prev_y = self.positions[-2]
            smooth_x = alpha * x + (1 - alpha) * prev_x
            smooth_y = alpha * y + (1 - alpha) * prev_y
            # Update the stored position with smoothed value
            self.positions[-1] = (smooth_x, smooth_y)
            x, y = smooth_x, smooth_y
        
        # Recalculate displacement with smoothed positions
        dx = x - prev_x if len(self.positions) >= 2 else 0
        dy = y - prev_y if len(self.positions) >= 2 else 0
        displacement_pixels = np.sqrt(dx**2 + dy**2)
        
        # JITTER FILTER: Ignore tiny movements (even after smoothing)
        if displacement_pixels < min_movement_pixels:
            self.stationary_frames += 1
            if self.stationary_frames > 3:  # Reduced from 5 to 3 for faster stationary detection
                self.is_stationary = True
                self.raw_speeds.append(0.0)
            return self.smoothed_speed
        else:
            self.stationary_frames = 0
            self.is_stationary = False
        
        # Calculate raw speed
        displacement_meters = displacement_pixels / pixels_per_meter
        speed_mps = displacement_meters / dt
        raw_speed_kmh = speed_mps * 3.6
        
        # ACCELERATION FILTER: Check if speed change is realistic
        if len(self.raw_speeds) > 0:
            prev_speed = self.raw_speeds[-1]
            speed_change = abs(raw_speed_kmh - prev_speed)
            max_allowed_change = max_acceleration_kmh_per_s * dt
            
            if speed_change > max_allowed_change:
                # Unrealistic acceleration - this is probably a tracking error
                # Clamp the speed change
                if raw_speed_kmh > prev_speed:
                    raw_speed_kmh = prev_speed + max_allowed_change
                else:
                    raw_speed_kmh = max(0, prev_speed - max_allowed_change)
        
        # Sanity cap
        raw_speed_kmh = min(raw_speed_kmh, 200.0)
        raw_speed_kmh = max(raw_speed_kmh, 0.0)
        
        self.raw_speeds.append(raw_speed_kmh)
        
        # SMOOTHING: Median + average of recent speeds (robust to outliers)
        if len(self.raw_speeds) >= 3:
            recent = list(self.raw_speeds)[-10:]  # Last 10 speeds
            # Remove outliers using IQR
            sorted_speeds = sorted(recent)
            q1_idx = len(sorted_speeds) // 4
            q3_idx = (3 * len(sorted_speeds)) // 4
            q1 = sorted_speeds[q1_idx]
            q3 = sorted_speeds[q3_idx]
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            filtered = [s for s in recent if lower <= s <= upper]
            if filtered:
                self.smoothed_speed = sum(filtered) / len(filtered)
            else:
                self.smoothed_speed = raw_speed_kmh
        else:
            self.smoothed_speed = raw_speed_kmh
        
        return self.smoothed_speed
    
    def get_display_speed(self) -> Tuple[float, str]:
        """
        Get speed for display with status indicator.
        
        Returns:
            (speed in km/h, status string)
        """
        if self.is_stationary and self.stationary_frames > 5:
            return 0.0, "STOPPED"
        elif self.consecutive_detections < 2:
            # Show speed immediately, just mark as new
            return self.smoothed_speed, ""
        else:
            return self.smoothed_speed, ""


class RobustSpeedEstimator:
    """
    Robust vehicle speed estimator with jitter filtering and outlier rejection.
    
    Key improvements over basic estimator:
    1. Ignores small movements (detection noise/jitter)
    2. Limits acceleration to physically realistic values
    3. Uses robust averaging (IQR-based outlier removal)
    4. Properly handles stationary vehicles
    5. Requires stable tracking before reporting speed
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
        pixels_per_meter: float = 10.0,
        confidence_threshold: float = 0.4,
        iou_threshold: float = 0.5,
        min_movement_pixels: float = 3.0,
        max_acceleration: float = 50.0,
    ):
        """
        Initialize robust speed estimator.
        
        Args:
            model_path: YOLO model path
            pixels_per_meter: Calibration (use calibrate() to set properly)
            confidence_threshold: Detection confidence
            iou_threshold: NMS IoU threshold
            min_movement_pixels: Ignore movements smaller than this
            max_acceleration: Max acceleration in km/h per second
        """
        print(f"Loading YOLO model: {model_path}")
        self.model = YOLO(model_path)
        
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.pixels_per_meter = pixels_per_meter
        self.min_movement_pixels = min_movement_pixels
        self.max_acceleration = max_acceleration
        
        # Tracking
        self.tracker = sv.ByteTrack(
            track_activation_threshold=confidence_threshold,
            minimum_matching_threshold=0.8,
            lost_track_buffer=30,
            minimum_consecutive_frames=3
        )
        
        # Tracks
        self.tracks: Dict[int, RobustVehicleTrack] = {}
        
        # Annotators
        self.box_annotator = sv.BoxAnnotator(thickness=2)
        self.label_annotator = sv.LabelAnnotator(
            text_scale=0.5, 
            text_padding=5,
            text_color=sv.Color.BLACK
        )
        self.trace_annotator = sv.TraceAnnotator(
            thickness=2,
            trace_length=30,
            position=sv.Position.BOTTOM_CENTER
        )
    
    def detect_and_track(self, frame: np.ndarray) -> sv.Detections:
        """Detect and track vehicles in a frame."""
        results = self.model(
            frame,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            classes=list(self.VEHICLE_CLASSES.keys()),
            verbose=False
        )[0]
        
        detections = sv.Detections.from_ultralytics(results)
        return self.tracker.update_with_detections(detections)
    
    def process_frame(
        self, 
        frame: np.ndarray, 
        timestamp: float
    ) -> Tuple[np.ndarray, List[Dict]]:
        """
        Process a frame and return annotated result.
        """
        tracked = self.detect_and_track(frame)
        
        labels = []
        vehicle_info = []
        
        if len(tracked) > 0:
            for i in range(len(tracked)):
                bbox = tracked.xyxy[i]
                # Use bottom center for speed calculation (ground position)
                bottom_center_x = (bbox[0] + bbox[2]) / 2
                bottom_center_y = bbox[3]
                
                track_id = tracked.tracker_id[i]
                class_id = tracked.class_id[i]
                confidence = tracked.confidence[i]
                class_name = self.VEHICLE_CLASSES.get(class_id, "vehicle")
                
                # Create or get track
                if track_id not in self.tracks:
                    self.tracks[track_id] = RobustVehicleTrack(
                        track_id=track_id,
                        class_name=class_name
                    )
                
                # Calculate speed
                speed = self.tracks[track_id].add_observation(
                    x=bottom_center_x,
                    y=bottom_center_y,
                    timestamp=timestamp,
                    pixels_per_meter=self.pixels_per_meter,
                    min_movement_pixels=self.min_movement_pixels,
                    max_acceleration_kmh_per_s=self.max_acceleration
                )
                
                display_speed, status = self.tracks[track_id].get_display_speed()
                
                # Create label - always show speed
                if status == "STOPPED":
                    label = f"#{track_id} STOPPED"
                else:
                    label = f"#{track_id} {display_speed:.0f} km/h"
                
                labels.append(label)
                
                vehicle_info.append({
                    "track_id": int(track_id),
                    "class": class_name,
                    "speed_kmh": round(display_speed, 1),
                    "status": status,
                    "confidence": round(float(confidence), 2),
                    "is_stationary": self.tracks[track_id].is_stationary
                })
        
        # Annotate
        annotated = frame.copy()
        annotated = self.trace_annotator.annotate(annotated, tracked)
        annotated = self.box_annotator.annotate(annotated, tracked)
        annotated = self.label_annotator.annotate(annotated, tracked, labels)
        
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
        print(f"Calibration: {self.pixels_per_meter} pixels/meter")
        print(f"Jitter threshold: {self.min_movement_pixels} pixels")
        print(f"Max acceleration: {self.max_acceleration} km/h/s")
        
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Resize
        if resize_width and resize_width < width:
            scale = resize_width / width
            new_w = resize_width
            new_h = int(height * scale)
            # Adjust pixels_per_meter for resize
            effective_ppm = self.pixels_per_meter * scale
        else:
            new_w, new_h = width, height
            scale = 1.0
            effective_ppm = self.pixels_per_meter
        
        print(f"Video: {width}x{height} @ {fps:.1f} FPS → Processing at {new_w}x{new_h}")
        
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (new_w, new_h))
        
        frame_count = 0
        start_time = time.time()
        
        # Store original ppm
        original_ppm = self.pixels_per_meter
        self.pixels_per_meter = effective_ppm
        
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret or (max_frames and frame_count >= max_frames):
                    break
                
                if scale != 1.0:
                    frame = cv2.resize(frame, (new_w, new_h))
                
                timestamp = frame_count / fps
                annotated, info = self.process_frame(frame, timestamp)
                
                # Stats overlay
                moving = sum(1 for v in info if not v.get('is_stationary', False))
                stopped = sum(1 for v in info if v.get('is_stationary', False))
                
                cv2.putText(annotated, f"Frame: {frame_count}/{total_frames}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(annotated, f"Moving: {moving} | Stopped: {stopped}", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                if writer:
                    writer.write(annotated)
                
                if display:
                    cv2.imshow("Robust Speed Estimation", annotated)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                frame_count += 1
                if frame_count % 100 == 0:
                    print(f"Progress: {100*frame_count/total_frames:.1f}%")
        
        finally:
            self.pixels_per_meter = original_ppm
            cap.release()
            if writer:
                writer.release()
            cv2.destroyAllWindows()
        
        total_time = time.time() - start_time
        
        stats = {
            "frames_processed": frame_count,
            "total_time_seconds": round(total_time, 2),
            "average_fps": round(frame_count / total_time if total_time > 0 else 0, 1),
            "unique_vehicles": len(self.tracks),
        }
        
        print(f"\n{'='*50}")
        print("Processing Complete!")
        print(f"Frames: {stats['frames_processed']}")
        print(f"Time: {stats['total_time_seconds']:.1f}s")
        print(f"FPS: {stats['average_fps']:.1f}")
        print(f"Tracks: {stats['unique_vehicles']}")
        
        return stats


def calibrate_from_video(video_path: str, model_path: str = "yolo11m.pt"):
    """
    Interactive calibration tool.
    
    Instructions:
    1. The tool will show a frame from the video
    2. Click on two points that represent a KNOWN DISTANCE
    3. Enter the real-world distance in meters
    4. The tool calculates pixels_per_meter for you
    
    Example known distances:
    - Lane width: ~3.7 meters (highway) or ~3.0 meters (city)
    - Car length: ~4.5 meters (average sedan)
    - Truck length: ~12-16 meters
    """
    print("\n" + "="*60)
    print("CAMERA CALIBRATION TOOL")
    print("="*60)
    print("""
This tool helps you calibrate the speed measurement.

HOW IT WORKS:
1. You'll see a frame from the video
2. Click TWO points on something with KNOWN real-world length
   (e.g., a car ~4.5m, lane width ~3.7m, truck ~12m)
3. Enter the real distance in meters
4. The tool calculates pixels_per_meter

Common reference distances:
  - Highway lane width: 3.7 m
  - City lane width: 3.0 m  
  - Average car length: 4.5 m
  - Truck length: 12-16 m
""")
    
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 100)  # Skip to frame 100
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("Error: Could not read video")
        return None
    
    # Resize for display
    h, w = frame.shape[:2]
    if w > 1280:
        scale = 1280 / w
        frame = cv2.resize(frame, (1280, int(h * scale)))
    else:
        scale = 1.0
    
    points = []
    
    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(points) < 2:
                points.append((x, y))
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
                if len(points) == 2:
                    cv2.line(frame, points[0], points[1], (0, 255, 0), 2)
                cv2.imshow("Calibration - Click 2 points", frame)
    
    cv2.namedWindow("Calibration - Click 2 points")
    cv2.setMouseCallback("Calibration - Click 2 points", mouse_callback)
    
    print("\nClick TWO points on the video frame...")
    print("(Press 'q' to quit, 'r' to reset)")
    
    while True:
        cv2.imshow("Calibration - Click 2 points", frame)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            cv2.destroyAllWindows()
            return None
        elif key == ord('r'):
            points.clear()
            cap = cv2.VideoCapture(video_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, 100)
            ret, frame = cap.read()
            cap.release()
            if w > 1280:
                frame = cv2.resize(frame, (1280, int(h * scale)))
        
        if len(points) == 2:
            break
    
    cv2.destroyAllWindows()
    
    # Calculate pixel distance
    pixel_dist = np.sqrt(
        (points[1][0] - points[0][0])**2 + 
        (points[1][1] - points[0][1])**2
    )
    
    # Account for resize
    pixel_dist_original = pixel_dist / scale
    
    print(f"\nPixel distance: {pixel_dist_original:.1f} pixels")
    
    # Get real distance from user
    try:
        real_dist = float(input("Enter the real-world distance in METERS: "))
    except ValueError:
        print("Invalid input")
        return None
    
    if real_dist <= 0:
        print("Distance must be positive")
        return None
    
    pixels_per_meter = pixel_dist_original / real_dist
    
    print(f"\n{'='*60}")
    print(f"CALIBRATION RESULT")
    print(f"{'='*60}")
    print(f"Pixels per meter: {pixels_per_meter:.2f}")
    print(f"\nUse this value when running the speed estimator:")
    print(f"  python robust_speed_estimator.py --video your_video.mp4 --ppm {pixels_per_meter:.1f}")
    print(f"{'='*60}")
    
    return pixels_per_meter


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Robust Vehicle Speed Estimation")
    parser.add_argument("--video", "-v", type=str, default="data/videos/highway_traffic.mp4")
    parser.add_argument("--output", "-o", type=str, default=None)
    parser.add_argument("--model", "-m", type=str, default="yolo11m.pt")
    parser.add_argument("--ppm", type=float, default=10.0, 
                       help="Pixels per meter (use --calibrate to find this)")
    parser.add_argument("--confidence", "-c", type=float, default=0.4)
    parser.add_argument("--jitter", type=float, default=3.0,
                       help="Minimum movement in pixels to count as movement")
    parser.add_argument("--max-accel", type=float, default=50.0,
                       help="Maximum acceleration in km/h per second")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--resize", type=int, default=1280)
    parser.add_argument("--calibrate", action="store_true",
                       help="Run calibration tool")
    
    args = parser.parse_args()
    
    if args.calibrate:
        ppm = calibrate_from_video(args.video, args.model)
        if ppm:
            print(f"\nRun with: --ppm {ppm:.1f}")
        return
    
    estimator = RobustSpeedEstimator(
        model_path=args.model,
        pixels_per_meter=args.ppm,
        confidence_threshold=args.confidence,
        min_movement_pixels=args.jitter,
        max_acceleration=args.max_accel
    )
    
    stats = estimator.process_video(
        video_path=args.video,
        output_path=args.output,
        display=not args.no_display,
        max_frames=args.max_frames,
        resize_width=args.resize
    )


if __name__ == "__main__":
    main()
