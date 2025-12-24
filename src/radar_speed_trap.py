"""
Speed trap using line-crossing method with plate capture.

Works by placing two virtual lines on the road - when a vehicle
crosses both, we calculate speed from the travel time and known distance.

Supports horizontal, vertical, and diagonal line configurations
for different camera angles. See PRESETS for common setups.

Author: Mohamed Hendawy
"""

import cv2
import numpy as np
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import time
import json
import csv
import os
from datetime import datetime

from ultralytics import YOLO
import supervision as sv


@dataclass 
class VehicleMeasurement:
    """Complete measurement record for a vehicle."""
    track_id: int
    vehicle_class: str
    speed_kmh: float
    travel_time_sec: float
    direction: str  # "approaching" or "receding"
    is_violation: bool
    timestamp: str
    frame_number: int
    snapshot_path: Optional[str] = None
    license_plate_path: Optional[str] = None
    license_plate_text: Optional[str] = None


@dataclass
class PlateCandidate:
    """Stores a plate detection candidate for multi-frame best selection."""
    plate_img: np.ndarray
    confidence: float  # LP model detection confidence
    frame_number: int
    bbox: Optional[np.ndarray] = None  # Vehicle bbox at capture time
    frame_snapshot: Optional[np.ndarray] = None  # Full frame at capture time


@dataclass
class CrossingData:
    """Stores precise crossing information for a vehicle."""
    track_id: int
    class_name: str
    
    # Line 1 crossing data
    line1_frame: Optional[int] = None
    line1_timestamp: Optional[float] = None
    line1_position: Optional[float] = None  # Position when crossing (for interpolation)
    line1_prev_position: Optional[float] = None  # Position in previous frame
    
    # Line 2 crossing data
    line2_frame: Optional[int] = None
    line2_timestamp: Optional[float] = None
    line2_position: Optional[float] = None
    line2_prev_position: Optional[float] = None
    
    # Multi-frame plate capture: best plate seen while in zone
    best_plate: Optional[PlateCandidate] = None
    in_zone: bool = False  # Currently between line 1 and line 2
    
    def has_both_crossings(self) -> bool:
        return self.line1_timestamp is not None and self.line2_timestamp is not None
    
    def get_interpolated_crossing_time(self, line_pos: float, prev_pos: float, 
                                        curr_pos: float, frame_time: float, 
                                        fps: float) -> float:
        """
        Calculate precise crossing time using linear interpolation.
        
        Instead of just using frame time, we estimate exactly when
        the vehicle crossed the line based on its position before and after.
        """
        if abs(curr_pos - prev_pos) < 0.001:
            return frame_time
        
        # How far through the frame did the crossing occur?
        fraction = (line_pos - prev_pos) / (curr_pos - prev_pos)
        frame_duration = 1.0 / fps
        
        # Interpolated time
        precise_time = frame_time - frame_duration + (fraction * frame_duration)
        return precise_time
    
    def get_direction(self) -> str:
        """Determine direction based on which line was crossed first."""
        if self.line1_timestamp is None or self.line2_timestamp is None:
            return "unknown"
        return "approaching" if self.line1_timestamp < self.line2_timestamp else "receding"
    
    def calculate_speed(self, distance_meters: float) -> Optional[float]:
        """Calculate speed with precise interpolated timing."""
        if not self.has_both_crossings():
            return None
        travel_time = abs(self.line2_timestamp - self.line1_timestamp)
        if travel_time <= 0:
            return None
        speed_mps = distance_meters / travel_time
        return speed_mps * 3.6  # km/h


class RadarSpeedTrap:
    """Line-crossing speed estimation with plate capture and violation logging."""
    
    VEHICLE_CLASSES = {
        2: "car",
        3: "motorcycle", 
        5: "bus",
        7: "truck"
    }
    
    # Preset line configurations for common camera setups
    # Each preset includes line positions, orientation, and recommended distance
    PRESETS = {
        "highway": {
            # Front-facing camera on highway overpass
            # Vehicles approaching/receding from camera
            "orientation": "horizontal",
            "line1_percent": 0.50,
            "line2_percent": 0.77,
            "distance_meters": 13.0,
            "description": "Front-facing highway camera (overhead or dashcam style)"
        },
        "side_view": {
            # Side-mounted camera, vehicles pass perpendicular to camera
            # Like VS13 CitroenC4Picasso videos
            "orientation": "vertical",
            "line1_percent": 0.30,
            "line2_percent": 0.70,
            "distance_meters": 26.0,
            "description": "Side-mounted camera (perpendicular traffic flow)"
        },
        "diagonal": {
            # Oblique angle camera, road recedes diagonally
            # Like VS13 Peugeot208 videos
            "line1_points": ((0.05, 0.98), (0.35, 0.30)),
            "line2_points": ((0.65, 0.98), (0.75, 0.30)),
            "distance_meters": 26.0,
            "description": "Diagonal/oblique camera angle (road receding at angle)"
        }
    }
    
    @classmethod
    def from_preset(cls, preset_name: str, **kwargs):
        """Create a RadarSpeedTrap with preset line configuration.
        
        Available presets:
        - 'highway': Front-facing camera, horizontal lines
        - 'side_view': Side-mounted camera, vertical lines
        - 'diagonal': Oblique angle camera, diagonal lines
        
        Any additional kwargs override the preset values.
        """
        if preset_name not in cls.PRESETS:
            raise ValueError(f"Unknown preset: {preset_name}. Available: {list(cls.PRESETS.keys())}")
        
        preset = cls.PRESETS[preset_name].copy()
        preset.pop("description", None)  # Remove description, not a constructor arg
        preset.update(kwargs)  # Override with user-provided values
        return cls(**preset)
    
    def __init__(
        self,
        model_path: str = "yolo11m.pt",
        lp_model_path: Optional[str] = None,  # License plate detection model
        line1_percent: float = 0.30,
        line2_percent: float = 0.70,
        line1_points: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None,
        line2_points: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None,
        distance_meters: float = 26.0,
        speed_limit_kmh: float = 80.0,
        confidence_threshold: float = 0.4,
        orientation: str = "vertical",  # "vertical", "horizontal", or "custom"
        output_dir: str = "radar_output",
        capture_plates: bool = True,  # Enable license plate capture
    ):
        """
        Initialize the RadarSpeedTrap.
        
        Line Configuration:
        - Simple: Use orientation + line1_percent/line2_percent
        - Custom/Diagonal: Use line1_points/line2_points as ((x1%, y1%), (x2%, y2%))
          Example diagonal: line1_points=((0.2, 0.8), (0.8, 0.5))
        """
        print(f"Loading YOLO model: {model_path}")
        self.model = YOLO(model_path)
        
        # License plate detection model (optional)
        self.lp_model = None
        self.capture_plates = capture_plates
        if lp_model_path and os.path.exists(lp_model_path):
            print(f"Loading License Plate model: {lp_model_path}")
            self.lp_model = YOLO(lp_model_path)
        elif capture_plates:
            print("No LP model - using vehicle region estimation for plates")
        
        self.line1_percent = line1_percent
        self.line2_percent = line2_percent
        self.line1_points = line1_points  # ((x1%, y1%), (x2%, y2%))
        self.line2_points = line2_points
        self.distance_meters = distance_meters
        self.speed_limit = speed_limit_kmh
        self.confidence_threshold = confidence_threshold
        self.orientation = orientation
        self.output_dir = output_dir
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Line positions (set on first frame) - can be points or single values
        self.line1_pos = None  # For simple lines: int; For custom: ((x1,y1), (x2,y2))
        self.line2_pos = None
        self.line1_coords = None  # Actual pixel coordinates for custom lines
        self.line2_coords = None
        self.frame_width = None
        self.frame_height = None
        self.fps = None
        
        # Tracking
        self.tracker = sv.ByteTrack(
            track_activation_threshold=confidence_threshold,
            minimum_matching_threshold=0.8,
            lost_track_buffer=30,
            minimum_consecutive_frames=3
        )
        
        # Previous positions for interpolation
        self.prev_positions: Dict[int, Tuple[float, float]] = {}  # (x, y) for each track
        
        # Crossing data
        self.crossings: Dict[int, CrossingData] = {}
        
        # Completed measurements
        self.measurements: List[VehicleMeasurement] = []
        
        # Statistics
        self.stats = {
            "total_vehicles": 0,
            "violations": 0,
            "max_speed": 0,
            "total_speed": 0,
        }
        
        # Annotators
        self.box_annotator = sv.BoxAnnotator(thickness=2)
        self.label_annotator = sv.LabelAnnotator(text_scale=0.5, text_padding=5)
        self.trace_annotator = sv.TraceAnnotator(thickness=2, trace_length=50)
    
    def setup_lines(self, frame_height: int, frame_width: int, fps: float):
        """Initialize detection lines - supports horizontal, vertical, and custom diagonal."""
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.fps = fps
        
        # Check for custom point-based lines (diagonal support)
        if self.line1_points is not None and self.line2_points is not None:
            self.orientation = "custom"
            # Convert percentage points to pixel coordinates
            self.line1_coords = (
                (int(self.line1_points[0][0] * frame_width), int(self.line1_points[0][1] * frame_height)),
                (int(self.line1_points[1][0] * frame_width), int(self.line1_points[1][1] * frame_height))
            )
            self.line2_coords = (
                (int(self.line2_points[0][0] * frame_width), int(self.line2_points[0][1] * frame_height)),
                (int(self.line2_points[1][0] * frame_width), int(self.line2_points[1][1] * frame_height))
            )
            print(f"\nRadar Speed Trap Setup (CUSTOM/DIAGONAL lines):")
            print(f"  Line 1: {self.line1_coords[0]} → {self.line1_coords[1]}")
            print(f"  Line 2: {self.line2_coords[0]} → {self.line2_coords[1]}")
        elif self.orientation == "vertical":
            self.line1_pos = int(frame_width * self.line1_percent)
            self.line2_pos = int(frame_width * self.line2_percent)
            # Also create coords for unified drawing
            self.line1_coords = ((self.line1_pos, 0), (self.line1_pos, frame_height))
            self.line2_coords = ((self.line2_pos, 0), (self.line2_pos, frame_height))
            print(f"\nRadar Speed Trap Setup (VERTICAL lines):")
            print(f"  Line 1: x = {self.line1_pos} ({self.line1_percent*100:.0f}%)")
            print(f"  Line 2: x = {self.line2_pos} ({self.line2_percent*100:.0f}%)")
        else:  # horizontal
            self.line1_pos = int(frame_height * self.line1_percent)
            self.line2_pos = int(frame_height * self.line2_percent)
            self.line1_coords = ((0, self.line1_pos), (frame_width, self.line1_pos))
            self.line2_coords = ((0, self.line2_pos), (frame_width, self.line2_pos))
            print(f"\nRadar Speed Trap Setup (HORIZONTAL lines):")
            print(f"  Line 1: y = {self.line1_pos} ({self.line1_percent*100:.0f}%)")
            print(f"  Line 2: y = {self.line2_pos} ({self.line2_percent*100:.0f}%)")
        
        print(f"  Distance: {self.distance_meters}m")
        print(f"  Speed limit: {self.speed_limit} km/h")
    
    def _point_side_of_line(self, point: Tuple[float, float], 
                            line_start: Tuple[int, int], line_end: Tuple[int, int]) -> float:
        """
        Determine which side of a line a point is on.
        Returns positive if on one side, negative on the other, 0 if on the line.
        """
        return ((line_end[0] - line_start[0]) * (point[1] - line_start[1]) - 
                (line_end[1] - line_start[1]) * (point[0] - line_start[0]))
    
    def _crosses_line(self, prev_pos: Tuple[float, float], curr_pos: Tuple[float, float],
                      line_coords: Tuple[Tuple[int, int], Tuple[int, int]]) -> bool:
        """Check if movement from prev_pos to curr_pos crosses the line."""
        side1 = self._point_side_of_line(prev_pos, line_coords[0], line_coords[1])
        side2 = self._point_side_of_line(curr_pos, line_coords[0], line_coords[1])
        return side1 * side2 < 0  # Different signs = crossed
    
    def get_tracking_coordinate(self, bbox: np.ndarray) -> Tuple[float, float]:
        """Get the center point to check for line crossing."""
        center_x = (bbox[0] + bbox[2]) / 2
        center_y = (bbox[1] + bbox[3]) / 2
        return (center_x, center_y)
    
    def check_line_crossing(
        self,
        track_id: int,
        current_pos: Tuple[float, float],  # Now (x, y) tuple
        frame_number: int,
        timestamp: float,
        class_name: str
    ) -> Optional[VehicleMeasurement]:
        """
        Check for line crossings with sub-frame interpolation.
        Returns a measurement if both lines have been crossed.
        Supports horizontal, vertical, and diagonal lines.
        """
        # Initialize crossing data if needed
        if track_id not in self.crossings:
            self.crossings[track_id] = CrossingData(
                track_id=track_id,
                class_name=class_name
            )
        
        crossing = self.crossings[track_id]
        prev_pos = self.prev_positions.get(track_id)
        self.prev_positions[track_id] = current_pos
        
        if prev_pos is None:
            return None
        
        # Use unified crossing detection for all line types
        # Check LINE 1 crossing
        if crossing.line1_timestamp is None:
            if self._crosses_line(prev_pos, current_pos, self.line1_coords):
                crossing.line1_frame = frame_number
                crossing.line1_prev_position = prev_pos[0] if self.orientation == "vertical" else prev_pos[1]
                crossing.line1_position = current_pos[0] if self.orientation == "vertical" else current_pos[1]
                # Use frame-based timing for interpolation
                crossing.line1_timestamp = timestamp
                crossing.in_zone = True  # Vehicle entered detection zone
        
        # Check LINE 2 crossing
        if crossing.line2_timestamp is None:
            if self._crosses_line(prev_pos, current_pos, self.line2_coords):
                crossing.line2_frame = frame_number
                crossing.line2_prev_position = prev_pos[0] if self.orientation == "vertical" else prev_pos[1]
                crossing.line2_position = current_pos[0] if self.orientation == "vertical" else current_pos[1]
                crossing.line2_timestamp = timestamp
        
        # If both lines crossed, create measurement
        if crossing.has_both_crossings() and track_id not in [m.track_id for m in self.measurements]:
            speed = crossing.calculate_speed(self.distance_meters)
            
            if speed and 5 < speed < 300:  # Sanity check
                direction = crossing.get_direction()
                is_violation = speed > self.speed_limit
                travel_time = abs(crossing.line2_timestamp - crossing.line1_timestamp)
                
                measurement = VehicleMeasurement(
                    track_id=track_id,
                    vehicle_class=class_name,
                    speed_kmh=round(speed, 1),
                    travel_time_sec=round(travel_time, 3),
                    direction=direction,
                    is_violation=is_violation,
                    timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    frame_number=frame_number
                )
                
                self.measurements.append(measurement)
                self.stats["total_vehicles"] += 1
                self.stats["total_speed"] += speed
                self.stats["max_speed"] = max(self.stats["max_speed"], speed)
                
                if is_violation:
                    self.stats["violations"] += 1
                
                status = "⚠️ VIOLATION!" if is_violation else "✓"
                print(f"  {status} #{track_id} ({class_name}): {speed:.1f} km/h [{direction}]")
                
                return measurement
        
        return None
    
    def _try_capture_plate_in_zone(self, frame: np.ndarray, track_id: int, 
                                    bbox: np.ndarray, frame_number: int):
        """Try to capture plate while vehicle is between the two lines (zone capture).
        Note: Currently unused - kept for reference but single-frame works better."""
        if self.lp_model is None or not self.capture_plates:
            return
        
        if track_id not in self.crossings:
            return
            
        crossing = self.crossings[track_id]
        if not crossing.in_zone:
            return
        
        # Get vehicle crop with padding
        x1, y1, x2, y2 = map(int, bbox)
        pad_x = int((x2 - x1) * 0.15)
        pad_y = int((y2 - y1) * 0.15)
        x1_pad = max(0, x1 - pad_x)
        y1_pad = max(0, y1 - pad_y)
        x2_pad = min(frame.shape[1], x2 + pad_x)
        y2_pad = min(frame.shape[0], y2 + pad_y)
        vehicle_img = frame[y1_pad:y2_pad, x1_pad:x2_pad]
        
        if vehicle_img.size == 0:
            return
        
        # Try to detect plate
        plate_img, confidence = self._detect_plate_with_model(vehicle_img)
        
        if plate_img is not None and plate_img.size > 0:
            # Check if this is better than our current best
            current_best = crossing.best_plate
            if current_best is None or confidence > current_best.confidence:
                # Store as new best plate
                crossing.best_plate = PlateCandidate(
                    plate_img=plate_img.copy(),
                    confidence=confidence,
                    frame_number=frame_number,
                    bbox=bbox.copy(),
                    frame_snapshot=frame.copy()  # Keep frame for snapshot
                )
    
    def capture_violation_snapshot(self, frame: np.ndarray, measurement: VehicleMeasurement, 
                                     bbox: Optional[np.ndarray] = None):
        """
        Save a complete violation record with proper folder structure.
        Enhanced with vehicle bounding box and plate text.
        
        Structure:
        output_dir/
        ├── violations.csv          # Cumulative CSV with all violations
        └── snapshots/
            ├── violation_001/
            │   ├── full_frame.jpg   # With vehicle bbox and info overlay
            │   └── plate.jpg
            ├── violation_002/
            │   ...
        """
        # Create folder structure
        snapshots_dir = os.path.join(self.output_dir, "snapshots")
        violation_num = len(self.measurements)
        violation_folder = os.path.join(snapshots_dir, f"violation_{violation_num:03d}")
        os.makedirs(violation_folder, exist_ok=True)
        
        # Create frame copy for annotation
        full_frame = frame.copy()
        
        # Draw vehicle bounding box if available
        if bbox is not None:
            x1, y1, x2, y2 = map(int, bbox)
            # Draw thick red box around the violating vehicle
            cv2.rectangle(full_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
            # Add label above the box
            label = f"#{measurement.track_id} | {measurement.speed_kmh:.0f} km/h"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(full_frame, (x1, y1 - 25), (x1 + label_size[0] + 10, y1), (0, 0, 255), -1)
            cv2.putText(full_frame, label, (x1 + 5, y1 - 7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Try to capture license plate first (to get plate text)
        plate_text = ""
        plate_path = None
        if self.capture_plates:
            plate_path, plate_text = self.capture_license_plate(frame, measurement, bbox, violation_folder)
        
        # Create info overlay box (semi-transparent)
        overlay = full_frame.copy()
        box_height = 160 if plate_text else 130
        cv2.rectangle(overlay, (10, 10), (480, box_height), (0, 0, 180), -1)
        cv2.addWeighted(overlay, 0.7, full_frame, 0.3, 0, full_frame)
        
        # Add text overlay
        cv2.putText(full_frame, "SPEED VIOLATION", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.putText(full_frame, f"Speed: {measurement.speed_kmh:.1f} km/h (Limit: {self.speed_limit})",
                    (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(full_frame, f"Vehicle: #{measurement.track_id} ({measurement.vehicle_class})",
                    (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(full_frame, f"Time: {measurement.timestamp}",
                    (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        if plate_text:
            cv2.putText(full_frame, f"Plate: {plate_text}", (20, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            measurement.plate_text = plate_text
        elif plate_path:
            cv2.putText(full_frame, "Plate captured (no OCR)", (20, 145),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        full_frame_path = os.path.join(violation_folder, "full_frame.jpg")
        cv2.imwrite(full_frame_path, full_frame)
        measurement.snapshot_path = full_frame_path
        
        # Update cumulative CSV
        self._update_violations_csv(measurement)
        
        return full_frame_path
    
    def _update_violations_csv(self, measurement: VehicleMeasurement):
        """Append violation to cumulative CSV file."""
        csv_path = os.path.join(self.output_dir, "violations.csv")
        file_exists = os.path.exists(csv_path)
        
        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow([
                    'Violation #', 'Track ID', 'Vehicle Type', 'Speed (km/h)', 
                    'Speed Limit', 'Over By', 'Direction', 'Timestamp', 'Frame',
                    'Full Frame Path', 'Plate Path', 'Plate Text'
                ])
            
            writer.writerow([
                len(self.measurements),
                measurement.track_id,
                measurement.vehicle_class,
                measurement.speed_kmh,
                self.speed_limit,
                f"+{measurement.speed_kmh - self.speed_limit:.1f}",
                measurement.direction,
                measurement.timestamp,
                measurement.frame_number,
                measurement.snapshot_path or '',
                measurement.license_plate_path or '',
                getattr(measurement, 'plate_text', '')
            ])
    
    def capture_license_plate(self, frame: np.ndarray, measurement: VehicleMeasurement,
                               bbox: Optional[np.ndarray], violation_folder: str) -> Tuple[Optional[str], str]:
        """
        Capture license plate for the specific violating vehicle.
        Returns (plate_image_path, plate_text) tuple.
        
        IMPORTANT: Prioritizes detection on VEHICLE CROP to ensure the plate
        matches the violating vehicle, not other vehicles in the frame.
        """
        plate_img = None
        confidence = 0.0
        
        # Note: Zone capture (trying to get plate while vehicle traverses between lines)
        # was disabled because it often captures wrong plates in crowded traffic.
        # Single-frame detection at violation moment is more accurate.
        
        # Method 1: Try LP detection on VEHICLE CROP if no best_plate
        if plate_img is None and bbox is not None:
            x1, y1, x2, y2 = map(int, bbox)
            # Add padding around vehicle for better plate visibility
            pad_x = int((x2 - x1) * 0.15)
            pad_y = int((y2 - y1) * 0.15)
            x1_pad = max(0, x1 - pad_x)
            y1_pad = max(0, y1 - pad_y)
            x2_pad = min(frame.shape[1], x2 + pad_x)
            y2_pad = min(frame.shape[0], y2 + pad_y)
            vehicle_img = frame[y1_pad:y2_pad, x1_pad:x2_pad]
            
            if vehicle_img.size > 0:
                if self.lp_model is not None:
                    plate_img, confidence = self._detect_plate_with_model(vehicle_img)
                    if plate_img is not None:
                        print(f"    📷 Plate detected on vehicle crop (conf: {confidence:.2f})")
                
                # Fallback: estimate plate region from vehicle bbox
                if plate_img is None:
                    plate_img = self._estimate_plate_region(vehicle_img, measurement.vehicle_class)
                    confidence = 0.1  # Low confidence for estimation
        
        # Method 2: Only use full frame if no bbox provided (rare)
        if plate_img is None and bbox is None and self.lp_model is not None:
            plate_img, confidence = self._detect_plate_with_model(frame)
            if plate_img is not None:
                print(f"    📷 Plate detected on full frame (conf: {confidence:.2f})")
        
        if plate_img is None or plate_img.size == 0:
            print(f"    ⚠️ No plate detected")
            return None, ""
        
        # Enhance and save
        plate_img = self._enhance_plate_image(plate_img)
        plate_path = os.path.join(violation_folder, "plate.jpg")
        cv2.imwrite(plate_path, plate_img)
        measurement.license_plate_path = plate_path
        
        # Try OCR to read plate text
        plate_text = self._read_plate_text(plate_img)
        
        print(f"    📷 License plate saved: {plate_path}")
        if plate_text:
            print(f"    📝 Plate text: {plate_text}")
        
        return plate_path, plate_text
    
    def _read_plate_text(self, plate_img: np.ndarray) -> str:
        """Read plate text using EasyOCR (if available)."""
        try:
            import easyocr
            if not hasattr(self, '_ocr_reader'):
                self._ocr_reader = easyocr.Reader(['en'], gpu=True, verbose=False)
            
            # Read text from plate image
            results = self._ocr_reader.readtext(plate_img)
            if results:
                # Combine all detected text, filter for plate-like patterns
                texts = [r[1].upper().replace(' ', '') for r in results if len(r[1]) > 2]
                if texts:
                    return texts[0]  # Return first/best result
        except ImportError:
            pass  # EasyOCR not installed
        except Exception as e:
            pass  # OCR failed
        return ""
    
    def _detect_plate_with_model(self, vehicle_img: np.ndarray) -> Tuple[Optional[np.ndarray], float]:
        """
        Detect license plate using dedicated YOLO model.
        Returns (plate_image, confidence) tuple for multi-frame comparison.
        """
        try:
            results = self.lp_model(vehicle_img, conf=0.3, verbose=False)[0]
            if len(results.boxes) > 0:
                # Get the best detection (highest confidence)
                best_idx = results.boxes.conf.argmax().item()
                confidence = results.boxes.conf[best_idx].item()
                best_box = results.boxes[best_idx].xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = map(int, best_box)
                plate_img = vehicle_img[y1:y2, x1:x2]
                if plate_img.size > 0:
                    return plate_img, confidence
        except Exception as e:
            pass
        return None, 0.0
    
    def _estimate_plate_region(self, vehicle_img: np.ndarray, vehicle_class: str) -> Optional[np.ndarray]:
        """
        Estimate license plate region based on vehicle type and camera angle.
        
        For best results with a dedicated LP model, the model should be used.
        This fallback captures the most likely plate region or the whole vehicle.
        
        Note: For side-view cameras (like VS13), the plate is not visible,
        so we capture the entire vehicle for identification purposes.
        """
        h, w = vehicle_img.shape[:2]
        
        if h < 30 or w < 30:
            return None
        
        # For side-view cameras where vehicle is wider than tall,
        # capture the front of the vehicle (right side for L→R traffic)
        aspect_ratio = w / h
        
        if aspect_ratio > 1.5:  # Wide aspect = side view (lowered from 2.0)
            # Side view - capture the RIGHT portion (front of vehicle approaching L→R)
            # The plate is on the front, which appears on the RIGHT side of the bbox
            front_start = int(w * 0.60)  # Get the rightmost 40% where plate is
            plate_region = vehicle_img[:, front_start:]
            return plate_region if plate_region.size > 0 else vehicle_img
        
        # Front/rear view estimation based on vehicle type
        if vehicle_class in ["bus", "truck"]:
            # Larger vehicles - plates typically lower and centered
            top = int(h * 0.65)
            bottom = int(h * 0.95)
            left = int(w * 0.20)
            right = int(w * 0.80)
        else:
            # Cars and motorcycles - plate in lower center
            top = int(h * 0.60)
            bottom = int(h * 0.95)
            left = int(w * 0.15)
            right = int(w * 0.85)
        
        # Extract plate region
        plate_region = vehicle_img[top:bottom, left:right]
        
        if plate_region.size == 0:
            return vehicle_img  # Fallback to whole vehicle
        
        return plate_region
    
    def _enhance_plate_image(self, plate_img: np.ndarray) -> np.ndarray:
        """
        Enhance license plate/vehicle image for better identification.
        
        Applies:
        - High-quality upscaling (Lanczos)
        - Mild denoising
        - Subtle sharpening (preserves original look)
        """
        h, w = plate_img.shape[:2]
        
        # Upscale small images with high quality Lanczos interpolation
        min_size = 300
        if w < min_size or h < min_size:
            scale = max(min_size / w, min_size / h)
            plate_img = cv2.resize(plate_img, None, fx=scale, fy=scale, 
                                   interpolation=cv2.INTER_LANCZOS4)
        
        # Mild denoising (preserves edges)
        denoised = cv2.fastNlMeansDenoisingColored(plate_img, None, 5, 5, 7, 21)
        
        # Subtle unsharp mask for clarity
        gaussian = cv2.GaussianBlur(denoised, (0, 0), 2.0)
        sharpened = cv2.addWeighted(denoised, 1.3, gaussian, -0.3, 0)
        
        return sharpened
    
    def draw_radar_overlay(self, frame: np.ndarray, current_vehicle_info: List[Dict]):
        """Draw the radar-style overlay on the frame."""
        # Draw detection lines
        if self.orientation == "vertical":
            cv2.line(frame, (self.line1_pos, 0), (self.line1_pos, self.frame_height), 
                     (0, 255, 0), 3)
            cv2.line(frame, (self.line2_pos, 0), (self.line2_pos, self.frame_height), 
                     (0, 255, 0), 3)
            cv2.putText(frame, "DETECTION ZONE", 
                        (self.line1_pos + 10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            cv2.line(frame, (0, self.line1_pos), (self.frame_width, self.line1_pos), 
                     (0, 255, 0), 3)
            cv2.line(frame, (0, self.line2_pos), (self.frame_width, self.line2_pos), 
                     (0, 255, 0), 3)
        
        # Draw statistics panel
        panel_x = 10
        panel_y = 60
        panel_w = 280
        panel_h = 130
        
        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), 
                      (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Statistics
        avg_speed = self.stats["total_speed"] / max(1, self.stats["total_vehicles"])
        
        cv2.putText(frame, f"RADAR SPEED TRAP", (panel_x + 10, panel_y + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, f"Speed Limit: {self.speed_limit:.0f} km/h", 
                    (panel_x + 10, panel_y + 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"Vehicles: {self.stats['total_vehicles']}", 
                    (panel_x + 10, panel_y + 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"Violations: {self.stats['violations']}", 
                    (panel_x + 10, panel_y + 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255) if self.stats['violations'] > 0 else (255, 255, 255), 1)
        cv2.putText(frame, f"Avg Speed: {avg_speed:.1f} km/h", 
                    (panel_x + 10, panel_y + 125),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Recent measurements panel
        if self.measurements:
            recent_y = panel_y + panel_h + 20
            cv2.putText(frame, "Recent:", (panel_x + 10, recent_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            for i, m in enumerate(self.measurements[-3:]):
                y = recent_y + 20 + (i * 20)
                color = (0, 0, 255) if m.is_violation else (0, 255, 0)
                text = f"#{m.track_id} {m.speed_kmh:.0f} km/h"
                cv2.putText(frame, text, (panel_x + 10, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    def process_frame(self, frame: np.ndarray, frame_number: int, timestamp: float) -> Tuple[np.ndarray, Optional[VehicleMeasurement], Optional[np.ndarray]]:
        """Process a single frame. Returns (annotated_frame, measurement, vehicle_bbox)."""
        
        if self.line1_pos is None:
            self.setup_lines(frame.shape[0], frame.shape[1], self.fps)
        
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
        new_measurement = None
        measurement_bbox = None
        current_vehicles = []
        
        if len(tracked) > 0:
            for i in range(len(tracked)):
                bbox = tracked.xyxy[i]
                tracking_coord = self.get_tracking_coordinate(bbox)
                
                track_id = tracked.tracker_id[i]
                class_id = tracked.class_id[i]
                class_name = self.VEHICLE_CLASSES.get(class_id, "vehicle")
                
                # Check for line crossings
                measurement = self.check_line_crossing(
                    track_id, tracking_coord, frame_number, timestamp, class_name
                )
                
                # Try to capture plate while vehicle is in detection zone
                # This runs on every frame while in zone to find best plate
                self._try_capture_plate_in_zone(frame, track_id, bbox, frame_number)
                
                if measurement:
                    new_measurement = measurement
                    measurement_bbox = bbox
                
                # Create label
                crossing = self.crossings.get(track_id)
                existing_measurement = next((m for m in self.measurements if m.track_id == track_id), None)
                
                if existing_measurement:
                    color_prefix = "!!" if existing_measurement.is_violation else ""
                    label = f"{color_prefix}#{track_id} {existing_measurement.speed_kmh:.0f} km/h"
                elif crossing and crossing.line1_timestamp is not None:
                    label = f"#{track_id} [timing...]"
                else:
                    label = f"#{track_id} {class_name}"
                
                labels.append(label)
                current_vehicles.append({
                    'track_id': int(track_id),
                    'class': class_name
                })
        
        # Annotate
        annotated = frame.copy()
        self.draw_radar_overlay(annotated, current_vehicles)
        annotated = self.trace_annotator.annotate(annotated, tracked)
        annotated = self.box_annotator.annotate(annotated, tracked)
        annotated = self.label_annotator.annotate(annotated, tracked, labels)
        
        return annotated, new_measurement, measurement_bbox
    
    def save_report(self, output_path: str):
        """Save CSV report of all measurements."""
        csv_path = output_path.replace('.mp4', '_report.csv')
        
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'Track ID', 'Vehicle Type', 'Speed (km/h)', 'Travel Time (s)',
                'Direction', 'Violation', 'Timestamp', 'Frame', 'Snapshot', 'License Plate'
            ])
            
            for m in self.measurements:
                writer.writerow([
                    m.track_id,
                    m.vehicle_class,
                    m.speed_kmh,
                    m.travel_time_sec,
                    m.direction,
                    'YES' if m.is_violation else 'NO',
                    m.timestamp,
                    m.frame_number,
                    m.snapshot_path or '',
                    m.license_plate_path or ''
                ])
        
        print(f"\nReport saved: {csv_path}")
        return csv_path
    
    def process_video(
        self,
        video_path: str,
        output_path: Optional[str] = None,
        display: bool = True,
        resize_width: int = 1280
    ) -> Dict:
        """Process a video file."""
        print(f"\n{'='*60}")
        print("RADAR SPEED TRAP")
        print(f"{'='*60}")
        print(f"Video: {video_path}")
        print(f"Speed limit: {self.speed_limit} km/h")
        
        cap = cv2.VideoCapture(video_path)
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if resize_width and resize_width < width:
            scale = resize_width / width
            new_w, new_h = resize_width, int(height * scale)
        else:
            new_w, new_h = width, height
            scale = 1.0
        
        print(f"Video: {width}x{height} @ {self.fps:.1f} FPS → {new_w}x{new_h}")
        
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, self.fps, (new_w, new_h))
        
        frame_count = 0
        start_time = time.time()
        
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                if scale != 1.0:
                    frame = cv2.resize(frame, (new_w, new_h))
                
                timestamp = frame_count / self.fps
                annotated, measurement, bbox = self.process_frame(frame, frame_count, timestamp)
                
                # Capture snapshot for violations
                if measurement and measurement.is_violation:
                    self.capture_violation_snapshot(frame.copy(), measurement, bbox)
                
                # Frame counter
                cv2.putText(annotated, f"Frame: {frame_count}/{total_frames}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                if writer:
                    writer.write(annotated)
                
                if display:
                    cv2.imshow("Radar Speed Trap", annotated)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                frame_count += 1
                if frame_count % 100 == 0:
                    print(f"Progress: {100*frame_count/total_frames:.1f}% | "
                          f"Vehicles: {self.stats['total_vehicles']} | "
                          f"Violations: {self.stats['violations']}")
        
        finally:
            cap.release()
            if writer:
                writer.release()
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass  # Headless system
        
        total_time = time.time() - start_time
        avg_speed = self.stats["total_speed"] / max(1, self.stats["total_vehicles"])
        
        print(f"\n{'='*60}")
        print("RESULTS")
        print(f"{'='*60}")
        print(f"Frames processed: {frame_count}")
        print(f"Processing FPS: {frame_count / total_time:.1f}")
        print(f"Vehicles measured: {self.stats['total_vehicles']}")
        print(f"Violations: {self.stats['violations']}")
        print(f"Average speed: {avg_speed:.1f} km/h")
        print(f"Max speed: {self.stats['max_speed']:.1f} km/h")
        print(f"{'='*60}")
        
        if output_path:
            self.save_report(output_path)
            print(f"Video: {output_path}")
        
        return {
            "frames": frame_count,
            "processing_fps": round(frame_count / total_time, 1),
            "vehicles": self.stats["total_vehicles"],
            "violations": self.stats["violations"],
            "avg_speed": round(avg_speed, 1),
            "max_speed": round(self.stats["max_speed"], 1),
            "measurements": [
                {
                    "track_id": m.track_id,
                    "class": m.vehicle_class,
                    "speed_kmh": m.speed_kmh,
                    "direction": m.direction,
                    "violation": m.is_violation
                }
                for m in self.measurements
            ]
        }


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Radar Speed Trap System")
    parser.add_argument("--video", "-v", type=str, required=True)
    parser.add_argument("--output", "-o", type=str, default=None)
    parser.add_argument("--model", "-m", type=str, default="yolo11m.pt")
    parser.add_argument("--distance", "-d", type=float, default=26.0,
                       help="Distance between detection lines (meters)")
    parser.add_argument("--limit", type=float, default=80.0,
                       help="Speed limit (km/h)")
    parser.add_argument("--line1", type=float, default=0.30)
    parser.add_argument("--line2", type=float, default=0.70)
    parser.add_argument("--orientation", type=str, default="vertical",
                       choices=["vertical", "horizontal"])
    parser.add_argument("--confidence", "-c", type=float, default=0.4)
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--resize", type=int, default=1280)
    parser.add_argument("--output-dir", type=str, default="radar_output")
    
    args = parser.parse_args()
    
    radar = RadarSpeedTrap(
        model_path=args.model,
        line1_percent=args.line1,
        line2_percent=args.line2,
        distance_meters=args.distance,
        speed_limit_kmh=args.limit,
        confidence_threshold=args.confidence,
        orientation=args.orientation,
        output_dir=args.output_dir
    )
    
    stats = radar.process_video(
        video_path=args.video,
        output_path=args.output,
        display=not args.no_display,
        resize_width=args.resize
    )
    
    return stats


if __name__ == "__main__":
    main()
