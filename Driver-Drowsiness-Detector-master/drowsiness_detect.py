"""Drowsiness detector using dlib landmarks and multi-signal logic."""

import json
import time
from collections import deque
from pathlib import Path

import cv2
import dlib
import numpy as np
import pygame
from imutils import face_utils
from scipy.spatial import distance

BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "config.json"
DEFAULT_CONFIG = {
    "eye_aspect_ratio_threshold": 0.15,
    "eye_aspect_ratio_consec_frames": 20,
    "camera_index": 0,
    "predictor_path": "shape_predictor_68_face_landmarks.dat",
    "alarm_sound_path": "audio/alert.wav",
    "alarm_volume": 1.0,
    "calibration_seconds": 3.0,
    "threshold_ratio": 0.75,
    "smoothing_window": 5,
    "perclos_window_seconds": 20.0,
    "perclos_threshold": 0.4,
    "warning_frames": 15,
    "critical_frames": 30,
    "blink_window_seconds": 60.0,
    "blink_rate_low_threshold": 8.0,
    "mar_threshold": 0.65,
    "yawn_consec_frames": 12,
    "yawn_window_seconds": 60.0,
    "head_pitch_down_threshold": 18.0,
    "head_pitch_up_threshold": 8.0,
    "signal_warmup_seconds": 45.0,
    "pitch_smoothing_window": 8,
    "mask_mode": False
}


def eye_aspect_ratio(eye):
    a = distance.euclidean(eye[1], eye[5])
    b = distance.euclidean(eye[2], eye[4])
    c = distance.euclidean(eye[0], eye[3])
    return (a + b) / (2 * c)


def mouth_aspect_ratio(mouth):
    a = distance.euclidean(mouth[2], mouth[10])
    b = distance.euclidean(mouth[4], mouth[8])
    c = distance.euclidean(mouth[0], mouth[6])
    return (a + b) / (2 * c)


def load_config():
    config = DEFAULT_CONFIG.copy()
    if CONFIG_PATH.exists():
        with CONFIG_PATH.open("r", encoding="utf-8") as config_file:
            config.update(json.load(config_file))
    return config


config = load_config()

pygame.mixer.init()
pygame.mixer.music.load(str(BASE_DIR / config["alarm_sound_path"]))
pygame.mixer.music.set_volume(float(config["alarm_volume"]))

static_threshold = float(config["eye_aspect_ratio_threshold"])
calibration_seconds = float(config["calibration_seconds"])
threshold_ratio = float(config["threshold_ratio"])
smoothing_window = max(1, int(config["smoothing_window"]))
perclos_window_seconds = float(config["perclos_window_seconds"])
perclos_threshold = float(config["perclos_threshold"])
warning_frames = int(config["warning_frames"])
critical_frames = int(config["critical_frames"])
blink_window_seconds = float(config["blink_window_seconds"])
blink_rate_low_threshold = float(config["blink_rate_low_threshold"])
mar_threshold = float(config["mar_threshold"])
yawn_consec_frames = int(config["yawn_consec_frames"])
yawn_window_seconds = float(config["yawn_window_seconds"])
head_pitch_down_threshold = float(config["head_pitch_down_threshold"])
head_pitch_up_threshold = float(config["head_pitch_up_threshold"])
signal_warmup_seconds = float(config["signal_warmup_seconds"])
pitch_smoothing_window = max(1, int(config["pitch_smoothing_window"]))
mask_mode = bool(config["mask_mode"])

is_alarm_on = False
last_alert_level = "NORMAL"
closed_counter = 0
yawn_counter = 0
alert_level = "NORMAL"
baseline_ear = None
dynamic_threshold = static_threshold
calibration_started_at = time.time()
calibration_samples = []
ear_smooth_buffer = deque(maxlen=smoothing_window)
perclos_buffer = deque()
blink_timestamps = deque()
yawn_timestamps = deque()
prev_is_closed = False
current_pitch = 0.0
pitch_buffer = deque(maxlen=pitch_smoothing_window)

face_cascade = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_default.xml")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(str(BASE_DIR / config["predictor_path"]))
(l_start, l_end) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(r_start, r_end) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(m_start, m_end) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

video_capture = cv2.VideoCapture(int(config["camera_index"]))
time.sleep(2)

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    timestamp = time.time()
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 0)

    face_rectangle = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in face_rectangle:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    smoothed_ear = None
    if len(faces) > 0:
        shape = predictor(gray, faces[0])
        shape = face_utils.shape_to_np(shape)
        left_eye = shape[l_start:l_end]
        right_eye = shape[r_start:r_end]
        mouth = shape[m_start:m_end]

        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)
        ear = (left_ear + right_ear) / 2.0

        ear_smooth_buffer.append(ear)
        smoothed_ear = float(np.mean(ear_smooth_buffer))
        mar = mouth_aspect_ratio(mouth)

        if (timestamp - calibration_started_at) <= calibration_seconds:
            calibration_samples.append(smoothed_ear)
            alert_level = "CALIBRATING"
        elif baseline_ear is None and calibration_samples:
            baseline_ear = float(np.mean(calibration_samples))
            dynamic_threshold = baseline_ear * threshold_ratio

        active_threshold = dynamic_threshold if baseline_ear is not None else static_threshold
        is_closed = smoothed_ear < active_threshold
        perclos_buffer.append((timestamp, is_closed))
        while perclos_buffer and (timestamp - perclos_buffer[0][0]) > perclos_window_seconds:
            perclos_buffer.popleft()

        closed_ratio = 0.0
        if perclos_buffer:
            closed_ratio = sum(1 for _, closed in perclos_buffer if closed) / len(perclos_buffer)

        if is_closed:
            closed_counter += 1
        else:
            closed_counter = 0

        # Blink is counted on closed->open transition to avoid overcounting.
        if prev_is_closed and not is_closed:
            blink_timestamps.append(timestamp)
        prev_is_closed = is_closed
        while blink_timestamps and (timestamp - blink_timestamps[0]) > blink_window_seconds:
            blink_timestamps.popleft()
        blink_rate = len(blink_timestamps) * (60.0 / max(1.0, blink_window_seconds))

        # Yawn signal is unreliable when wearing a mask.
        if not mask_mode:
            if mar > mar_threshold:
                yawn_counter += 1
                if yawn_counter == yawn_consec_frames:
                    yawn_timestamps.append(timestamp)
            else:
                yawn_counter = 0
            while yawn_timestamps and (timestamp - yawn_timestamps[0]) > yawn_window_seconds:
                yawn_timestamps.popleft()
            yawn_count = len(yawn_timestamps)
        else:
            yawn_counter = 0
            yawn_timestamps.clear()
            yawn_count = 0

        # Head pose (pitch) from a small set of facial landmarks.
        image_points = np.array(
            [
                shape[30],  # Nose tip
                shape[8],   # Chin
                shape[36],  # Left eye left corner
                shape[45],  # Right eye right corner
                shape[48],  # Left mouth corner
                shape[54],  # Right mouth corner
            ],
            dtype="double"
        )
        model_points = np.array(
            [
                (0.0, 0.0, 0.0),
                (0.0, -330.0, -65.0),
                (-225.0, 170.0, -135.0),
                (225.0, 170.0, -135.0),
                (-150.0, -150.0, -125.0),
                (150.0, -150.0, -125.0),
            ]
        )
        focal_length = frame.shape[1]
        center = (frame.shape[1] / 2.0, frame.shape[0] / 2.0)
        camera_matrix = np.array(
            [[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]],
            dtype="double"
        )
        dist_coeffs = np.zeros((4, 1))
        success, rotation_vector, _ = cv2.solvePnP(
            model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
        )
        if success:
            rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
            projection_matrix = np.hstack((rotation_matrix, np.zeros((3, 1))))
            _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(projection_matrix)
            current_pitch = float(euler_angles[0, 0])
        pitch_buffer.append(current_pitch)
        smoothed_pitch = float(np.mean(pitch_buffer))

        elapsed = timestamp - calibration_started_at
        feature_gate_open = elapsed >= signal_warmup_seconds
        head_nodging = feature_gate_open and (
            smoothed_pitch > head_pitch_down_threshold or smoothed_pitch < -head_pitch_up_threshold
        )
        low_blink_rate = feature_gate_open and (blink_rate < blink_rate_low_threshold)

        fatigue_score = 0
        if closed_counter >= warning_frames:
            fatigue_score += 1
        if closed_counter >= critical_frames and closed_ratio >= perclos_threshold:
            fatigue_score += 2
        if (not mask_mode) and yawn_count >= 2:
            fatigue_score += 1
        if low_blink_rate:
            fatigue_score += 1
        if head_nodging:
            fatigue_score += 1

        # Prevent false alarms: auxiliary signals alone must not trigger drowsy state.
        eye_drowsy_evidence = (
            closed_counter >= warning_frames
            or closed_ratio >= (perclos_threshold * 0.8)
        )

        if eye_drowsy_evidence and fatigue_score >= 4:
            alert_level = "CRITICAL"
        elif eye_drowsy_evidence and fatigue_score >= 2:
            alert_level = "WARNING"
        elif alert_level != "CALIBRATING":
            alert_level = "NORMAL"

        should_ring_once = (
            alert_level in ("WARNING", "CRITICAL")
            and last_alert_level not in ("WARNING", "CRITICAL")
        )
        if should_ring_once:
            pygame.mixer.music.play()
            is_alarm_on = True
        elif not pygame.mixer.music.get_busy():
            is_alarm_on = False

        last_alert_level = alert_level

        left_eye_hull = cv2.convexHull(left_eye)
        right_eye_hull = cv2.convexHull(right_eye)
        cv2.drawContours(frame, [left_eye_hull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [right_eye_hull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [cv2.convexHull(mouth)], -1, (255, 255, 0), 1)

        cv2.putText(frame, f"EAR(raw): {ear:.3f}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.putText(frame, f"EAR(smooth): {smoothed_ear:.3f}", (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.putText(frame, f"Threshold: {active_threshold:.3f}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.putText(frame, f"Counter: {closed_counter}", (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.putText(frame, f"PERCLOS: {closed_ratio:.2f}", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.putText(frame, f"Blink/min: {blink_rate:.1f}", (20, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        mar_text = "OFF(mask)" if mask_mode else f"{mar:.3f}"
        cv2.putText(frame, f"MAR: {mar_text}  Yawn: {yawn_count}", (20, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.putText(frame, f"Pitch: {smoothed_pitch:.1f}", (20, 205), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.putText(frame, f"FeatureGate: {'ON' if feature_gate_open else 'WARMUP'}", (20, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.putText(frame, f"MaskMode: {'ON' if mask_mode else 'OFF'}", (20, 255), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    else:
        closed_counter = 0
        yawn_counter = 0
        prev_is_closed = False
        if is_alarm_on and not pygame.mixer.music.get_busy():
            is_alarm_on = False
        if alert_level != "CALIBRATING":
            alert_level = "NO_FACE"
        last_alert_level = alert_level

    level_color = (0, 255, 0)
    if alert_level == "WARNING":
        level_color = (0, 200, 255)
    elif alert_level == "CRITICAL":
        level_color = (0, 0, 255)
    elif alert_level in ("CALIBRATING", "NO_FACE"):
        level_color = (255, 255, 0)

    cv2.putText(frame, f"Level: {alert_level}", (20, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.75, level_color, 2)
    cv2.putText(frame, f"Alarm: {'ON' if is_alarm_on else 'OFF'}", (20, 305), cv2.FONT_HERSHEY_SIMPLEX, 0.75, level_color, 2)
    if baseline_ear is not None:
        cv2.putText(frame, f"Baseline EAR: {baseline_ear:.3f}", (20, 330), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    if alert_level == "CRITICAL":
        cv2.putText(frame, "You are Drowsy", (120, 360), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    cv2.imshow("Driver Drowsiness Detector", frame)
    if (cv2.waitKey(1) & 0xFF) == ord("q"):
        break

video_capture.release()
cv2.destroyAllWindows()
