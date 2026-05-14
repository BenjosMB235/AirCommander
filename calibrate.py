# calibrate.py — Session de calibrage personnalisé
# Lance ce script une fois avant d'utiliser AirCommander

import os
os.environ["GLOG_minloglevel"] = "3"
os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import json
import time
import numpy as np
from gesture_utils import get_distance, normalize_landmarks

MODEL_PATH = "hand_landmarker.task"
base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
    running_mode=vision.RunningMode.VIDEO
)
detector = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)

def get_landmarks_from_frame():
    """Capture une frame et retourne les landmarks si une main est détectée."""
    success, frame = cap.read()
    if not success:
        return None, None
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    global frame_index
    timestamp_ms = int(time.time() * 1000)  # Temps réel en millisecondes
    result = detector.detect_for_video(mp_image, timestamp_ms)
    if result.hand_landmarks:
        return frame, result.hand_landmarks[0]
    return frame, None


def collect_samples(instruction, n_samples=60, countdown=3):
    """
    Affiche une instruction, attend que l'utilisateur se prépare,
    puis collecte n_samples mesures avec timeout de sécurité.
    """
    print(f"\n>>> {instruction}")
    print(f"    Préparation dans {countdown} secondes...")

    # Countdown
    start = time.time()
    while time.time() - start < countdown:
        frame, _ = get_landmarks_from_frame()
        if frame is not None:
            remaining = countdown - int(time.time() - start)
            cv2.putText(frame, instruction, (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(frame, f"Debut dans : {remaining}s", (20, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
            cv2.imshow("AirCommander - Calibrage", frame)
            cv2.waitKey(1)

    # Collecte avec timeout
    samples = []
    timeout_start = time.time()
    MAX_DURATION = 15  # secondes max pour collecter les samples

    while len(samples) < n_samples:

        # Timeout de sécurité
        if time.time() - timeout_start > MAX_DURATION:
            print(f"    ⚠️  Timeout — seulement {len(samples)} samples collectés")
            break

        frame, landmarks = get_landmarks_from_frame()

        if frame is None:
            continue

        if landmarks is not None:
            value = get_distance(landmarks[4], landmarks[8])
            samples.append(value)
        else:
            # Afficher un message si aucune main détectée
            cv2.putText(frame, "Montre ta main !", (20, 140),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # Affichage progression (toujours, même sans main)
        progress = int(len(samples) / n_samples * 300)
        cv2.rectangle(frame, (20, 60), (320, 85), (50, 50, 50), -1)
        cv2.rectangle(frame, (20, 60), (20 + progress, 85), (0, 255, 100), -1)
        cv2.putText(frame, instruction, (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(frame, f"Collecte : {len(samples)}/{n_samples}", (20, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.imshow("AirCommander - Calibrage", frame)
        cv2.waitKey(1)

    # Vérification : assez de samples ?
    if len(samples) < 10:
        print("    ❌ Pas assez de samples — relance le calibrage")
        return None

    return samples


def run_calibration():
    print("=" * 50)
    print("  AIRCOMMANDER — SESSION DE CALIBRAGE")
    print("=" * 50)

    samples_pinch_closed = collect_samples(
        "PINCH : rapproche pouce et index au maximum", n_samples=60
    )
    if samples_pinch_closed is None:
        print("Calibrage annulé.")
        return

    pinch_closed_mean = float(np.mean(samples_pinch_closed))
    pinch_closed_std  = float(np.std(samples_pinch_closed))
    print(f"    Pinch fermé  → mean={pinch_closed_mean:.4f}  std={pinch_closed_std:.4f}")

    samples_pinch_open = collect_samples(
        "OPEN : ecarte bien pouce et index", n_samples=60
    )
    if samples_pinch_open is None:
        print("Calibrage annulé.")
        return

    pinch_open_mean = float(np.mean(samples_pinch_open))
    print(f"    Pinch ouvert → mean={pinch_open_mean:.4f}")

    # Nouvelle formule
    pinch_threshold = pinch_closed_mean + 2 * pinch_closed_std
    pinch_threshold = max(0.05, min(0.20, pinch_threshold))
    print(f"\n    Seuil PINCH calculé : {pinch_threshold:.4f}")
    print(f"    (mean + 2×std = {pinch_closed_mean:.4f} + 2×{pinch_closed_std:.4f})")

    config = {
        "pinch_threshold":   round(pinch_threshold, 4),
        "dead_zone":         0.005,
        "ema_alpha":         0.5,
        "speed_factor":      1.8,
        "click_cooldown":    0.5,
        "buffer_size":       5,
        "buffer_threshold":  4,
        "calibrated":        True,
        "pinch_closed_mean": round(pinch_closed_mean, 4),
        "pinch_open_mean":   round(pinch_open_mean, 4),
    }

    with open("config.json", "w") as f:
        json.dump(config, f, indent=2)

    print("\n✅ Calibrage terminé ! config.json sauvegardé.")
    cap.release()
    cv2.destroyAllWindows()
    detector.close()
    return config


if __name__ == "__main__":
    run_calibration()