# main.py — AirCommander Phase 4
# Nouveautés : gesture buffer, stabilisation, HUD amélioré

import os
os.environ["GLOG_minloglevel"] = "3"
os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import math
import pyautogui
import time
from collections import deque

pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0

SCREEN_W, SCREEN_H = pyautogui.size()

# --- Modèle ---
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

# --- Connexions squelette ---
CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17)
]

# ─────────────────────────────────────────────
# PARAMÈTRES
# ─────────────────────────────────────────────

PINCH_THRESHOLD  = 0.09
DEAD_ZONE        = 0.005
EMA_ALPHA        = 0.5
SPEED_FACTOR     = 1.8
CLICK_COOLDOWN   = 0.5
BUFFER_SIZE      = 5     # Nombre de frames pour valider un geste
BUFFER_THRESHOLD = 4     # Nombre minimum de frames identiques pour valider

# ─────────────────────────────────────────────
# ÉTAT GLOBAL
# ─────────────────────────────────────────────

smooth_x, smooth_y   = 0.5, 0.5
prev_wrist_y         = None
last_left_click      = 0
last_right_click     = 0

# Gesture buffer : file des BUFFER_SIZE derniers gestes détectés
gesture_buffer       = deque(maxlen=BUFFER_SIZE)
stable_gesture       = "UNKNOWN"   # Geste validé après stabilisation

# Historique des gestes validés (pour affichage HUD)
gesture_history      = deque(maxlen=3)

# FPS
fps_times            = deque(maxlen=30)

# ─────────────────────────────────────────────
# FONCTIONS
# ─────────────────────────────────────────────

def get_fingers_state(landmarks):
    fingers = []
    fingers.append(landmarks[4].x < landmarks[3].x)
    for tip, pip in zip([8,12,16,20], [6,10,14,18]):
        fingers.append(landmarks[tip].y < landmarks[pip].y)
    return fingers


def get_distance(p1, p2):
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)


def detect_gesture(landmarks):
    fingers = get_fingers_state(landmarks)
    pinch_distance = get_distance(landmarks[4], landmarks[8])

    if pinch_distance < PINCH_THRESHOLD:
        return "PINCH", fingers
    if all(fingers):
        return "OPEN_HAND", fingers
    if not any(fingers):
        return "FIST", fingers
    if fingers[1] and fingers[2] and not fingers[3] and not fingers[4]:
        return "VICTORY", fingers
    if fingers[1] and not fingers[2] and not fingers[3] and not fingers[4]:
        return "POINTING", fingers
    if fingers[0] and not any(fingers[1:]):
        return "THUMBS_UP", fingers
    if fingers[1] and fingers[2] and fingers[3] and not fingers[4]:
        return "THREE", fingers
    if not fingers[0] and all(fingers[1:]):
        return "FOUR", fingers
    return "UNKNOWN", fingers


def stabilize_gesture(raw_gesture):
    """
    Ajoute le geste brut au buffer et retourne le geste stable
    seulement si BUFFER_THRESHOLD frames consécutives sont identiques.
    """
    gesture_buffer.append(raw_gesture)

    if len(gesture_buffer) < BUFFER_SIZE:
        return stable_gesture  # Pas encore assez de données

    # Compter les occurrences du geste le plus fréquent dans le buffer
    counts = {}
    for g in gesture_buffer:
        counts[g] = counts.get(g, 0) + 1
    most_common = max(counts, key=counts.get)

    if counts[most_common] >= BUFFER_THRESHOLD:
        return most_common
    return stable_gesture  # Pas assez stable → garder le geste précédent


def apply_ema(new_x, new_y, old_x, old_y, alpha):
    return alpha * new_x + (1-alpha) * old_x, alpha * new_y + (1-alpha) * old_y


def move_cursor(norm_x, norm_y):
    cx = (norm_x - 0.5) * SPEED_FACTOR + 0.5
    cy = (norm_y - 0.5) * SPEED_FACTOR + 0.5
    sx = max(0, min(SCREEN_W-1, int(cx * SCREEN_W)))
    sy = max(0, min(SCREEN_H-1, int(cy * SCREEN_H)))
    pyautogui.moveTo(sx, sy)


def draw_hud(frame, gesture, action_text, pinch_dist, fps):
    """Dessine le HUD en bas de la fenêtre."""
    h, w, _ = frame.shape

    # Fond semi-transparent en bas
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h-110), (w, h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # Geste actif
    color = GESTURE_COLORS.get(gesture, (150,150,150))
    cv2.putText(frame, f"{gesture}", (15, h-75),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 2)

    # Action en cours
    cv2.putText(frame, action_text, (15, h-45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1)

    # Historique des gestes
    history_text = "  >  ".join(gesture_history)
    cv2.putText(frame, history_text, (15, h-18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (140, 140, 140), 1)

    # FPS (coin supérieur droit)
    cv2.putText(frame, f"FPS: {fps:.0f}", (w-100, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)

    # Barre pinch (coin supérieur droit)
    ratio = min(pinch_dist / PINCH_THRESHOLD, 1.0)
    bar_color = (0, int(255*(1-ratio)), int(255*ratio))
    cv2.rectangle(frame, (w-110, 45), (w-10, 58), (50,50,50), -1)
    cv2.rectangle(frame, (w-110, 45), (w-10+int(-100*(1-ratio)), 58), bar_color, -1)
    cv2.putText(frame, "PINCH", (w-110, 72),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180,180,180), 1)


# ─────────────────────────────────────────────
# BOUCLE PRINCIPALE
# ─────────────────────────────────────────────

GESTURE_COLORS = {
    "PINCH":     (0, 255, 0),
    "OPEN_HAND": (255, 200, 0),
    "FIST":      (0, 0, 255),
    "VICTORY":   (200, 0, 255),
    "POINTING":  (0, 255, 255),
    "THUMBS_UP": (0, 165, 255),
    "THREE":     (255, 0, 150),
    "FOUR":      (150, 255, 0),
    "UNKNOWN":   (150, 150, 150)
}

cap = cv2.VideoCapture(0)
frame_index = 0

print("AirCommander Phase 4 — HUD & Stabilisation")
print("  POINTING → curseur | PINCH → clic G | VICTORY → clic D | FIST → scroll")
print("  'q' pour quitter")

while True:
    t_start = time.time()

    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
    if timestamp_ms == 0:
        timestamp_ms = frame_index * 33
    frame_index += 1

    result = detector.detect_for_video(mp_image, timestamp_ms)

    action_text  = ""
    pinch_dist   = 1.0

    if result.hand_landmarks:
        landmarks = result.hand_landmarks[0]
        points = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]

        # Geste brut → stabilisation
        raw_gesture, fingers = detect_gesture(landmarks)
        new_stable = stabilize_gesture(raw_gesture)

        # Mettre à jour l'historique si le geste change
        if new_stable != stable_gesture and new_stable != "UNKNOWN":
            gesture_history.append(new_stable)
        stable_gesture = new_stable

        color = GESTURE_COLORS.get(stable_gesture, (150,150,150))
        pinch_dist = get_distance(landmarks[4], landmarks[8])

        # ── Actions selon geste stable ──

        if stable_gesture == "POINTING":
            smooth_x, smooth_y = apply_ema(
                landmarks[8].x, landmarks[8].y,
                smooth_x, smooth_y, EMA_ALPHA
            )
            dist = math.sqrt((landmarks[8].x - smooth_x)**2 + (landmarks[8].y - smooth_y)**2)
            if dist > DEAD_ZONE:
                move_cursor(smooth_x, smooth_y)
            action_text = "Deplacement curseur"

        elif stable_gesture == "PINCH":
            now = time.time()
            if now - last_left_click > CLICK_COOLDOWN:
                pyautogui.click()
                last_left_click = now
                action_text = ">>> CLIC GAUCHE <<<"
            else:
                action_text = "Clic gauche (cooldown)"

        elif stable_gesture == "VICTORY":
            now = time.time()
            if now - last_right_click > CLICK_COOLDOWN:
                pyautogui.rightClick()
                last_right_click = now
                action_text = ">>> CLIC DROIT <<<"
            else:
                action_text = "Clic droit (cooldown)"

        elif stable_gesture == "FIST":
            wrist_y = landmarks[0].y
            if prev_wrist_y is not None:
                delta_y = wrist_y - prev_wrist_y
                if abs(delta_y) > 0.005:
                    scroll_amount = int(delta_y * -200)
                    pyautogui.scroll(scroll_amount)
                    action_text = f"SCROLL {'↑' if scroll_amount > 0 else '↓'}"
            prev_wrist_y = wrist_y

        else:
            prev_wrist_y = None

        # Squelette
        for start, end in CONNECTIONS:
            cv2.line(frame, points[start], points[end], color, 2)
        for x, y in points:
            cv2.circle(frame, (x, y), 4, color, -1)

    else:
        stable_gesture = stabilize_gesture("UNKNOWN")
        prev_wrist_y   = None
        cv2.putText(frame, "Aucune main", (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)

    # FPS
    fps_times.append(time.time() - t_start)
    fps = 1.0 / (sum(fps_times) / len(fps_times)) if fps_times else 0

    # HUD
    draw_hud(frame, stable_gesture, action_text, pinch_dist, fps)

    cv2.imshow("AirCommander - Phase 4", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
detector.close()