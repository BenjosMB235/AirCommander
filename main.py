# main.py — AirCommander Phase 3
# Objectif : contrôler la souris avec les gestes de la main

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

# Désactive la protection anti-crash de pyautogui (déplacer souris vers coin = stop)
pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0  # Pas de délai entre les actions

# --- Résolution de l'écran ---
SCREEN_W, SCREEN_H = pyautogui.size()
print(f"Résolution écran : {SCREEN_W}x{SCREEN_H}")

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
# PARAMÈTRES DE CONTRÔLE
# ─────────────────────────────────────────────

PINCH_THRESHOLD = 0.09      # Seuil de détection du pinch
DEAD_ZONE       = 0.005     # Mouvement minimum pour bouger la souris
EMA_ALPHA       = 0.5       # Lissage (0.1 = très lissé, 1.0 = brut)
CLICK_COOLDOWN  = 0.5       # Délai minimum entre deux clics (secondes)

# ─────────────────────────────────────────────
# ÉTAT GLOBAL
# ─────────────────────────────────────────────

# Position lissée du curseur (coordonnées normalisées 0-1)
smooth_x, smooth_y = 0.5, 0.5

# Pour le scroll : position y précédente du poignet
prev_wrist_y = None

# Timestamps des derniers clics (anti-spam)
last_left_click  = 0
last_right_click = 0

# ─────────────────────────────────────────────
# FONCTIONS
# ─────────────────────────────────────────────

def get_fingers_state(landmarks):
    fingers = []
    thumb_tip = landmarks[4]
    thumb_ip  = landmarks[3]
    fingers.append(thumb_tip.x < thumb_ip.x)

    finger_tips = [8, 12, 16, 20]
    finger_pips = [6, 10, 14, 18]
    for tip, pip in zip(finger_tips, finger_pips):
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
    if fingers[0] and not fingers[1] and not fingers[2] and not fingers[3] and not fingers[4]:
        return "THUMBS_UP", fingers
    if fingers[1] and fingers[2] and fingers[3] and not fingers[4]:
        return "THREE", fingers
    if not fingers[0] and fingers[1] and fingers[2] and fingers[3] and fingers[4]:
        return "FOUR", fingers
    return "UNKNOWN", fingers


def apply_ema(new_x, new_y, old_x, old_y, alpha):
    """Lissage exponentiel : réduit le tremblement du curseur."""
    sx = alpha * new_x + (1 - alpha) * old_x
    sy = alpha * new_y + (1 - alpha) * old_y
    return sx, sy


def move_cursor(norm_x, norm_y):
    """Convertit les coordonnées normalisées en pixels écran et déplace la souris."""
    # Le centre de la webcam (0.5, 0.5) correspond au centre de l'écran
    # On amplifie les déplacements depuis le centre avec SPEED_FACTOR
    SPEED_FACTOR = 1.8
    centered_x = (norm_x - 0.5) * SPEED_FACTOR + 0.5
    centered_y = (norm_y - 0.5) * SPEED_FACTOR + 0.5

    screen_x = int(centered_x * SCREEN_W)
    screen_y = int(centered_y * SCREEN_H)
    # Clamp pour rester dans les limites de l'écran
    screen_x = max(0, min(SCREEN_W - 1, screen_x))
    screen_y = max(0, min(SCREEN_H - 1, screen_y))
    pyautogui.moveTo(screen_x, screen_y)


# ─────────────────────────────────────────────
# BOUCLE PRINCIPALE
# ─────────────────────────────────────────────

cap = cv2.VideoCapture(0)
frame_index = 0

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

print("AirCommander Phase 3 — Contrôle souris actif")
print("  POINTING   → déplacer le curseur")
print("  PINCH      → clic gauche")
print("  VICTORY    → clic droit")
print("  FIST       → scroll (bouge la main haut/bas)")
print("  Appuie sur 'q' pour quitter")

while True:
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

    action_text = ""  # Message d'action affiché à l'écran

    if result.hand_landmarks:
        landmarks = result.hand_landmarks[0]
        points = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]

        gesture, fingers = detect_gesture(landmarks)
        color = GESTURE_COLORS[gesture]

        # Position brute de l'index (landmark 8) pour le curseur
        raw_x = landmarks[8].x
        raw_y = landmarks[8].y

        # ── POINTING : déplacer le curseur ──
        if gesture == "POINTING":
            # Appliquer le lissage EMA
            smooth_x, smooth_y = apply_ema(raw_x, raw_y, smooth_x, smooth_y, EMA_ALPHA)

            # Appliquer la dead zone
            dist_moved = math.sqrt((raw_x - smooth_x)**2 + (raw_y - smooth_y)**2)
            if dist_moved > DEAD_ZONE:
                move_cursor(smooth_x, smooth_y)

            action_text = "MODE : Deplacement curseur"

        # ── PINCH : clic gauche ──
        elif gesture == "PINCH":
            now = time.time()
            if now - last_left_click > CLICK_COOLDOWN:
                pyautogui.click()
                last_left_click = now
                action_text = ">>> CLIC GAUCHE <<<"
            else:
                action_text = "PINCH detecte (cooldown...)"

        # ── VICTORY : clic droit ──
        elif gesture == "VICTORY":
            now = time.time()
            if now - last_right_click > CLICK_COOLDOWN:
                pyautogui.rightClick()
                last_right_click = now
                action_text = ">>> CLIC DROIT <<<"
            else:
                action_text = "VICTORY detecte (cooldown...)"

        # ── FIST : scroll ──
        elif gesture == "FIST":
            wrist_y = landmarks[0].y  # Landmark 0 = poignet
            if prev_wrist_y is not None:
                delta_y = wrist_y - prev_wrist_y
                # delta_y positif = main descend → scroll vers le bas
                if abs(delta_y) > 0.005:  # Seuil minimal de mouvement
                    scroll_amount = int(delta_y * -200)  # Inverser pour sens naturel
                    pyautogui.scroll(scroll_amount)
                    action_text = f"SCROLL {'↑' if scroll_amount > 0 else '↓'} ({scroll_amount})"
            prev_wrist_y = wrist_y

        else:
            prev_wrist_y = None  # Reset scroll si autre geste

        # Dessiner le squelette
        for start, end in CONNECTIONS:
            cv2.line(frame, points[start], points[end], color, 2)
        for x, y in points:
            cv2.circle(frame, (x, y), 4, color, -1)

        # Affichage HUD
        cv2.putText(frame, f"Geste : {gesture}", (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        cv2.putText(frame, action_text, (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # Barre pinch
        pinch_dist = get_distance(landmarks[4], landmarks[8])
        ratio = min(pinch_dist / PINCH_THRESHOLD, 1.0)
        bar_color = (0, int(255 * (1 - ratio)), int(255 * ratio))
        cv2.rectangle(frame, (10, 115), (210, 130), (50, 50, 50), -1)
        cv2.rectangle(frame, (10, 115), (10 + int(200 * ratio), 130), bar_color, -1)
        cv2.putText(frame, f"Pinch: {pinch_dist:.3f}", (10, 112),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    else:
        cv2.putText(frame, "Aucune main", (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        prev_wrist_y = None

    cv2.imshow("AirCommander - Phase 3", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
detector.close()