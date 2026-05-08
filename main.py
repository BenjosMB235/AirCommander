# main.py — AirCommander Phase 2
# Objectif : reconnaissance de gestes (main ouverte, pinch, geste V)

import os
os.environ["GLOG_minloglevel"] = "3"
os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import math

# --- Modèle ---
MODEL_PATH = "hand_landmarker.task"

# --- Détecteur ---
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

# --- Connexions pour dessiner le squelette ---
CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17)
]

# ─────────────────────────────────────────────
# FONCTIONS DE RECONNAISSANCE DE GESTES
# ─────────────────────────────────────────────

def get_fingers_state(landmarks):
    """
    Retourne une liste de 5 booléens indiquant si chaque doigt est levé.
    Ordre : [pouce, index, majeur, annulaire, auriculaire]
    """
    fingers = []

    # Pouce : comparaison horizontale (x) car il est sur le côté
    # Si le bout du pouce (4) est plus à gauche que son articulation (3) → levé
    thumb_tip = landmarks[4]
    thumb_ip  = landmarks[3]
    fingers.append(thumb_tip.x < thumb_ip.x)

    # Les 4 autres doigts : comparaison verticale (y)
    # TIP en haut (y petit) par rapport à PIP → doigt levé
    finger_tips = [8, 12, 16, 20]   # Index, Majeur, Annulaire, Auriculaire
    finger_pips = [6, 10, 14, 18]   # Leurs articulations PIP

    for tip, pip in zip(finger_tips, finger_pips):
        fingers.append(landmarks[tip].y < landmarks[pip].y)

    return fingers  # [pouce, index, majeur, annulaire, auriculaire]


def get_distance(p1, p2):
    """
    Calcule la distance euclidienne entre deux landmarks (coordonnées normalisées).
    """
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)


def detect_gesture(landmarks):
    """
    Identifie le geste principal parmi :
    - 'PINCH'       : pouce + index rapprochés
    - 'OPEN_HAND'   : tous les doigts levés
    - 'FIST'        : poing fermé
    - 'VICTORY'     : geste V (index + majeur levés)
    - 'POINTING'    : index seul levé
    - 'THUMBS_UP'   : pouce seul levé
    - 'THREE'       : index + majeur + annulaire levés
    - 'FOUR'        : tous sauf le pouce
    - 'UNKNOWN'     : geste non reconnu
    """
    fingers = get_fingers_state(landmarks)
    # fingers = [pouce, index, majeur, annulaire, auriculaire]

    # Distance normalisée pouce-index
    pinch_distance = get_distance(landmarks[4], landmarks[8])

    # 1. PINCH : seuil relevé à 0.09 pour être plus facile à déclencher
    if pinch_distance < 0.09:
        return "PINCH", fingers

    # 2. OPEN_HAND : tous les doigts levés
    if all(fingers):
        return "OPEN_HAND", fingers

    # 3. FIST : tous les doigts repliés
    if not any(fingers):
        return "FIST", fingers

    # 4. VICTORY (V) : index + majeur levés, annulaire + auriculaire repliés
    if fingers[1] and fingers[2] and not fingers[3] and not fingers[4]:
        return "VICTORY", fingers

    # 5. POINTING : index seul levé
    if fingers[1] and not fingers[2] and not fingers[3] and not fingers[4]:
        return "POINTING", fingers

    # 6. THUMBS_UP : pouce seul levé
    if fingers[0] and not fingers[1] and not fingers[2] and not fingers[3] and not fingers[4]:
        return "THUMBS_UP", fingers

    # 7. THREE : index + majeur + annulaire levés
    if fingers[1] and fingers[2] and fingers[3] and not fingers[4]:
        return "THREE", fingers

    # 8. FOUR : index + majeur + annulaire + auriculaire levés (pouce replié)
    if not fingers[0] and fingers[1] and fingers[2] and fingers[3] and fingers[4]:
        return "FOUR", fingers

    return "UNKNOWN", fingers
    """
    Identifie le geste principal parmi :
    - 'PINCH'       : pouce + index rapprochés
    - 'VICTORY'     : geste V (index + majeur levés)
    - 'OPEN_HAND'   : tous les doigts levés
    - 'FIST'        : poing fermé
    - 'POINTING'    : index seul levé
    - 'UNKNOWN'     : geste non reconnu
    """
    fingers = get_fingers_state(landmarks)
    # fingers = [pouce, index, majeur, annulaire, auriculaire]

    # Distance normalisée pouce-index
    pinch_distance = get_distance(landmarks[4], landmarks[8])

    # 1. PINCH : pouce et index très proches
    if pinch_distance < 0.06:
        return "PINCH", fingers

    # 2. OPEN_HAND : tous les doigts levés
    if all(fingers):
        return "OPEN_HAND", fingers

    # 3. FIST : tous les doigts repliés
    if not any(fingers):
        return "FIST", fingers

    # 4. VICTORY (V) : index + majeur levés, annulaire + auriculaire repliés
    if fingers[1] and fingers[2] and not fingers[3] and not fingers[4]:
        return "VICTORY", fingers

    # 5. POINTING : index seul levé
    if fingers[1] and not fingers[2] and not fingers[3] and not fingers[4]:
        return "POINTING", fingers

    return "UNKNOWN", fingers


# ─────────────────────────────────────────────
# BOUCLE PRINCIPALE
# ─────────────────────────────────────────────

cap = cv2.VideoCapture(0)
frame_index = 0

# Couleurs par geste (BGR)
GESTURE_COLORS = {
    "PINCH":     (0, 255, 0),    # Vert
    "OPEN_HAND": (255, 200, 0),  # Bleu clair
    "FIST":      (0, 0, 255),    # Rouge
    "VICTORY":   (200, 0, 255),  # Violet
    "POINTING":  (0, 255, 255),  # Jaune
    "THUMBS_UP": (0, 165, 255),  # Orange
    "THREE":     (255, 0, 150),  # Rose
    "FOUR":      (150, 255, 0),  # Vert clair
    "UNKNOWN":   (150, 150, 150) # Gris
}
print("AirCommander Phase 2 — Reconnaissance de gestes")
print("Gestes disponibles : PINCH | OPEN_HAND | FIST | VICTORY | POINTING")
print("Appuie sur 'q' pour quitter")

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

    if result.hand_landmarks:
        landmarks = result.hand_landmarks[0]
        points = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]

        # Reconnaître le geste
        gesture, fingers = detect_gesture(landmarks)
        color = GESTURE_COLORS[gesture]

        # Dessiner le squelette avec la couleur du geste
        for start, end in CONNECTIONS:
            cv2.line(frame, points[start], points[end], color, 2)
        for x, y in points:
            cv2.circle(frame, (x, y), 4, color, -1)

        # Afficher le geste détecté
        cv2.putText(frame, f"Geste : {gesture}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

        # Afficher l'état de chaque doigt (debug)
        finger_names = ["Pouce", "Index", "Majeur", "Annulaire", "Auriculaire"]
        debug_text = "  ".join([
            f"{name[0]}:{'O' if state else 'X'}"
            for name, state in zip(finger_names, fingers)
        ])
        cv2.putText(frame, debug_text, (10, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

        # Afficher la distance pinch avec indicateur visuel
        pinch_dist = get_distance(landmarks[4], landmarks[8])
        PINCH_THRESHOLD = 0.09

        # Couleur de la barre : vert si dans la zone, rouge sinon
        ratio = min(pinch_dist / PINCH_THRESHOLD, 1.0)  # 0.0 = pincé, 1.0 = ouvert
        bar_color = (0, int(255 * (1 - ratio)), int(255 * ratio))  # Vert→Rouge

        # Barre de progression horizontale
        bar_x, bar_y = 10, 115
        bar_w = 200
        bar_filled = int(bar_w * ratio)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + 15), (50, 50, 50), -1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_filled, bar_y + 15), bar_color, -1)
        cv2.putText(frame, f"Pinch: {pinch_dist:.3f} / {PINCH_THRESHOLD}", (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200, 200, 200), 1)

    else:
        cv2.putText(frame, "Aucune main", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

    cv2.imshow("AirCommander - Phase 2", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
detector.close()