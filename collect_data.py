# collect_data.py — Collecte de données d'entraînement pour le classifieur ML
# Pour chaque geste : appuie sur la touche indiquée pour enregistrer des samples

import os
os.environ["GLOG_minloglevel"] = "3"
os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import csv
import time
from gesture_utils import normalize_landmarks

MODEL_PATH = "hand_landmarker.task"
DATA_FILE  = "gesture_data.csv"

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

# Gestes à collecter et leurs touches associées
GESTURES = {
    ord('p'): "POINTING",
    ord('f'): "FIST",
    ord('o'): "OPEN_HAND",
    ord('v'): "VICTORY",
    ord('c'): "PINCH",
}

# Initialiser le CSV si nouveau fichier
if not os.path.exists(DATA_FILE):
    with open(DATA_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        # Header : label + 42 features (x0,y0,...,x20,y20)
        header = ["label"] + [f"{'x' if i%2==0 else 'y'}{i//2}" for i in range(42)]
        writer.writerow(header)
    print(f"✅ Nouveau fichier {DATA_FILE} créé")
else:
    print(f"📂 Fichier {DATA_FILE} existant — les nouvelles données seront ajoutées")

# Compteur de samples par geste
sample_counts = {g: 0 for g in GESTURES.values()}
recording = False
current_gesture = None
last_record_time = 0
RECORD_INTERVAL = 0.05  # 1 sample toutes les 50ms = ~20 samples/seconde

print("\n=== COLLECTE DE DONNÉES ===")
print("Touches :")
for key, gesture in GESTURES.items():
    print(f"  [{chr(key)}] → {gesture}")
print("  [q] → Quitter et sauvegarder")
print("\nMaintenez la touche enfoncée pour enregistrer en continu.")
print("Visez 100-150 samples par geste.\n")

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    timestamp_ms = int(time.time() * 1000)  # Temps réel en millisecondes

    result = detector.detect_for_video(mp_image, timestamp_ms)

    hand_detected = bool(result.hand_landmarks)

    # Enregistrement si touche maintenue + main détectée
    now = time.time()
    if recording and hand_detected and (now - last_record_time) > RECORD_INTERVAL:
        landmarks = result.hand_landmarks[0]
        features = normalize_landmarks(landmarks)
        with open(DATA_FILE, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([current_gesture] + features)
        sample_counts[current_gesture] += 1
        last_record_time = now

    # ── HUD ──
    # Fond header
    cv2.rectangle(frame, (0, 0), (w, 45), (30, 30, 30), -1)

    # Statut main
    hand_color = (0, 255, 0) if hand_detected else (0, 0, 255)
    hand_text  = "Main OK" if hand_detected else "Pas de main"
    cv2.putText(frame, hand_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, hand_color, 2)

    # Geste en cours
    if recording and current_gesture:
        cv2.putText(frame, f"REC: {current_gesture}", (w//2 - 80, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 80, 255), 2)
        # Point rouge clignotant
        if int(now * 2) % 2 == 0:
            cv2.circle(frame, (w - 30, 25), 10, (0, 0, 255), -1)

    # Compteurs par geste
    y_offset = 70
    for gesture, count in sample_counts.items():
        bar_w = min(int(count / 150 * 200), 200)
        color = (0, 200, 100) if count >= 100 else (0, 140, 255)
        cv2.rectangle(frame, (10, y_offset), (210, y_offset + 14), (50,50,50), -1)
        cv2.rectangle(frame, (10, y_offset), (10 + bar_w, y_offset + 14), color, -1)
        cv2.putText(frame, f"{gesture}: {count}", (220, y_offset + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        y_offset += 22

    # Instructions
    cv2.putText(frame, "Maintenez: P=Pointing F=Fist O=Open V=Victory C=Pinch | Q=Quitter",
                (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 160, 160), 1)

    # Squelette si main détectée
    if hand_detected and result.hand_landmarks:
        landmarks = result.hand_landmarks[0] if result.hand_landmarks else None
        if landmarks is None:
            cv2.imshow("AirCommander - Collecte de donnees", frame)
            continue
        points = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
        CONNECTIONS = [
            (0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
            (0,9),(9,10),(10,11),(11,12),(0,13),(13,14),(14,15),(15,16),
            (0,17),(17,18),(18,19),(19,20),(5,9),(9,13),(13,17)
        ]
        color = (0, 80, 255) if recording else (0, 255, 200)
        for s, e in CONNECTIONS:
            cv2.line(frame, points[s], points[e], color, 2)
        for x, y in points:
            cv2.circle(frame, (x, y), 3, color, -1)

    cv2.imshow("AirCommander - Collecte de donnees", frame)

    key = cv2.waitKey(30) & 0xFF

    if key == ord('q'):
        break
    elif key in GESTURES:
        # Toggle : si même geste → stop, sinon → nouveau geste
        if recording and current_gesture == GESTURES[key]:
            recording = False
            current_gesture = None
            print(f"⏹ Arrêt enregistrement")
        else:
            recording = True
            current_gesture = GESTURES[key]
            print(f"⏺ Enregistrement : {current_gesture}")

cap.release()
cv2.destroyAllWindows()
detector.close()

print("\n=== RÉSUMÉ DE LA COLLECTE ===")
for gesture, count in sample_counts.items():
    status = "✅" if count >= 100 else "⚠️  (vise 100+)"
    print(f"  {gesture}: {count} samples {status}")
print(f"\nDonnées sauvegardées dans {DATA_FILE}")