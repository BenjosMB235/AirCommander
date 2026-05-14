# gesture_utils.py — Utilitaires de normalisation et reconnaissance de gestes

import math
import numpy as np


def normalize_landmarks(landmarks):
    """
    Normalise les 21 landmarks pour être invariants à :
    - La position de la main dans l'image
    - La distance de la main à la caméra
    - (partiellement) L'orientation

    Retourne une liste de 42 floats [x0, y0, x1, y1, ..., x20, y20]
    prête à être utilisée comme feature vector pour un classifieur ML.
    """
    # Étape 1 : extraire les coordonnées brutes
    points = [(lm.x, lm.y) for lm in landmarks]

    # Étape 2 : translation → centrer sur le poignet (landmark 0)
    wrist_x, wrist_y = points[0]
    points = [(x - wrist_x, y - wrist_y) for x, y in points]

    # Étape 3 : mise à l'échelle → normaliser par la distance poignet-majeur (0→9)
    # Cette distance représente la "taille" de la main
    ref_dist = math.sqrt(points[9][0]**2 + points[9][1]**2)
    if ref_dist < 1e-6:
        ref_dist = 1e-6  # Éviter division par zéro

    points = [(x / ref_dist, y / ref_dist) for x, y in points]

    # Étape 4 : aplatir en vecteur 1D [x0, y0, x1, y1, ..., x20, y20]
    feature_vector = []
    for x, y in points:
        feature_vector.extend([x, y])

    return feature_vector  # 42 valeurs


def get_fingers_state(landmarks):
    """
    Détecte si chaque doigt est levé.
    Retourne [pouce, index, majeur, annulaire, auriculaire]
    """
    fingers = []
    fingers.append(landmarks[4].x < landmarks[3].x)
    for tip, pip in zip([8, 12, 16, 20], [6, 10, 14, 18]):
        fingers.append(landmarks[tip].y < landmarks[pip].y)
    return fingers


def get_distance(p1, p2):
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)


def detect_gesture_rules(landmarks, pinch_threshold=0.09):
    """
    Détection par règles géométriques (fallback si pas de modèle ML).
    """
    fingers = get_fingers_state(landmarks)
    pinch_distance = get_distance(landmarks[4], landmarks[8])

    if pinch_distance < pinch_threshold:
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