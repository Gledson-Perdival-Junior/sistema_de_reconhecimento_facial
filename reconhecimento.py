import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import (
    PoseLandmarker, PoseLandmarkerOptions,
    FaceLandmarker, FaceLandmarkerOptions,
    GestureRecognizer, GestureRecognizerOptions,
    HandLandmarker, HandLandmarkerOptions,
)
import urllib.request
import os

SMILE_THRESHOLD    = 0.3
TOUCH_THRESHOLD    = 0.05
EYE_WIDE_THRESHOLD = 0.48
WINDOW_WIDTH       = 720
WINDOW_HEIGHT      = 450
EMOJI_WINDOW_SIZE  = (WINDOW_WIDTH, WINDOW_HEIGHT)

try:
    smiling_emoji      = cv2.imread("smile.jpg")
    straight_face_emoji= cv2.imread("plain.png")
    hands_up_emoji     = cv2.imread("air.jpg")
    touching_face_emoji= cv2.imread("touch.png")
    scary_face_emoji   = cv2.imread("scary.jpg")

    for name, img in [("smile.jpg", smiling_emoji), ("plain.png", straight_face_emoji),
                      ("air.jpg", hands_up_emoji),  ("touch.png", touching_face_emoji),
                      ("scary.jpg", scary_face_emoji)]:
        if img is None:
            raise FileNotFoundError(f"{name} não encontrado")

    smiling_emoji       = cv2.resize(smiling_emoji,       EMOJI_WINDOW_SIZE)
    straight_face_emoji = cv2.resize(straight_face_emoji, EMOJI_WINDOW_SIZE)
    hands_up_emoji      = cv2.resize(hands_up_emoji,      EMOJI_WINDOW_SIZE)
    touching_face_emoji = cv2.resize(touching_face_emoji, EMOJI_WINDOW_SIZE)
    scary_face_emoji    = cv2.resize(scary_face_emoji,    EMOJI_WINDOW_SIZE)

except Exception as e:
    print("Erro ao carregar imagens:", e)
    exit()

base_opts_pose = mp_python.BaseOptions(model_asset_path="pose_landmarker.task")
pose_opts = PoseLandmarkerOptions(
    base_options=base_opts_pose,
    running_mode=vision.RunningMode.IMAGE,
    num_poses=1,
    min_pose_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

base_opts_face = mp_python.BaseOptions(model_asset_path="face_landmarker.task")
face_opts = FaceLandmarkerOptions(
    base_options=base_opts_face,
    running_mode=vision.RunningMode.IMAGE,
    num_faces=1,
    min_face_detection_confidence=0.5,
)

base_opts_hand = mp_python.BaseOptions(model_asset_path="hand_landmarker.task")
hand_opts = HandLandmarkerOptions(
    base_options=base_opts_hand,
    running_mode=vision.RunningMode.IMAGE,
    num_hands=2,
    min_hand_detection_confidence=0.5,
)

pose_detector = PoseLandmarker.create_from_options(pose_opts)
face_detector = FaceLandmarker.create_from_options(face_opts)
hand_detector = HandLandmarker.create_from_options(hand_opts)

def eye_aspect_ratio(landmarks, ids):
    left, right, top, bottom = ids
    lx, ly = landmarks[left].x, landmarks[left].y
    rx, ry = landmarks[right].x, landmarks[right].y
    tx, ty = landmarks[top].x, landmarks[top].y
    bx, by = landmarks[bottom].x, landmarks[bottom].y
    vertical   = np.hypot(tx - bx, ty - by)
    horizontal = np.hypot(lx - rx, ly - ry)
    return vertical / horizontal if horizontal != 0 else 0

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Erro ao abrir webcam.")
    exit()

cv2.namedWindow('Camera Feed',  cv2.WINDOW_NORMAL)
cv2.namedWindow('Emoji Output', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Camera Feed',  WINDOW_WIDTH, WINDOW_HEIGHT)
cv2.resizeWindow('Emoji Output', WINDOW_WIDTH, WINDOW_HEIGHT)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        continue

    frame = cv2.flip(frame, 1)
    rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    current_state = "STRAIGHT_FACE"

    # ── Pose: mãos levantadas ──────────────────
    pose_result = pose_detector.detect(mp_image)
    if pose_result.pose_landmarks:
        lm = pose_result.pose_landmarks[0]
        # índices: 11=left shoulder, 12=right shoulder, 15=left wrist, 16=right wrist
        ls, rs = lm[11], lm[12]
        lw, rw = lm[15], lm[16]
        if lw.y < ls.y or rw.y < rs.y:
            current_state = "HANDS_UP"

    # ── Face Mesh: sorriso e olhos arregalados ──
    face_result = face_detector.detect(mp_image)
    if current_state != "HANDS_UP" and face_result.face_landmarks:
        face = face_result.face_landmarks[0]

        # Sorriso
        left_corner  = face[291]
        right_corner = face[61]
        upper_lip    = face[13]
        lower_lip    = face[14]

        mouth_width  = np.hypot(right_corner.x - left_corner.x,
                                right_corner.y - left_corner.y)
        mouth_height = np.hypot(lower_lip.x - upper_lip.x,
                                lower_lip.y - upper_lip.y)
        if mouth_width > 0 and (mouth_height / mouth_width) > SMILE_THRESHOLD:
            current_state = "SMILING"

    if current_state not in ["HANDS_UP", "SMILING"] and face_result.face_landmarks:
        face = face_result.face_landmarks[0]

        left_eye_ids  = [362, 263, 386, 374]
        right_eye_ids = [33,  133, 159, 145]
        left_EAR  = eye_aspect_ratio(face, left_eye_ids)
        right_EAR = eye_aspect_ratio(face, right_eye_ids)
        if (left_EAR + right_EAR) / 2 > EYE_WIDE_THRESHOLD:
            current_state = "EYES_WIDE"

    # ── Mão tocando o rosto ────────────────────
    if current_state not in ["HANDS_UP", "SMILING", "EYES_WIDE"] and face_result.face_landmarks:
        hand_result = hand_detector.detect(mp_image)
        if hand_result.hand_landmarks:
            face_pts = [(lm.x, lm.y) for lm in face_result.face_landmarks[0]]
            # índices das pontas dos dedos: 4=thumb, 8=index, 12=middle, 16=ring
            finger_tips = [4, 8, 12, 16]

            found = False
            for hand in hand_result.hand_landmarks:
                if found:
                    break
                for tip_idx in finger_tips:
                    if found:
                        break
                    fx, fy = hand[tip_idx].x, hand[tip_idx].y
                    for (rx, ry) in face_pts:
                        if np.hypot(fx - rx, fy - ry) < TOUCH_THRESHOLD:
                            current_state = "TOUCHING_FACE"
                            found = True
                            break

    emoji_map = {
        "SMILING":       smiling_emoji,
        "HANDS_UP":      hands_up_emoji,
        "TOUCHING_FACE": touching_face_emoji,
        "EYES_WIDE":     scary_face_emoji,
        "STRAIGHT_FACE": straight_face_emoji,
    }
    emoji = emoji_map.get(current_state, straight_face_emoji)

    cam = cv2.resize(frame, (WINDOW_WIDTH, WINDOW_HEIGHT))
    cv2.putText(cam, current_state, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow('Camera Feed',  cam)
    cv2.imshow('Emoji Output', emoji)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
pose_detector.close()
face_detector.close()
hand_detector.close()