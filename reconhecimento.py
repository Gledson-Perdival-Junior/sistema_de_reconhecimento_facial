import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

SMILE_THRESHOLD = 0.3
TOUCH_THRESHOLD = 0.03
EYE_WIDE_THRESHOLD = 0.48
WINDOW_WIDTH = 720
WINDOW_HEIGHT = 450
EMOJI_WINDOW_SIZE = (WINDOW_WIDTH, WINDOW_HEIGHT)

try:
    smiling_emoji = cv2.imread("smile.jpg")
    straight_face_emoji = cv2.imread("plain.png")
    hands_up_emoji = cv2.imread("air.jpg")
    touching_face_emoji = cv2.imread("touch.png")   
    scary_face_emoji = cv2.imread("scary.jpg") 

    if smiling_emoji is None: raise FileNotFoundError("smile.jpg not found")
    if straight_face_emoji is None: raise FileNotFoundError("plain.png not found")
    if hands_up_emoji is None: raise FileNotFoundError("air.jpg not found")
    if touching_face_emoji is None: raise FileNotFoundError("touch.png not found")
    if scary_face_emoji is None: raise FileNotFoundError("scary.png not found")

    smiling_emoji = cv2.resize(smiling_emoji, EMOJI_WINDOW_SIZE)
    straight_face_emoji = cv2.resize(straight_face_emoji, EMOJI_WINDOW_SIZE)
    hands_up_emoji = cv2.resize(hands_up_emoji, EMOJI_WINDOW_SIZE)
    touching_face_emoji = cv2.resize(touching_face_emoji, EMOJI_WINDOW_SIZE)
    scary_face_emoji = cv2.resize(scary_face_emoji, EMOJI_WINDOW_SIZE)

except Exception as e:
    print("Erro ao carregar imagens:", e)
    exit()

blank_emoji = np.zeros((EMOJI_WINDOW_SIZE[1], EMOJI_WINDOW_SIZE[0], 3), dtype=np.uint8)

def eye_aspect_ratio(landmarks, ids):
    left, right, top, bottom = ids
    lx, ly = landmarks[left].x, landmarks[left].y
    rx, ry = landmarks[right].x, landmarks[right].y
    tx, ty = landmarks[top].x, landmarks[top].y
    bx, by = landmarks[bottom].x, landmarks[bottom].y
    vertical = np.hypot(tx - bx, ty - by)
    horizontal = np.hypot(lx - rx, ly - ry)
    return vertical / horizontal if horizontal != 0 else 0

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Erro ao abrir webcam.")
    exit()

cv2.namedWindow('Camera Feed', cv2.WINDOW_NORMAL)
cv2.namedWindow('Emoji Output', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Camera Feed', WINDOW_WIDTH, WINDOW_HEIGHT)
cv2.resizeWindow('Emoji Output', WINDOW_WIDTH, WINDOW_HEIGHT)

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose, \
     mp_face_mesh.FaceMesh(max_num_faces=1, min_detection_confidence=0.5) as face_mesh, \
     mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5) as hands:

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            continue

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False

        current_state = "STRAIGHT_FACE"

        pose_results = pose.process(rgb)
        if pose_results.pose_landmarks:
            lm = pose_results.pose_landmarks.landmark
            ls, rs = lm[mp_pose.PoseLandmark.LEFT_SHOULDER], lm[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            lw, rw = lm[mp_pose.PoseLandmark.LEFT_WRIST], lm[mp_pose.PoseLandmark.RIGHT_WRIST]

            if lw.y < ls.y or rw.y < rs.y:
                current_state = "HANDS_UP"

        face_results = face_mesh.process(rgb)
        if current_state != "HANDS_UP" and face_results.multi_face_landmarks:
            face = face_results.multi_face_landmarks[0]

            left_corner = face.landmark[291]
            right_corner = face.landmark[61]
            upper_lip = face.landmark[13]
            lower_lip = face.landmark[14]

            mouth_width = np.hypot(right_corner.x - left_corner.x,
                                   right_corner.y - left_corner.y)
            mouth_height = np.hypot(lower_lip.x - upper_lip.x,
                                    lower_lip.y - upper_lip.y)

            if mouth_width > 0:
                mar = mouth_height / mouth_width
                if mar > SMILE_THRESHOLD:
                    current_state = "SMILING"

        if current_state not in ["HANDS_UP", "SMILING"] and face_results.multi_face_landmarks:
            face = face_results.multi_face_landmarks[0].landmark

            left_eye_ids = [362, 263, 386, 374]  
            right_eye_ids = [33, 133, 159, 145]  

            left_EAR = eye_aspect_ratio(face, left_eye_ids)
            right_EAR = eye_aspect_ratio(face, right_eye_ids)

            avg_EAR = (left_EAR + right_EAR) / 2

            if avg_EAR > EYE_WIDE_THRESHOLD:
                current_state = "EYES_WIDE"

        if current_state not in ["HANDS_UP", "SMILING", "EYES_WIDE"] and face_results.multi_face_landmarks:
            face = face_results.multi_face_landmarks[0]
            hands_results = hands.process(rgb)

            if hands_results.multi_hand_landmarks:
                face_points = [(lm.x, lm.y) for lm in face.landmark]

                for hand in hands_results.multi_hand_landmarks:
                    for finger in [
                        mp_hands.HandLandmark.INDEX_FINGER_TIP,
                        mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
                        mp_hands.HandLandmark.RING_FINGER_TIP,
                        mp_hands.HandLandmark.THUMB_TIP
                    ]:

                        fx = hand.landmark[finger].x
                        fy = hand.landmark[finger].y

                        for (rx, ry) in face_points:
                            d = np.hypot(fx - rx, fy - ry)
                            if d < TOUCH_THRESHOLD:
                                current_state = "TOUCHING_FACE"
                                break
                        if current_state == "TOUCHING_FACE":
                            break

        if current_state == "SMILING":
            emoji = smiling_emoji
        elif current_state == "HANDS_UP":
            emoji = hands_up_emoji
        elif current_state == "TOUCHING_FACE":
            emoji = touching_face_emoji
        elif current_state == "EYES_WIDE":
            emoji = scary_face_emoji
        else:
            emoji = straight_face_emoji

        cam = cv2.resize(frame, (WINDOW_WIDTH, WINDOW_HEIGHT))
        cv2.putText(cam, current_state, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

        cv2.imshow('Camera Feed', cam)
        cv2.imshow('Emoji Output', emoji)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
