import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import pyautogui as pya
import time

def fingers_up(hand_landmarks):
    tips = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky  4 8 12 16 20
    joint = [3, 6, 10, 14, 18]  # Joint below tip (to compare y-coordinate)    3 6 10 14 18

    fingers = []

    thumb_tip_x = hand_landmarks[tips[0]].x  #compare thumb tip to joint by x to determine if its open
    thumb_joint_x = hand_landmarks[joint[0]].x
    if thumb_tip_x < thumb_joint_x:
        fingers.append(1)
    else:
        fingers.append(0)

    for i in range(1, 5):
        tip_y = hand_landmarks[tips[i]].y
        joint_y = hand_landmarks[joint[i]].y
        if tip_y < joint_y:
            fingers.append(1)
        else:
            fingers.append(0)
    return fingers

def detect_gesture(fingers):
    if fingers == [0, 0, 0, 0, 0]:
        pya.hotkey("alt", "f4")
        return "DIE"
    elif fingers == [1, 1, 1, 1, 1]:
        return "HALT"
    elif fingers == [0, 1, 1, 0, 0]:
        pya.press("volumeup")
        return "LOUDER"
    elif fingers == [1, 1, 0, 0, 0]:
        pya.press("volumedown")
        return "SILENCE"
    elif fingers == [0, 1, 0, 0, 0]:
        pya.press("playpause")
        return "SPEAK/MUTE"
    elif fingers == [1, 0, 0, 0, 0]:
        pya.press("prevtrack")
        return "FALL BACK"
    elif fingers == [0, 0, 0, 0, 1]:
        pya.press("nexttrack")
        return "COME FORTH"
    elif fingers == [1, 1, 1, 0, 0]:
        pya.press("num0")
        return "TERMINATE"
    else:
        return "unknown gesture {}".format(fingers)


base_options=python.BaseOptions(model_asset_path="hand_landmarker.task")

options=vision.HandLandmarkerOptions(base_options=base_options,
                                     running_mode=vision.RunningMode.IMAGE,
                                     min_tracking_confidence=0.1,
                                     min_hand_detection_confidence=0.1,
                                     min_hand_presence_confidence=0.1,
                                     num_hands=1)

tracker=vision.HandLandmarker.create_from_options(options)

cam=cv2.VideoCapture(0)

last_execution=0

while True:
    cool_down = 0.1
    current_time = time.time()
    if current_time - last_execution < cool_down:
        continue
    isCamOn , frame = cam.read()
    if not isCamOn:
        break
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image=mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result=tracker.detect(image)
    if result.hand_landmarks:
        h, w, _ = frame.shape
        for hand_landmarks in result.hand_landmarks:
            for landmark in hand_landmarks:
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                cv2.circle(frame, (x, y), 4, (0, 0, 0), 2)

            fingers = fingers_up(hand_landmarks)
            gesture = detect_gesture(fingers)
            last_execution=time.time()

            cv2.putText(frame, gesture, (50, 50), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 0), 2)

    cv2.imshow("MVC", frame)
    if cv2.waitKey(1) & 0xFF == ord('0'):
        break

cam.release()
cv2.destroyAllWindows()