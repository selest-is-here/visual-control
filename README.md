# visual-control
A visual media control using finger gestures

A media controller using hand gestures in python

currently works with right hand, palm facing your default camera

requirements:
mediapipe == 0.10.35
hand_landmarker.task
opencv == cv2
pyautogui

a webcam

gestures:
fist = alt+f4
index and middle finger up = volume up
thumb and index up = volume down
index only up = media play/pause
thumb only open = previous track
pinky only open = next track
thumb, index, middle finger open = close the app itself
all fingers open = default recognized gesture to do nothing

note: there is a 0.3 second wait time before the next command executes to avoid spam/ unintentional over use
