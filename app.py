import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import random
import os
from collections import deque
from pathlib import Path
from keras import mixed_precision
from keras.layers import InputLayer
from design.layout import draw_header, draw_footer, draw_mode_selector
from design.styles import FONT
from main import decide_winner

# ───────────── CONFIG ───────────── #
IMG_SIZE = 224
LABELS = ["rock", "paper", "scissors"]
MODEL_PATH = r"C:\Users\asult\PycharmProjects\flask\rps_model_v1.h5"
Q_PATH = "qtable.npy"
FEEDBACK_DIR = Path("feedback")
FEEDBACK_DIR.mkdir(exist_ok=True)

STABLE_GESTURE_FRAMES = 5
MOTION_STABLE_REQUIRED = 5
MOTION_THRESHOLD_PIXELS = 15
CONF_THRESHOLD = 0.5

# ───────────── MODEL SETUP ───────────── #
class PatchedInputLayer(InputLayer):
    def __init__(self, *a, **k):
        if "batch_shape" in k and "batch_input_shape" not in k:
            k["batch_input_shape"] = k.pop("batch_shape")
        super().__init__(*a, **k)

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False,
    custom_objects={"InputLayer": PatchedInputLayer, "DTypePolicy": mixed_precision.Policy}
)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

# ───────────── MEDIAPIPE SETUP ───────────── #
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
drawer = mp.solutions.drawing_utils

# ───────────── RL SETUP ───────────── #
wins_set = {("rock", "scissors"), ("paper", "rock"), ("scissors", "paper")}
reward_table = {"Computer Wins": 1, "Draw": 0, "Player Wins": -1}
Q = np.load(Q_PATH) if os.path.exists(Q_PATH) else np.zeros((3, 3), np.float32)
alpha, gamma = 0.1, 0.9
epsilon = 0.25
learning = False
last_state = last_action = None
player_wins = 0
comp_wins = 0
total_games = {"Player Wins": 0, "Computer Wins": 0, "Draw": 0}

# ───────────── UI MODE ───────────── #
def select_mode():
    W, H = 640, 480
    norm = (50, 150, 290, 310)
    dev = (350, 150, 590, 310)
    mode = None
    cv2.namedWindow("Select Mode")

    def cb(e, x, y, *_):
        nonlocal mode
        if e == cv2.EVENT_LBUTTONDOWN:
            if norm[0] <= x <= norm[2] and norm[1] <= y <= norm[3]: mode = False
            if dev[0] <= x <= dev[2] and dev[1] <= y <= dev[3]: mode = True

    cv2.setMouseCallback("Select Mode", cb)
    while mode is None:
        canvas = np.zeros((H, W, 3), np.uint8)
        draw_mode_selector(canvas, norm, dev)
        cv2.imshow("Select Mode", canvas)
        if cv2.waitKey(10) & 0xFF == ord('q'): mode = False
    cv2.destroyWindow("Select Mode")
    return mode

# ───────────── HAND CALIBRATION ───────────── #
def hand_calibration_screen():
    guide = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(guide, "Place your hand in the box", (120, 200), FONT, 0.9, (255, 255, 255), 2)
    cv2.rectangle(guide, (220, 140), (420, 340), (100, 255, 100), 2)
    for _ in range(80):
        cv2.imshow("Calibration", guide)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    cv2.destroyWindow("Calibration")

DEV_MODE = select_mode()
learning = DEV_MODE
hand_calibration_screen()

# ───────────── MAIN LOOP ───────────── #
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cv2.namedWindow("RPS")
gesture_hist = deque(maxlen=STABLE_GESTURE_FRAMES)
motion_hist = deque(maxlen=MOTION_STABLE_REQUIRED)
prev_center = None
show_update = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    res = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cur_conf = None
    stable = None

    if res.multi_hand_landmarks:
        for lm in res.multi_hand_landmarks:
            drawer.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)
            xs = [p.x for p in lm.landmark]
            ys = [p.y for p in lm.landmark]
            x1, y1 = max(int(min(xs) * w) - 20, 0), max(int(min(ys) * h) - 20, 0)
            x2, y2 = min(int(max(xs) * w) + 20, w), min(int(max(ys) * h) + 20, h)
            roi = frame[y1:y2, x1:x2]
            if roi.size == 0:
                continue

            center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
            if prev_center:
                dx = abs(center[0] - prev_center[0])
                dy = abs(center[1] - prev_center[1])
                motion_hist.append(dx < MOTION_THRESHOLD_PIXELS and dy < MOTION_THRESHOLD_PIXELS)
            else:
                motion_hist.append(False)
            prev_center = center

            thumb = cv2.resize(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB), (100, 100))
            thumb = cv2.cvtColor(thumb, cv2.COLOR_RGB2BGR)
            frame[10:110, w - 110:w - 10] = thumb
            cv2.rectangle(frame, (w - 110, 10), (w - 10, 110), (255, 255, 255), 1)

            if not all(motion_hist):
                continue

            img = cv2.resize(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB), (IMG_SIZE, IMG_SIZE)) / 255.0
            pred = model.predict(img[None, ...], verbose=0)[0]
            if pred.max() < CONF_THRESHOLD:
                continue

            gesture = LABELS[int(pred.argmax())]
            cur_conf = float(pred.max())
            gesture_hist.append(gesture)
            cv2.putText(frame, gesture, (w - 110, 130), FONT, 0.6, (200, 255, 255), 2)

    if len(gesture_hist) == STABLE_GESTURE_FRAMES and len(set(gesture_hist)) == 1:
        stable = gesture_hist[-1]
        gesture_hist.clear()
        s_idx = LABELS.index(stable)

        if last_state is not None and last_action is not None and learning:
            reward = reward_table[decide_winner(LABELS[last_state], LABELS[last_action])]
            Q[last_state, last_action] += alpha * (reward + gamma * Q[s_idx].max() - Q[last_state, last_action])

        ai = LABELS[int(Q[s_idx].argmax())] if learning else random.choice(LABELS)
        a_idx = LABELS.index(ai)
        last_state, last_action = s_idx, a_idx
        outcome = decide_winner(stable, ai)
        total_games[outcome] += 1

        # Dynamic Difficulty
        if outcome == "Player Wins":
            player_wins += 1
        elif outcome == "Computer Wins":
            comp_wins += 1

        if player_wins - comp_wins >= 3:
            epsilon = max(0.05, epsilon - 0.05)
        elif comp_wins - player_wins >= 3:
            epsilon = min(0.5, epsilon + 0.05)

        draw_header(frame, stable, ai, outcome, conf=cur_conf, dev=DEV_MODE)

    draw_footer(frame, learning, show_update)
    cv2.putText(frame, f"P:{total_games['Player Wins']} C:{total_games['Computer Wins']} D:{total_games['Draw']}",
                (10, h - 70), FONT, 0.6, (255, 255, 255), 1)
    cv2.imshow("RPS", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    if key == ord('r'):
        gesture_hist.clear()
        last_state = last_action = None
        player_wins = comp_wins = 0
    if key == ord('l'):
        learning = not learning

cap.release()
cv2.destroyAllWindows()
np.save(Q_PATH, Q)
if DEV_MODE:
    model.save("rps_model_updated.h5")
