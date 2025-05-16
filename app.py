import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import random
import os
from collections import deque
from pathlib import Path
from keras.layers import InputLayer
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Input
from design.layout import draw_header, draw_footer, draw_mode_selector
from design.styles import FONT
from main import decide_winner
from keras.mixed_precision import Policy
from keras_facenet import FaceNet
import pickle
from player_stats import update_player_stats, get_player_level_message, player_stats

IMG_SIZE = 224
LABELS = ["rock", "paper", "scissors"]
GLOBAL_MODEL_PATH = "global_dqn_model.h5"
GESTURE_MODEL_PATH = "rps_model_v1.h5"
STABLE_GESTURE_FRAMES = 5
CONF_THRESHOLD = 0.5
STATE_HISTORY = 3
FACE_DB_PATH = "face_db.pkl"

experience_buffer = deque(maxlen=2000)

class PatchedInputLayer(InputLayer):
    def __init__(self, *args, **kwargs):
        if "batch_shape" in kwargs and "batch_input_shape" not in kwargs:
            kwargs["batch_input_shape"] = kwargs.pop("batch_shape")
        super().__init__(*args, **kwargs)

def create_dqn_model():
    model = Sequential([
        Input(shape=(STATE_HISTORY * 3,)),
        Dense(64, activation='relu'),
        Dense(64, activation='relu'),
        Dense(3)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

dqn_model = load_model(GLOBAL_MODEL_PATH) if os.path.exists(GLOBAL_MODEL_PATH) else create_dqn_model()
gesture_model = tf.keras.models.load_model(
    GESTURE_MODEL_PATH,
    compile=False,
    custom_objects={"InputLayer": PatchedInputLayer, "DTypePolicy": Policy}
)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
drawer = mp.solutions.drawing_utils

embedder = FaceNet()
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def load_face_db():
    if os.path.exists(FACE_DB_PATH):
        with open(FACE_DB_PATH, "rb") as f:
            data = pickle.load(f)
            return data["names"], data["embeddings"]
    else:
        return [], []

def save_face_db(names, embeddings):
    with open(FACE_DB_PATH, "wb") as f:
        pickle.dump({"names": names, "embeddings": embeddings}, f)

known_names, known_embeddings = load_face_db()

def detect_and_crop_face(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda box: box[2]*box[3])
    face_img = frame[y:y+h, x:x+w]
    face_img = cv2.resize(face_img, (160, 160))
    return face_img

def get_face_embedding(face_img):
    rgb_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    embedding = embedder.embeddings([rgb_img])[0]
    return embedding

def register_face(name, face_img):
    embedding = get_face_embedding(face_img)
    known_names.append(name)
    known_embeddings.append(embedding)
    save_face_db(known_names, known_embeddings)

def recognize_face(face_img, threshold=0.7):
    embedding = get_face_embedding(face_img)
    if not known_embeddings:
        return None
    distances = [np.linalg.norm(embedding - db_emb) for db_emb in known_embeddings]
    min_dist = min(distances)
    if min_dist < threshold:
        return known_names[np.argmin(distances)]
    return None

def train_online(model, buffer, batch_size=32, gamma=0.95):
    if len(buffer) < batch_size:
        return
    batch = random.sample(buffer, batch_size)
    X, y = [], []
    for state, action, reward in batch:
        q_vals = model.predict(np.array([state]), verbose=0)[0]
        q_vals[action] = reward
        X.append(state)
        y.append(q_vals)
    model.fit(np.array(X), np.array(y), epochs=1, verbose=0)

def draw_mode_selector(canvas, norm, dev, exp=None, label=None):
    cv2.rectangle(canvas, norm[:2], norm[2:], (50, 255, 50), -1)
    cv2.putText(canvas, "Normal Mode", (norm[0]+10, norm[1]+80), FONT, 2, (0,0,0), 5)
    cv2.rectangle(canvas, dev[:2], dev[2:], (50, 180, 255), -1)
    cv2.putText(canvas, "Developer Mode", (dev[0]+10, dev[1]+80), FONT, 2, (0,0,0), 5)
    if exp is not None:
        cv2.rectangle(canvas, exp[:2], exp[2:], (70, 70, 255), -1)
        cv2.putText(canvas, "Experience Mode", (exp[0]+10, exp[1]+80), FONT, 2, (255,255,255), 5)
    if label is not None:
        cv2.rectangle(canvas, label[:2], label[2:], (220, 50, 220), -1)
        cv2.putText(canvas, "Labeling Mode", (label[0]+10, label[1]+80), FONT, 2, (255,255,255), 5)

def select_mode():
    W, H = 1280, 960
    norm = (100, 300, 400, 500)
    dev = (500, 300, 800, 500)
    exp = (900, 300, 1200, 500)
    label = (100, 600, 400, 800)
    mode = None
    cv2.namedWindow("Select Mode")
    def cb(e, x, y, *_):
        nonlocal mode
        if e == cv2.EVENT_LBUTTONDOWN:
            if norm[0] <= x <= norm[2] and norm[1] <= y <= norm[3]: mode = "normal"
            if dev[0] <= x <= dev[2] and dev[1] <= y <= dev[3]: mode = "developer"
            if exp[0] <= x <= exp[2] and exp[1] <= y <= exp[3]: mode = "experience"
            if label[0] <= x <= label[2] and label[1] <= y <= label[3]: mode = "labeling"
    cv2.setMouseCallback("Select Mode", cb)
    while mode is None:
        canvas = np.zeros((H, W, 3), np.uint8)
        draw_mode_selector(canvas, norm, dev, exp, label)
        cv2.imshow("Select Mode", canvas)
        if cv2.waitKey(10) & 0xFF == ord('q'): mode = "normal"
    cv2.destroyWindow("Select Mode")
    return mode

MODE = select_mode()
EXPERIENCE_MODE = MODE == "experience"
DEVELOPER_MODE = MODE == "developer"
DEV_MODE = DEVELOPER_MODE
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)
cv2.namedWindow("RPS")
gesture_hist = deque(maxlen=STABLE_GESTURE_FRAMES)
last_moves = deque(maxlen=STATE_HISTORY)
current_player = None
status_message = ""

persistent_prediction = {
    "player": "",
    "ai": "",
    "outcome": "",
    "conf": None
}

ai_self_training_state = {
    "rounds": 0,
    "wins": 0,
    "losses": 0,
    "draws": 0
}

import datetime
DATASET_DIR = "gesture_dataset"
for l in LABELS:
    os.makedirs(os.path.join(DATASET_DIR, l), exist_ok=True)
ai_self_training_state = {
    "rounds": 0,
    "wins": 0,
    "losses": 0,
    "draws": 0
}

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    frame = cv2.flip(frame, 1)
    face_img = detect_and_crop_face(frame)
    live_face_message = ""
    player_level_msg = ""
    if face_img is not None:
        user = recognize_face(face_img)
        if user:
            live_face_message = f"Face detected: {user}"
            if current_player is None:
                current_player = user
                status_message = f"👋 Welcome back, {user}!"
                player_level_msg = get_player_level_message(user)
        else:
            live_face_message = "Unknown face. Press 'n' to register."
            if current_player is None:
                status_message = "👤 New face detected! Press 'n' to register."
            if cv2.waitKey(1) & 0xFF == ord('n'):
                cv2.putText(frame, "Type your name in terminal.", (10, 100), FONT, 0.8, (0, 255, 255), 2)
                cv2.imshow("RPS", frame)
                print("Please enter your name in terminal:")
                name = input("Enter your name: ")
                register_face(name, face_img)
                current_player = name
                status_message = f"👋 Registered and logged in as {name}!"
                player_level_msg = get_player_level_message(name)
    else:
        live_face_message = "No face detected"
        player_level_msg = ""
    h, w, _ = frame.shape
    frame_small = cv2.resize(frame, (320, 240))
    res = hands.process(cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB))
    found_gesture = False
    if res.multi_hand_landmarks:
        for lm in res.multi_hand_landmarks:
            drawer.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)
            xs = [p.x for p in lm.landmark]
            ys = [p.y for p in lm.landmark]
            x1, y1 = int(min(xs) * w), int(min(ys) * h)
            x2, y2 = int(max(xs) * w), int(max(ys) * h)
            roi = frame[y1:y2, x1:x2]
            if roi.size == 0: continue
            img = cv2.resize(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB), (IMG_SIZE, IMG_SIZE)) / 255.0
            pred = gesture_model.predict(np.array([img]), verbose=0)[0]
            if pred.max() < CONF_THRESHOLD: continue
            gesture = LABELS[int(pred.argmax())]
            gesture_hist.append(gesture)
            if len(gesture_hist) == STABLE_GESTURE_FRAMES and len(set(gesture_hist)) == 1:
                stable = gesture_hist[-1]; gesture_hist.clear()
                ai_input = [0] * STATE_HISTORY * 3
                last_moves.append(LABELS.index(stable))
                for i, move in enumerate(last_moves):
                    ai_input[i * 3 + move] = 1
                ai_q = dqn_model.predict(np.array([ai_input]), verbose=0)[0]
                ai_idx = int(np.argmax(ai_q))
                ai = LABELS[ai_idx]
                outcome = decide_winner(stable, ai)
                reward = {"Player Wins": -1, "Draw": 0, "Computer Wins": 1}[outcome]
                experience_buffer.append((ai_input, ai_idx, reward))
                train_online(dqn_model, experience_buffer)
                persistent_prediction = {
                    "player": stable,
                    "ai": ai,
                    "outcome": outcome,
                    "conf": pred.max()
                }
                if current_player:
                    update_player_stats(current_player, stable, ai, outcome)
                    player_level_msg = get_player_level_message(current_player)
                found_gesture = True
                if current_player:
                    update_player_stats(current_player, stable, ai, outcome)
                    player_level_msg = get_player_level_message(current_player)
    if EXPERIENCE_MODE:
        try:
            rounds_to_play = int(input("Enter number of rounds for AI self-training (default 200): ") or 200)
        except:
            rounds_to_play = 200
        try:
            max_seconds = float(input("Enter max duration in seconds for self-training (default 10): ") or 10)
        except:
            max_seconds = 10
        import time
        t0 = time.time()
        ai_self_training_state["rounds"] = 0
        ai_self_training_state["wins"] = 0
        ai_self_training_state["losses"] = 0
        ai_self_training_state["draws"] = 0
        while ai_self_training_state["rounds"] < rounds_to_play and (time.time() - t0) < max_seconds:
            simulated_player_idx = random.randint(0, 2)
            simulated_player_gesture = LABELS[simulated_player_idx]
            ai_input = [0] * STATE_HISTORY * 3
            last_moves.append(simulated_player_idx)
            for i, move in enumerate(last_moves):
                ai_input[i * 3 + move] = 1
            ai_q = dqn_model.predict(np.array([ai_input]), verbose=0)[0]
            ai_idx = int(np.argmax(ai_q))
            ai_gesture = LABELS[ai_idx]
            outcome = decide_winner(simulated_player_gesture, ai_gesture)
            reward = {"Player Wins": -1, "Draw": 0, "Computer Wins": 1}[outcome]
            experience_buffer.append((ai_input, ai_idx, reward))
            train_online(dqn_model, experience_buffer)
            ai_self_training_state["rounds"] += 1
            if outcome == "Player Wins": ai_self_training_state["wins"] += 1
            elif outcome == "Computer Wins": ai_self_training_state["losses"] += 1
            else: ai_self_training_state["draws"] += 1
            show_text = f"Experience Mode: AI Self-Training {ai_self_training_state['rounds']} / {rounds_to_play} " \
                        f"| W:{ai_self_training_state['wins']} L:{ai_self_training_state['losses']} D:{ai_self_training_state['draws']}" \
                        f" | {time.time()-t0:.1f}s/{max_seconds}s"
            train_img = np.zeros_like(frame)
            cv2.putText(train_img, show_text, (120, 400), FONT, 2.0, (100, 255, 255), 8)
            cv2.imshow("RPS", train_img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        continue
    if MODE == "labeling":
        label_message = "Press R, P, S to save this gesture | Press Q to quit"
        while True:
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.flip(frame, 1)
            roi_img = frame
            h, w, _ = frame.shape
            cv2.putText(frame, label_message, (50, 60), FONT, 1.4, (200, 50, 255), 4)
            cv2.imshow("RPS", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key in [ord('r'), ord('p'), ord('s')]:
                gesture = {ord('r'): "rock", ord('p'): "paper", ord('s'): "scissors"}[key]
                now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                save_path = os.path.join(DATASET_DIR, gesture, f"{now}.jpg")
                cv2.imwrite(save_path, roi_img)
                cv2.putText(frame, f"Saved as {gesture.upper()}", (50, 120), FONT, 1.4, (50, 255, 50), 4)
                cv2.imshow("RPS", frame)
                cv2.waitKey(400)
        continue
    draw_footer(frame, True, 0)
    draw_header(
        frame,
        persistent_prediction["player"] if persistent_prediction["player"] else "Waiting...",
        persistent_prediction["ai"] if persistent_prediction["ai"] else "",
        persistent_prediction["outcome"] if persistent_prediction["outcome"] else "",
        conf=persistent_prediction["conf"] if persistent_prediction["conf"] else 0,
        dev=DEV_MODE
    )
    cv2.putText(frame, status_message, (800, 60), FONT, 1.3, (0, 255, 255), 3)
    cv2.putText(frame, live_face_message, (800, 120), FONT, 1.3, (0, 150, 255), 3)
    cv2.putText(frame, player_level_msg, (800, 180), FONT, 1.3, (0, 255, 180), 3)
    cv2.imshow("RPS", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
dqn_model.save(GLOBAL_MODEL_PATH)
