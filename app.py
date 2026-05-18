# --- 0. THE ASYNC BRIDGE (MUST BE AT THE ABSOLUTE TOP) ---
import eventlet
eventlet.monkey_patch()

import sys
import os
import threading

# --- 1. PYTHON RUNTIME SAFEGUARD ---
if sys.version_info < (3, 10):
    print("\n❌ CRITICAL SYSTEM ERROR: SignBridge requires Python 3.10.11 or newer.")
    sys.exit(1)

# --- 2. ZERO-DEPENDENCY .ENV LOADER ---
if os.path.exists(".env"):
    with open(".env", "r") as f:
        for line in f:
            cleaned = line.strip()
            if cleaned and not cleaned.startswith("#") and "=" in cleaned:
                k, v = cleaned.split("=", 1)
                os.environ[k.strip()] = v.strip()

from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import cv2
import numpy as np
import base64
import time

# NEW SDK IMPORTS
from google import genai
from google.genai import types

from Sign_Detector import HandSignDetector

# --- 3. CORE BACKEND INITIALIZATION ---
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

detector = HandSignDetector()

current_sign = ""
last_sign = ""
hold_start_time = 0
CONFIRMATION_TIME = 1.0  

# --- 4. NEW GEMINI SDK SETUP ---
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
client = genai.Client(api_key=GEMINI_API_KEY)

# Lock the rules into the model's actual System Instructions
translation_rules = (
    "You are an expert Sign Language to English translator. "
    "You MUST rewrite broken words into a natural, grammatically perfect spoken English sentence. "
    "CRITICAL RULES:\n"
    "1. NEVER repeat the exact input. You MUST add missing verbs (are, is, am), articles (the, a), and punctuation.\n"
    "2. If input is 'How You', output MUST be 'How are you?'\n"
    "3. If input is 'Me Store Go', output MUST be 'I am going to the store.'\n"
    "4. Reply with ONLY the final polished sentence. No quotes, no explanations."
)

@app.route('/')
def index():
    return render_template('index.html')

# --- 5. COMPUTER VISION FRAMERATE PIPELINE ---
@socketio.on('video_frame')
def handle_video(data_image):
    global current_sign, last_sign, hold_start_time

    header, encoded = data_image.split(",", 1)
    binary_data = base64.b64decode(encoded)
    nparr = np.frombuffer(binary_data, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    frame = cv2.flip(frame, 1)
    
    results = detector.process_frame(frame)
    h, w, c = frame.shape
    
    detected_word = "..."
    confidence = 0

    if results.multi_hand_landmarks:
        x_min, y_min = w, h
        x_max, y_max = 0, 0

        for hand_landmarks in results.multi_hand_landmarks:
            detector.mp_draw.draw_landmarks(
                frame, hand_landmarks, detector.mp_hands.HAND_CONNECTIONS,
                detector.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                detector.mp_draw.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
            )
            for lm in hand_landmarks.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                if x < x_min: x_min = x
                if x > x_max: x_max = x
                if y < y_min: y_min = y
                if y > y_max: y_max = y

        x_min -= 20; y_min -= 20; x_max += 20; y_max += 20

        lm_list = detector.extract_landmarks(results)
        detected_word, confidence = detector.predict(lm_list)

        if detected_word != "Nothing" and confidence > 50:
            color = (255, 0, 255)
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
            cv2.rectangle(frame, (x_min, y_min - 45), (x_max, y_min), color, cv2.FILLED)
            cv2.putText(frame, f"{detected_word} {int(confidence)}%", (x_min + 5, y_min - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    if detected_word != "Unknown" and detected_word != "Nothing" and confidence > 60:
        if detected_word == last_sign:
            if (time.time() - hold_start_time) > CONFIRMATION_TIME:
                socketio.emit('new_word', {'word': detected_word})
                detected_word = "..." 
                last_sign = ""        
        else:
            last_sign = detected_word
            hold_start_time = time.time()
    else:
        last_sign = ""

    _, buffer = cv2.imencode('.jpg', frame)
    frame_encoded = base64.b64encode(buffer).decode('utf-8')
    emit('response_back', {'image': 'data:image/jpeg;base64,' + frame_encoded, 'status': detected_word})

# --- 6. INTELLIGENT GRAMMAR CONTEXTUALIZATION ENDPOINT ---
@socketio.on('request_grammar_correction')
def handle_grammar_correction(data):
    raw_sentence = data.get('raw_text', '').strip()
    if not raw_sentence: return

    print(f"\n[DEBUG] 1. Timer hit! Raw Tokens: '{raw_sentence}'")

    def background_worker(sentence):
        try:
            print(f"[DEBUG] 2. Asking Modern Gemini API to translate...")
            
            # The brand new syntax for the 2026 SDK update
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=f"Translate this sign language: {sentence}",
                config=types.GenerateContentConfig(
                    system_instruction=translation_rules,
                    temperature=0.0
                )
            )
            final_text = response.text.strip()
            
            print(f"[DEBUG] 3. Success! Sending to UI: '{final_text}'")
            socketio.emit('grammar_corrected', {'corrected_text': final_text, 'raw_text': sentence})
            
        except Exception as e:
            print(f"\n❌ [CRITICAL ERROR] {e}\n")
            socketio.emit('grammar_corrected', {'corrected_text': sentence, 'raw_text': sentence})

    socketio.start_background_task(background_worker, raw_sentence)

if __name__ == '__main__':
    socketio.run(app, debug=True)