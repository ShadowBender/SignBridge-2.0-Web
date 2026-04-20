from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import cv2
import numpy as np
import base64
import time
from Sign_Detector import HandSignDetector

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# Load your AI
detector = HandSignDetector()

# State variables
current_sign = ""
last_sign = ""
hold_start_time = 0
CONFIRMATION_TIME = 2.0  

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('video_frame')
def handle_video(data_image):
    global current_sign, last_sign, hold_start_time

    # 1. Decode the image
    header, encoded = data_image.split(",", 1)
    binary_data = base64.b64decode(encoded)
    nparr = np.frombuffer(binary_data, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # 2. FLIP THE FRAME (Mirror Effect)
    frame = cv2.flip(frame, 1)

    # 3. Process the frame
    results = detector.process_frame(frame)
    h, w, c = frame.shape
    
    detected_word = "..."
    confidence = 0

    if results.multi_hand_landmarks:
        # Calculate Bounding Box for ALL hands
        x_min, y_min = w, h
        x_max, y_max = 0, 0

        for hand_landmarks in results.multi_hand_landmarks:
            # Draw Skeleton
            detector.mp_draw.draw_landmarks(
                frame, hand_landmarks, detector.mp_hands.HAND_CONNECTIONS,
                detector.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                detector.mp_draw.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
            )

            # Update Box Coordinates
            for lm in hand_landmarks.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                if x < x_min: x_min = x
                if x > x_max: x_max = x
                if y < y_min: y_min = y
                if y > y_max: y_max = y

        # Add padding to the box
        x_min -= 20; y_min -= 20; x_max += 20; y_max += 20

        # Predict Sign
        lm_list = detector.extract_landmarks(results)
        detected_word, confidence = detector.predict(lm_list)

        # --- DRAWING THE BOX AND PERCENTAGE ---
        if detected_word != "Nothing" and confidence > 50:
            color = (255, 0, 255) # Purple neon
            
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
            cv2.rectangle(frame, (x_min, y_min - 45), (x_max, y_min), color, cv2.FILLED)
            
            label_text = f"{detected_word} {int(confidence)}%"
            cv2.putText(frame, label_text, (x_min + 5, y_min - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # 4. HOLD LOGIC (2 Seconds)
    if detected_word != "Unknown" and detected_word != "Nothing" and confidence > 60:
        if detected_word == last_sign:
            if (time.time() - hold_start_time) > CONFIRMATION_TIME:
                
                # TIMEOUT REACHED! Send word to frontend
                socketio.emit('new_word', {'word': detected_word})
                
                detected_word = "..." 
                last_sign = ""        
        else:
            last_sign = detected_word
            hold_start_time = time.time()
    else:
        last_sign = ""

    # 5. Encode and Send Back
    _, buffer = cv2.imencode('.jpg', frame)
    frame_encoded = base64.b64encode(buffer).decode('utf-8')
    
    emit('response_back', {'image': 'data:image/jpeg;base64,' + frame_encoded, 'status': detected_word})


if __name__ == '__main__':
    socketio.run(app, debug=True)