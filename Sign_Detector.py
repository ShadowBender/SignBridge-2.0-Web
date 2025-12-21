import cv2
import mediapipe as mp
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pickle
import os

# --- Direct Imports ---
from mediapipe.python.solutions import hands as mp_hands
from mediapipe.python.solutions import drawing_utils as mp_drawing
# ----------------------

class HandSignDetector:
    def __init__(self, data_file="hand_data.pkl"):
        self.mp_hands = mp_hands
        self.mp_draw = mp_drawing
        
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        self.data_file = data_file
        self.X = [] 
        self.y = [] 
        self.load_data()
        
        self.clf = KNeighborsClassifier(n_neighbors=3)
        self.is_trained = False
        if len(self.X) > 0:
            self.train_model()

    def process_frame(self, frame):
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)
        return results

    def extract_landmarks(self, results):
        """
        Extracts landmarks. 
        Force-limits data to exactly 84 features (2 hands).
        """
        data_point = []
        
        # 1. Collect all detected landmarks
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for lm in hand_landmarks.landmark:
                    data_point.append(lm.x)
                    data_point.append(lm.y)
        
        # 2. THE FIX: Strict truncation
        # If we have more than 2 hands (126+ features), CUT IT OFF at 84.
        REQUIRED_SIZE = 42 * 2  # 84
        if len(data_point) > REQUIRED_SIZE:
            data_point = data_point[:REQUIRED_SIZE]
            
        # 3. Padding
        # If we have less than 2 hands, fill with zeros
        while len(data_point) < REQUIRED_SIZE:
            data_point.append(0.0)
            
        return data_point

    def add_data(self, landmarks, label):
        self.X.append(landmarks)
        self.y.append(label)

    def save_data(self):
        with open(self.data_file, 'wb') as f:
            pickle.dump({'X': self.X, 'y': self.y}, f)
        print("Data saved successfully.")

    def load_data(self):
        if os.path.exists(self.data_file):
            with open(self.data_file, 'rb') as f:
                data = pickle.load(f)
                self.X = data['X']
                self.y = data['y']
            print(f"Loaded {len(self.X)} samples.")

    def train_model(self):
        if len(self.X) > 0:
            self.clf.fit(self.X, self.y)
            self.is_trained = True
            print("Model trained/updated.")

    def predict(self, landmarks):
        if not self.is_trained:
            return "Unknown", 0.0
        
        prediction = self.clf.predict([landmarks])[0]
        proba = self.clf.predict_proba([landmarks])
        confidence = np.max(proba) * 100
        return prediction, confidence