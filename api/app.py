import os
import pickle
import cv2
import mediapipe as mp
import numpy as np
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware

# Initialize FastAPI app
app = FastAPI()


model_path = os.path.join(os.getcwd(), 'model_v4.p')
model_arabic_path = os.path.join(os.getcwd(), 'model_arabic-v1.p')


# CORS to allow React frontend to communicate with the backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Replace with your React frontend's address
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model
model_dict = pickle.load(open(model_path, 'rb'))
model = model_dict['model']

#Load arabic model
model_arabic_dict = pickle.load(open(model_arabic_path, 'rb'))
model_arabic = model_arabic_dict['model']

# Mediapipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Labels dictionary
labels_dict = {i: chr(65 + i) for i in range(26)}

dictionary = {
    0: "A",
    1: "B",
    2: "C",
    4: "E",
    6: "G",
    7: "H",
    8: "I",
    9: "J",
    10: "K",
    11: "L",
    14: "O",
    17: "R",
    18: "S",
    19: "T",
    20: "U",
    21: "V",
    22: "W",
    23: "X",
    24: "Y",
    30: "Thank you",
    31: "Sorry",
    32: "Please",
    26: "Hello",
    38: "I / Me",
    40: "Us",
    42: "My",
    46: "Do",
    47: "Go",
    48: "Busy",
    51: "Food"
}

arabic_dictionary = {
    0:"ع",
    1: "ال",
    2: "ا",
    3:"ب",
    4: "ض",
    5: "د",
    6: "ف",
    8: "ح",
    9: "ه",
    10: "ج",
    12: "خ",
    13: "لا",
    14: "ل",
    15: "م",
    16: "ن",
    17: "ق",
    18: "ر",
    19: "ص",
    20: "س",
    21: "ش",
    22: "ط",
    25: "ذ",
    26: "ث",
    28: "ي",
    30: "ز"
}


@app.get("/")
async def root():
    return {"message": "Server is running"}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        try:
            # Receive frame as bytes from the client
            data = await websocket.receive_bytes()

            # Convert the received bytes into an image
            np_img = np.frombuffer(data, np.uint8)
            frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

            # Process the frame with Mediapipe and extract landmarks
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)
            data_aux, x_, y_ = [], [], []

            H, W, _ = frame.shape

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y
                        x_.append(x)
                        y_.append(y)

                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y
                        data_aux.append(x - min(x_))
                        data_aux.append(y - min(y_))

                if len(data_aux) == model.n_features_in_:
                    prediction = model.predict([np.asarray(data_aux)])
                    predicted_character = dictionary.get(int(prediction[0]), "Unknown")

                    print(prediction)

                    # Send the prediction to the client
                    await websocket.send_text(predicted_character)

        except Exception as e:
            print(f"Error: {e}")
            break
    await websocket.close()


# Arabic Websocket Endpoint
@app.websocket("/ws/arabic")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        try:
            # Receive frame as bytes from the client
            data = await websocket.receive_bytes()

            # Convert the received bytes into an image
            np_img = np.frombuffer(data, np.uint8)
            frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

            # Process the frame with Mediapipe and extract landmarks
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)
            data_aux, x_, y_ = [], [], []

            H, W, _ = frame.shape

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y
                        x_.append(x)
                        y_.append(y)

                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y
                        data_aux.append(x - min(x_))
                        data_aux.append(y - min(y_))

                if len(data_aux) == model.n_features_in_:
                    prediction = model_arabic.predict([np.asarray(data_aux)])
                    predicted_character = arabic_dictionary.get(int(prediction[0]), "Unknown")

                    print(prediction)

                    # Send the prediction to the client
                    await websocket.send_text(predicted_character)

        except Exception as e:
            print(f"Error: {e}")
            break
    await websocket.close()

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)
