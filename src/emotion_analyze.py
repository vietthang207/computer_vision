from cv2 import WINDOW_NORMAL

import cv2
from emotion_utils import *

import pickle
import glob
from classi import get_data
import numpy as np
from cartoon_filter import cartoonize

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

def predict_emotion(image):
     # use learnt model
    filename = 'finalized_model.sav'
    model = pickle.load(open(filename, 'rb'))

    normalized_face, (x, y, w, h) = detect_face(image)
    clahe_image = clahe.apply(normalized_face)
    normalized_face = get_data(clahe_image)
    p = []
    p.append(normalized_face)
    prediction = model.predict(np.array(p))  # do prediction
    prediction = prediction[0]
    return prediction


def video():
    emotions = ["anger", "contempt", "disgust", "fear", "happy", "neutral", "sadness", "surprise"] #Emotion list

    video = cv2.VideoCapture('../videos/emotion.mp4')
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    writer = cv2.VideoWriter('../videos/output_face.avi', fourcc, 20.0, (width, height))

    num_frame = 0
    while True:
        ret, frame = video.read()
        if ret == True:
            num_frame += 1
            print('frame: ' + str(num_frame))

            prediction = predict_emotion(frame)
            mode = emotions[prediction]
            output = cartoonize(frame, mode)
            if output is not None:
                writer.write(output)
            else:
                writer.write(frame)
        else:
            break

    video.release()
    writer.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    video()
