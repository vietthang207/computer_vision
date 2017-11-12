from cv2 import WINDOW_NORMAL

import cv2
from utils import nparray_as_image, draw_with_alpha, find_faces, detect_face

import pickle
import glob
from classi import get_data
import numpy as np
from cartoon_filter import cartoonize

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

def _load_emoticons(emotions):
    return [nparray_as_image(cv2.imread('emoji/%s.png' % emotion, -1), mode=None) for emotion in emotions]


def show(model, emoticons):
    files = glob.glob("predict_dataset/*")
    for file in files:
        print(file)
        image = cv2.imread(file)
        print(image.shape)
        normalized_face, (x, y, w, h) = detect_face(image)
        # for normalized_face, (x, y, w, h) in detect_face(image):
        print(normalized_face.shape)
        # cv2.imshow("a", normalized_face)
        # cv2.waitKey(5000)
        print(image.shape)
        # normalized_face = cv2.resize(normalized_face, (268, 268))
        # gray = cv2.cvtColor(normalized_face, cv2.COLOR_BGR2GRAY) #convert to grayscale
        clahe_image = clahe.apply(normalized_face)
        normalized_face = get_data(clahe_image)
        p = []
        p.append(normalized_face)
        prediction = model.predict(np.array(p))  # do prediction
        print(prediction.shape )
        print(prediction)
        prediction = prediction[0]
        image_to_draw = emoticons[prediction]
        cv2.imshow('HIEU', image)
        cv2.waitKey(500)
        # cartoonize(image)
        draw_with_alpha(image, image_to_draw, (x, y, w, h))
        cv2.imshow('HIEU', image)
        cv2.waitKey(500)


if __name__ == '__main__':
    # emotions = ['neutral', 'anger', 'disgust', 'happy', 'surprise']
    emotions = ["anger", "contempt", "disgust", "fear", "happy", "neutral", "sadness", "surprise"] #Emotion list

    emoticons = _load_emoticons(emotions)

    # use learnt model
    filename = 'finalized_model.sav'
    loaded_model = pickle.load(open(filename, 'rb'))
    show(loaded_model, emoticons)