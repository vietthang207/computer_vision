from cv2 import WINDOW_NORMAL

import cv2
from utils import nparray_as_image, draw_with_alpha, find_faces

import pickle
import glob
from classi import get_data
import numpy as np
from cartoon_filter import cartoonize

def _load_emoticons(emotions):
    return [nparray_as_image(cv2.imread('graphics/%s.png' % emotion, -1), mode=None) for emotion in emotions]


def show_webcam_and_run(model, emoticons):
    files = glob.glob("predict_dataset/*")
    for file in files:
        image = cv2.imread(file)
        for normalized_face, (x, y, w, h) in find_faces(image):
            normalized_face = cv2.resize(normalized_face, (268, 268))
            normalized_face = get_data(normalized_face)
            p = []
            p.append(normalized_face)
            prediction = model.predict(np.array(p))  # do prediction
            print(prediction.shape )
            prediction = prediction[0]
            image_to_draw = emoticons[prediction]
            cv2.imshow('HIEU', image)
            cv2.waitKey(2000)
            # cartoonize(image)
            draw_with_alpha(image, image_to_draw, (x, y, w, h))
            cv2.imshow('HIEU', image)
            cv2.waitKey(2000)


if __name__ == '__main__':
    emotions = ['neutral', 'anger', 'disgust', 'happy', 'sadness', 'surprise']
    emoticons = _load_emoticons(emotions)

    # use learnt model
    filename = 'finalized_model.sav'
    loaded_model = pickle.load(open(filename, 'rb'))
    show_webcam_and_run(loaded_model, emoticons)