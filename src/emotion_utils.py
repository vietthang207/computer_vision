import cv2
import numpy as np
from PIL import Image
import numpy as np
import math

def nparray_as_image(nparray, mode='RGB'):
    """
    Converts numpy's array of image to PIL's Image.
    :param nparray: Numpy's array of image.
    :param mode: Mode of the conversion. Defaults to 'RGB'.
    :return: PIL's Image containing the image.
    """
    return Image.fromarray(np.asarray(np.clip(nparray, 0, 255), dtype='uint8'), mode)

faceDet = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
faceDet2 = cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")
faceDet3 = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
faceDet4 = cv2.CascadeClassifier("haarcascade_frontalface_alt_tree.xml")

def detect_face(frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #Convert image to grayscale

        #Detect face using 4 different classifiers
        face = faceDet.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
        face2 = faceDet2.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
        face3 = faceDet3.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
        face4 = faceDet4.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)

        #Go over detected faces, stop at first detected face, return empty if no face.
        if len(face) == 1:
            facefeatures = face
        elif len(face2) == 1:
            facefeatures = face2
        elif len(face3) == 1:
            facefeatures = face3
        elif len(face4) == 1:
            facefeatures = face4
        else:
            facefeatures = None

        if facefeatures is None:
            return None, None
        #Cut and save face
        for (x, y, w, h) in facefeatures: #get coordinates and size of rectangle containing face
            # print ("face found in file: %s" %f)
            gray = gray[y:y+h, x:x+w] #Cut the frame to size

            try:
                out = cv2.resize(gray, (350, 350)) #Resize face so all images have same size
                return (out, (x, y, w, h))
            except:
               pass #If error, pass file