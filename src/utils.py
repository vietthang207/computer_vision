"""
This module contains functions used to manipulate images in OpenCV and PIL's Image.
"""
import cv2
import numpy as np
from PIL import Image
import numpy as np
import math


def conv2d(image, kernel, padding='same'):
    # return image
    image_height = image.shape[0]
    image_width = image.shape[1]
    kernel_height = kernel.shape[0]
    kernel_width = kernel.shape[1]
    offset_height = kernel_height // 2
    offset_width = kernel_width // 2
    padded_image = np.zeros(((image_height + offset_height * 2), (image_width + offset_width * 2)))
    output = np.zeros(((image_height + offset_height * 2), (image_width + offset_width * 2)))

    for i in range(image_height):
        for j in range(image_width):
            padded_image[i + offset_height][j + offset_width] = image[i][j]

    for x in range(image_height):
        for y in range(image_width):
            i = x + offset_height
            j = y + offset_width
            pi = padded_image[x:x+kernel_height, y:y+kernel_width]
            output[i][j] = np.sum(np.multiply(pi, kernel))

    return output[offset_height : offset_height + image_height, offset_width : offset_width + image_width]


def gaussWin(N, alpha=2.5):
    gw = np.zeros(N);
    for i in range(N):
        n = i - (N-1.0)/2
        arg = -1.0/2 * ((alpha * n/((N-1.0)/2)) ** 2)
        gw[i] = np.exp(arg)

    gw.shape = (N, 1)
    return gw


def sqr(x):
    return np.sum(np.multiply(x, x))


def gauss(x, mu, s):
    n = len(x)
    sigma = np.eye(n) * s
    if n == len(mu):
        det = np.linalg.det(sigma)
        if det == 0:
            raise NameError("Singular covariance matrix")

        norm_const = 1.0 / (math.pow((2*math.pi), n/2.0) * math.pow(det, 0.5))
        x_mu = x - mu
        inv = np.linalg.inv(sigma)
        result = math.pow(math.e, -0.5 * (np.dot(np.transpose(x_mu), np.dot(inv, x_mu))))
        return result * norm_const

if __name__ == "__main__":
    image = np.array([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5]])
    kernel = np.array([[1.0/9, 1.0/9, 1.0/9], [1.0/9, 1.0/9, 1.0/9], [1.0/9, 1.0/9, 1.0/9]])
    res = conv2d(image, kernel)
    print(res)


# if cv2.__version__ == '3.1.0':
#     from PIL import Image
# else:
#     import Image


faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def find_faces(image):
    faces_coordinates = _locate_faces(image)
    cutted_faces = [image[y:y + h, x:x + w] for (x, y, w, h) in faces_coordinates]
    normalized_faces = [_normalize_face(face) for face in cutted_faces]
    return zip(normalized_faces, faces_coordinates)

def _normalize_face(face):
    face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    face = cv2.resize(face, (350, 350))

    return face;

def _locate_faces(image):
    faces = faceCascade.detectMultiScale(
        image,
        scaleFactor=1.1,
        minNeighbors=15,
        minSize=(70, 70)
    )

    return faces  # list of (x, y, w, h)
def image_as_nparray(image):
    """
    Converts PIL's Image to numpy's array.
    :param image: PIL's Image object.
    :return: Numpy's array of the image.
    """
    return np.asarray(image)


def nparray_as_image(nparray, mode='RGB'):
    """
    Converts numpy's array of image to PIL's Image.
    :param nparray: Numpy's array of image.
    :param mode: Mode of the conversion. Defaults to 'RGB'.
    :return: PIL's Image containing the image.
    """
    return Image.fromarray(np.asarray(np.clip(nparray, 0, 255), dtype='uint8'), mode)


def load_image(source_path):
    """
    Loads RGB image and converts it to grayscale.
    :param source_path: Image's source path.
    :return: Image loaded from the path and converted to grayscale.
    """
    source_image = cv2.imread(source_path)
    return cv2.cvtColor(source_image, cv2.COLOR_BGR2GRAY)


def draw_with_alpha(source_image, image_to_draw, coordinates):
    """
    Draws a partially transparent image over another image.
    :param source_image: Image to draw over.
    :param image_to_draw: Image to draw.
    :param coordinates: Coordinates to draw an image at. Tuple of x, y, width and height.
    """
    x, y, w, h = coordinates
    image_to_draw = image_to_draw.resize((h, w), Image.ANTIALIAS)
    image_array = image_as_nparray(image_to_draw)
    for c in range(0, 3):
        source_image[y:y + h, x:x + w, c] = image_array[:, :, c] * (image_array[:, :, 3] / 255.0) \
                                            + source_image[y:y + h, x:x + w, c] * (1.0 - image_array[:, :, 3] / 255.0)
