import cv2
import numpy as np
from utils import *
import corner_detector as cd
import edge_detector as ed
import time

class CartoonFilter(object):
    """Cartoon filter for image"""
    def __init__(self, ratio, edge_threshold, edge_thickness):
        self.ratio = ratio
        self.edge_threshold = edge_threshold
        self.edge_thickness = edge_thickness
        self.intensity = [1,1,1]

    def cartoon_color(self, img, bitmap, edgemap):
        # Actual full step for cartoon effect:

        # 1: Apply a bilateral filter to reduce the color palette of the image.
        # 2: Convert the original color image to grayscale.
        # 3: Apply a median blur to reduce image noise in the grayscale image.
        # 4: Create an edge mask from the grayscale image using adaptive thresholding.
        # 5: Combine the color image from step 1 with the edge mask from step 4.

        # 2,3,4 should already be done to do edge detection. 1,5 is in here. just pass the edgemap + blurred colored image here

        height = img.shape[0]
        width = img.shape[1]
        edgemap = self.edgemap_modify(edgemap)

        for i in range(1, height):
            for j in range(1, width):
                if edgemap[i,j]:
                    img[i,j] = [0,0,0]
                elif bitmap[i][j] == 1:
                    img[i][j] = self.color_change(img[i][j])
        return img


    def edgemap_modify(self, edgemap):
        # increase the spread of the edge map SQUARE_WISE by twice the amount of thickness
        # ugly, best if thickness <3

        thicc = self.edge_thickness
        thres = self.edge_threshold
        height = edgemap.shape[0]
        width = edgemap.shape[1]
        edgemap2 = np.zeros((height,width))

        for i in range(1, height):
            for j in range(1, width):
                if edgemap[i,j] < thres:
                    for y in range (i-thicc+1,i+thicc):	#minimum thickness 1
                        for x in range(j-thicc+1,j+thicc):
                            if 0 <= y < height and 0 <= x < width:
                                edgemap2[y,x] = 1
                    #if i + thicc < height and j + thicc < width:
                    #    edgemap2[i+thicc, j+thicc] = 1
        return edgemap2

    def color_change(self,pixel):

        ratio = self.ratio
        [b,g,r] = pixel

        # use gray scale to maintain color ratio
        gr = 1 + b/3 + g/3 + r/3
        scaled_gr = gr//ratio*ratio
        [b,g,r] = [scaled_gr/gr*b,scaled_gr/gr*g,scaled_gr/gr*r]

        # multiply channel intensity
        pixel = np.array([b,g,r]) * np.array(self.intensity)
        return pixel

def cartoonize(frame1, mode):
    n = frame1.shape[0]
    m = frame1.shape[1]
    bm = np.ones((n, m))
    frameb = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)
    em = ed.get_edge_map(frameb)
    cf = CartoonFilter(25, 210, 1)

    if mode == 'happy':
        cf.intensity = [0.8,0.5,1]
    elif mode == 'surprise' or mode == 'fear':
        cf.intensity = [0,1,1]
    elif mode == 'sad':
        cf.intensity = [0.7,0.7,0.7]
    elif mode == 'contemp':
        cf.intensity = [0.5,0.8,0.9]
    else:
        cf.intensity = [1,1,1]
    frame2 = cf.cartoon_color(frame1, bm, em)
    return frame2

def video():
    video = cv2.VideoCapture('../videos/thang.mp4')
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    writer = cv2.VideoWriter('../videos/output_face.avi', fourcc, 20.0, (width, height))

    num_frame = 0
    while True:
        ret, frame = video.read()
        if ret == True:
            num_frame += 1
            print(num_frame)
            output = cartoonize(frame, 'happy')
            if output is not None:
                #cv2.imshow('1', output)
                #cv2.waitKey(1)
                writer.write(output)
        else:
            break

    video.release()
    writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    video()
