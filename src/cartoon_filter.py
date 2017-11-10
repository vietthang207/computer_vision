import cv2
import numpy as np
from utils import *
import corner_detector as cd
import edge_detector as ed

class CartoonFilter(object):
	"""Cartoon filter for image"""
	def __init__(self, ratio, edge_threshold):
		self.ratio = ratio
		self.edge_threshold = edge_threshold

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
		r = self.ratio
		for i in range(1, height):
			for j in range(1, width):
				if edgemap[i,j] < self.edge_threshold:		# cond which edge to accept to be done later 
					img[i,j] = [0,0,0]
				elif bitmap[i][j] == 1:
					img[i,j] = [img[i,j,0]//r*r,img[i,j,1]//r*r,img[i,j,2]//r*r]
		return img

	# def cartoon_color(img, edgemap):
		
	# 	height = img.shape[0]
	# 	width = img.shape[1]
		
	# 	for i in range(1, height):
	# 		for j in range(1, width):
	# 			if edgemap[i,j]:		# cond which edge to accept to be done later 
	# 				img[i,j] = [0,0,0]
	# 			else:
	# 				img[i,j] = [img[i,j,0]//10*10,img[i,j,1]//10*10,img[i,j,2]//10*10]
	# 	return img

def cartoonize(frame1):
	n = frame1.shape[0]
	m = frame1.shape[1]
	bm = np.ones((n, m))
	frameb = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)
	em = ed.get_edge_map(frameb)
	# cv2.imshow('em', em.astype(np.uint8))
	# cv2.imwrite('cat bla.jpg', em)
	cf = CartoonFilter(40, 250)
	frame2 = cf.cartoon_color(frame1, bm, em)	
	cv2.imshow("new", frame2)
	cv2.waitKey(2000)

if __name__ == "__main__":
	cap = cv2.VideoCapture('../videos/traffic.mp4')
	ret, frame1 = cap.read()
	frame1 = cv2.imread("cat-61079_960_720.jpg")
	cv2.imshow("original", frame1)
	cv2.waitKey()
	n = frame1.shape[0]
	m = frame1.shape[1]
	bm = np.ones((n, m))
	frameb = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)
	em = ed.get_edge_map(frameb)
	cv2.imshow('em', em.astype(np.uint8))
	cv2.imwrite('cat bla.jpg', em)
	cf = CartoonFilter(40, 200)
	frame2 = cf.cartoon_color(frame1, bm, em)	
	cv2.imshow("new", frame2)
	cv2.waitKey()