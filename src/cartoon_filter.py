import cv2
import numpy as np
from utils import *
import corner_detector as cd
import edge_detector as ed

class CartoonFilter(object):
	"""Cartoon filter for image"""
	def __init__(self, ratio, edge_threshold, edge_thickness):
		self.ratio = ratio
		self.edge_threshold = edge_threshold
		self.edge_thickness = edge_thickness
		self.intensity = [1,1,1]

	def setPresetMode(self,mode):
		if mode == 1:
			self.ratio = 20
			self.edge_threshold = 230
			self.edge_thickness = 1
			self.intensity = [1,0.95,0.92]
		elif mode == 2:
			self.ratio = 25
			self.edge_threshold = 230
			self.edge_thickness = 1
			self.intensity = [0.85,0.85,0.85]
		elif mode == 3:
			self.ratio = 30
			self.edge_threshold = 230
			self.edge_thickness = 1
			self.intensity = [0.7,0.7,0.8]


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
		edgemap2 = self.edgemap_modify(edgemap)

		for i in range(1, height):
			for j in range(1, width):
				if edgemap2[i,j]:
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

def cartoonize(frame1):
	n = frame1.shape[0]
	m = frame1.shape[1]
	bm = np.ones((n, m))
	frameb = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)
	em = ed.get_edge_map(frameb)
	# cv2.imshow('em', em.astype(np.uint8))
	# cv2.imwrite('cat bla.jpg', em)
	cf = CartoonFilter(40, 250, 1)
	frame2 = cf.cartoon_color(frame1, bm, em)
	cv2.imshow("new", frame2)
	cv2.waitKey(2000)

if __name__ == "__main__":
	cap = cv2.VideoCapture('../videos/vid2.mp4')
	cap.set(1,30);
	ret, frame1 = cap.read()
	# cv2.imshow("original", frame1)
	n = frame1.shape[0]
	m = frame1.shape[1]
	bm = np.ones((n, m))
	frameb = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)
	em = ed.get_edge_map(frameb)
	# cv2.imshow('edgemap', em.astype(np.uint8))
	cf = CartoonFilter(30, 200, 1)
	cf.setPresetMode(1);
	frame2 = cf.cartoon_color(frame1, bm, em)
	cv2.imshow("mode 1", frame2)
	cv2.imwrite("mode1.jpg", frame2)
	cf.setPresetMode(2);
	frame3 = cf.cartoon_color(frame1, bm, em)
	cv2.imshow("mode 2", frame3)
	cv2.imwrite("mode2.jpg", frame3)
	cf.setPresetMode(3);
	frame4 = cf.cartoon_color(frame1, bm, em)
	cv2.imshow("mode 3", frame4)
	cv2.imwrite("mode3.jpg", frame4)
	cv2.waitKey()
