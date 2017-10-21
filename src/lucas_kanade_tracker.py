import numpy as np
from utils import *
import cv2

class LucasKanadeTracker(object):
	"""docstring for LucasKanadeTracker"""
		
	def calculate_velocity(self, frame1, frame2, kernel):
		frame_height = frame1.shape[0]
		frame_width = frame1.shape[1]
		kernel_size = kernel.shape[0]
		W = np.zeros((kernel_size * kernel_size, kernel_size * kernel_size))
		for i in range(kernel_size):
			for j in range(kernel_size):
				pos = i * kernel_size + j
				W[pos][pos] = np.sqrt(kernel[i][j])

		# gradient
		Ix = conv2d(frame1, np.array([[-1, 1], [-1, 1]]), 'same')
		Iy = conv2d(frame1, np.array([[-1, -1], [1, 1]]), 'same')
		# I - J: partial derivative wrt time
		It = conv2d(frame1, np.array([[1, 1], [1, 1]]), 'same') + conv2d(frame2, np.array([[-1, -1], [-1, -1]]), 'same')
		print('finish conv2d')
		vx = np.zeros(frame1.shape)
		vy = np.zeros(frame1.shape)
		# return vx, vy
		offset = kernel_size // 2

		for i in range(offset, frame_height - offset - 1):
			print(i)
			for j in range(offset, frame_width - offset - 1):
				# gradients inside window w
				Ix_w = Ix[i - offset : i + offset + 1, j - offset : j + offset + 1]
				Iy_w = Iy[i - offset : i + offset + 1, j - offset : j + offset + 1]
				It_w = It[i - offset : i + offset + 1, j - offset : j + offset + 1]

				# print(Ix_w.shape)
				Ix_w = Ix_w.reshape(kernel_size * kernel_size, 1)
				Iy_w = Iy_w.reshape(kernel_size * kernel_size, 1)
				# TODO: which sign is this?
				b = - It_w.reshape(kernel_size * kernel_size, 1)
				A = np.array([Ix_w, Iy_w])
				A = A.reshape(kernel_size * kernel_size, 2)
				# print(A.shape)

				A_t = np.transpose(A)
				# print(A_t.shape, W.shape, A.shape, b.shape)
				# v = np.linalg.inv(A_t.dot(W).dot(A)).dot(A_t).dot(W).dot(b)
				# A = W.dot(A)
				# b = W.dot(b)
				v = np.linalg.pinv(A).dot(b)
				vx[i][j] = v[0]
				vy[i][j] = v[1]
		return vx, vy
				
if __name__ == "__main__":
	cap = cv2.VideoCapture('../videos/traffic.mp4')
	ret, frame1 = cap.read()
	frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
	ret, frame2 = cap.read()
	frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
	cv2.imshow('frame1', frame1)
	cv2.waitKey()
	kernel = gaussWin(45).dot(gaussWin(45).transpose())
	print(kernel.shape)
	tracker = LucasKanadeTracker()
	vx, vy = tracker.calculate_velocity(frame1, frame2, kernel)
	frame_height = frame1.shape[0]
	frame_width = frame1.shape[1]
	print(frame1.shape)
	for i in range(frame_height):
		for j in range(frame_width):
			v = np.sqrt(vx[i][j]*vx[i][j] + vy[i][j]*vy[i][j])
			if v > 0.2:
				frame1[i][j] = 0

	print(frame1.shape)
	cv2.imshow('frame1 velo', frame1)
	cv2.waitKey()
