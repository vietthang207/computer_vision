import numpy as np
from utils import *

class LucasKanadeTracker(object):
	"""docstring for LucasKanadeTracker"""
	def __init__(self):
		
	def calculate_velocity(self, frame1, frame2, kernel):
		frame_height = frame1.shape[0]
		frame_width = frame1.shape[1]
		kernel_size = kernel.shape[0]
		W = np.zeros((kernel_size * kernel_size, kernel_size * kernel_size))
		for i in range(kernel_size):
			for j in range(kernel_size):
				pos = i * kernel_size + j
				W[pos][pos] = kernel[i][j]

		# gradient
		Ix = conv2d(frame1, [-1 1; -1 1], 'same')
		Iy = conv2d(frame1, [-1 -1; 1 1], 'same')
		# I - J: partial derivative wrt time
		It = conv2d(frame1, [1 1; 1 1], 'same') - conv2d(frame2, [-1 -1; -1 -1], 'same')

		vx = np.zeros(frame1.shape)
		vy = np.zeros(frame1.shape)

		offset = kernel_size // 2

		for i in range(offset, frame_height - offset - 1):
			for j in range(offset, frame_width - offset - 1):
				# gradients inside window w
				Ix_w = Ix[i - offset : i + offset + 1][j - offset : j + offset + 1]
				Iy_w = Iy[i - offset : i + offset + 1][j - offset : j + offset + 1]
				It_w = It[i - offset : i + offset + 1][j - offset : j + offset + 1]

				Ix_w = Ix_w.reshape(kernel_size * kernel_size, 1)
				Iy_w = Iy_w.reshape(kernel_size * kernel_size, 1)
				# TODO: which sign is this?
				b = It.reshape(kernel_size * kernel_size, 1)
				A = concatenate((Ix_w, Iy_w))

				A_t = np.transpose(A)
				v = np.linalg.inv(A_t.dot(W).dot(A)).dot(A_t).dot(W).dot(b)
				vx[i][j] = v[0]
				vy[i][j] = v[1]
				
