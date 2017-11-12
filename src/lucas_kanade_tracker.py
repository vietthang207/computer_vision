import numpy as np
from utils import *
import cv2
from corner_detector import *
from constants import *

def pyramid_reduce(frame):
	sz = frame.shape
	a = 0.375
	ker1d = np.array([[0.25-a/2, 0.25, a, 0.25, 0.25-a/2]])
	kernel = np.kron(ker1d, ker1d.transpose())
	kernel = np.array([[1]])
	smoothened_frame = (conv2d(frame, kernel, 'same')).astype(np.float32)
	# for i in range(sz[0]):
		# for j in range(sz[1]):
			# if (smoothened_frame[i][j] != frame[i][j]):
				# print("diff", i, j, smoothened_frame[i][j], frame[i][j])
	return smoothened_frame[::2, ::2]

def gen_pyramid(frame, max_level):
	pyr = []
	pyr.append(frame)
	for i in range(max_level - 1):
		pyr.append(pyramid_reduce(pyr[-1]))
	return pyr

def bitmap_to_feature_list(bitmap):
	feature_list = np.zeros((Constant.NUM_FEATURE_TO_TRACK, 1, 2), np.float32)
	sz = bitmap.shape
	c = 0
	for i in range(sz[0]):
		for j in range(sz[1]):
			if bitmap[i][j] != 0:
				feature_list[c] = [i, j]
				c += 1
	print(c)
	return feature_list

def calculate_velocity(frame1, frame2, kernel, coordinate):
	frame_height = frame1.shape[0]
	frame_width = frame1.shape[1]
	kernel_size = kernel.shape[0]
	min_img_size = 8
	max_pyr_level = int((np.log2(min(frame_width, frame_height)/min_img_size)))
	max_pyr_level = 1
	print("max_pyr_level", max_pyr_level)
	pyramid1 = gen_pyramid(frame1, max_pyr_level)
	pyramid2 = gen_pyramid(frame2, max_pyr_level)
	# output = feature_list * 1.0 / pow(2, max_pyr_level)
	velocity = np.zeros(coordinate.shape, np.float32)
	st = np.ones((coordinate.shape[0]))

	for level in range(max_pyr_level - 1, -1, -1):
		print("level", level)
		frame1 = pyramid1[level]
		frame2 = pyramid2[level]
		frame_height = frame1.shape[0]
		frame_width = frame1.shape[1]
		scaled_coordinate = coordinate * 1.0 / pow(2, level)
		scaled_coordinate_new = scaled_coordinate + velocity * 2
		
		W = np.zeros((kernel_size * kernel_size, kernel_size * kernel_size))
		for i in range(kernel_size):
			for j in range(kernel_size):
				pos = i * kernel_size + j
				W[pos][pos] = np.sqrt(kernel[i][j])

		offset = kernel_size // 2

		for k in range(coordinate.shape[0]):
			i , j = int(scaled_coordinate[k, 0, 0]), int(scaled_coordinate[k, 0, 1])
			i_new, j_new =  int(scaled_coordinate_new[k, 0, 0]), int(scaled_coordinate_new[k, 0, 1])
			if i != i_new or j != j_new:
				print("diff")
			if i < offset or j < offset or i > frame_height - offset - 1 or j > frame_width - offset - 1:
				st[k] = 0
				continue
			if i_new < offset or j_new < offset or i_new >= frame_height - offset - 1 or j_new >= frame_width - offset - 1:
				st[k] = 0
				continue
			# print(f, i, j)
			# gradients inside window w
			Ix_w = frame1[i - offset : i + offset + 1, j - offset +1 : j + offset + 2] - frame1[i - offset : i + offset + 1, j - offset : j + offset + 1]
			Iy_w = - frame1[i - offset +1 : i + offset + 2, j - offset : j + offset + 1] + frame1[i - offset : i + offset + 1, j - offset : j + offset + 1]
			It_w = frame2[i - offset : i + offset + 1, j - offset : j + offset + 1] - frame1[i - offset : i + offset + 1, j - offset : j + offset + 1]
			# Ix_w = np.zeros_like(Ix_w)
			# Iy_w = np.zeros_like(Iy_w)
			# It_w = np.zeros_like(It_w)

			# print(Ix_w.shape)
			Ix_w = Ix_w.reshape(kernel_size * kernel_size, 1)
			Iy_w = Iy_w.reshape(kernel_size * kernel_size, 1)
			# TODO: which sign is this?
			b = - It_w.reshape(kernel_size * kernel_size, 1)
			# A = np.array([Ix_w, Iy_w])
			A = np.zeros((kernel_size * kernel_size, 2))
			for i1 in range(kernel_size * kernel_size):
				A[i1, 0] = Ix_w[i1]
				A[i1, 1] = Iy_w[i1]
			# print(A.shape)

			# A_t = np.transpose(A)
			# print(A_t.shape, W.shape, A.shape, b.shape)
			# v = np.linalg.inv(A_t.dot(W).dot(A)).dot(A_t).dot(W).dot(b)
			A = W.dot(A)
			b = W.dot(b)
			v = np.linalg.pinv(A).dot(b)
			velocity[k, 0, 0] = velocity[k, 0, 0]*2 + v[0]
			velocity[k, 0, 1] = velocity[k, 0, 1]*2 + v[1]
	output = (coordinate + velocity)
	return output, st
				
if __name__ == "__main__":
	cap = cv2.VideoCapture('../videos/traffic.mp4')
	ret, old_frame = cap.read()
	old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
	kernel = gaussWin(7).dot(gaussWin(7).transpose())
	# feature_list = corner_detector(old_gray, 7, 7)
	# p0 = bitmap_to_feature_list(feature_list)
	lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
	feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )
	p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
	# print("shape", p0.shape)
	color = np.random.randint(0,255,(Constant.NUM_FEATURE_TO_TRACK,3))
	mask = np.zeros_like(old_frame)
	while(1):
	    ret,frame = cap.read()
	    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	    # calculate optical flow
	    # p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
	    p1, st = calculate_velocity(old_gray, frame_gray, kernel, p0)
	    # Select good points
	    good_new = p1[st==1]
	    good_old = p0[st==1]
	    # draw the tracks
	    for i,(new,old) in enumerate(zip(good_new,good_old)):
	        a,b = new.ravel()
	        c,d = old.ravel()
	        cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
	        cv2.circle(frame,(a,b),5,color[i].tolist(),-1)
	    img = cv2.add(frame,mask)
	    cv2.imshow('frame',img)
	    k = cv2.waitKey(30) & 0xff
	    if k == 27:
	        break
	    # Now update the previous frame and previous points
	    old_gray = frame_gray.copy()
	    p0 = good_new.reshape(-1,1,2)
	cv2.destroyAllWindows()
	cap.release()
	# ret, frame2 = cap.read()
	# frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
	# cv2.imshow('frame1', frame1)
	# cv2.waitKey()
	
	# print(kernel.shape)
	# blur = gen_pyramid(frame1, 5)
	# for i in range(len(blur)):
	# 	cv2.imshow("blur " + str(i), blur[i])
	# 	cv2.waitKey()

	# tracker = LucasKanadeTracker()
	# vx, vy = tracker.calculate_velocity(frame1, frame2, kernel)
	# frame_height = frame1.shape[0]
	# frame_width = frame1.shape[1]
	# print(frame1.shape)
	# for i in range(frame_height):
	# 	for j in range(frame_width):
	# 		v = np.sqrt(vx[i][j]*vx[i][j] + vy[i][j]*vy[i][j])
	# 		if v > 0.5:
	# 			frame1[i][j] = 0

	# print(frame1.shape)
	# cv2.imshow('frame1 velo', frame1)
	# cv2.waitKey()
