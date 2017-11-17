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
	# kernel = gaussWin(13).dot(gaussWin(13).transpose())
	smoothened_frame = (conv2d(frame, kernel, 'same')).astype(np.float32)
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
	return feature_list

def find_good_features(frame):
	feature_list = corner_detector(frame, 7, 7)
	p0 = bitmap_to_feature_list(feature_list)
	return p0

def lk_pyr(frame1, frame2, kernel, coordinate):
	frame_height = frame1.shape[0]
	frame_width = frame1.shape[1]
	kernel_size = kernel.shape[0]
	min_img_size = 16
	max_pyr_level = int((np.log2(min(frame_width, frame_height)/min_img_size)))
	max_pyr_level = Constant.MAX_PYRAMID_LEVEL 
	# print("max_pyr_level", max_pyr_level)
	pyramid1 = gen_pyramid(frame1, max_pyr_level)
	pyramid2 = gen_pyramid(frame2, max_pyr_level)
	velocity = np.zeros(coordinate.shape, np.float32)
	st = np.ones((coordinate.shape[0]))

	for level in range(max_pyr_level - 1, -1, -1):
		frame1 = pyramid1[level]
		frame2 = pyramid2[level]
		scaled_coordinate = coordinate * 1.0 / pow(2, level)
		scaled_coordinate2 = (coordinate + velocity * 2) / pow(2, level)
		addition_velocity, st1 = lk_one_level(frame1, frame2, kernel, scaled_coordinate, scaled_coordinate2)
		velocity = velocity * 2 + addition_velocity
		for i in range(st.shape[0]):
			if st1[i] == 0:
				st[i] = 0
	return coordinate + velocity, st

def lk_one_level(frame1, frame2, kernel, coordinate, coordinate2):
	frame1 = frame1.astype(np.float32)
	frame2 = frame2.astype(np.float32)
	kernel = kernel.astype(np.float32)
	frame_height = frame1.shape[0]
	frame_width = frame1.shape[1]
	kernel_size = kernel.shape[0]
	velocity = np.zeros(coordinate.shape, np.float32)
	st = np.ones((coordinate.shape[0]))

	offset = kernel_size // 2

	for k in range(coordinate.shape[0]):
		i , j = int(coordinate[k, 0, 1]), int(coordinate[k, 0, 0])
		i2, j2 = int(coordinate2[k, 0, 1]), int(coordinate2[k, 0, 0]) 

		Ix_w = np.zeros_like(kernel)
		Iy_w = np.zeros_like(kernel)
		It_w = np.zeros_like(kernel)

		for x in range(kernel_size):
			for y in range(kernel_size):
				r = i - offset + x
				c = j - offset + y
				r2 = i2 - offset + x
				c2 = j2 - offset + y
				if r >= 0 and c >= 0 and r < frame_height - 1 and c <frame_width - 1:						
					Ix_w[x, y] = frame1[r, c + 1] - frame1[r, c]
					Iy_w[x, y] = frame1[r + 1, c] - frame1[r, c]
					It_w[x, y] = frame1[i, j] - frame2[i2, j2]
				else:
					st[k] = 0
					break

		z1 = z2 = z3 = z4 = b1 = b2 = 0
		for i1 in range(kernel_size):
			for j1 in range(kernel_size):
				z1 += Ix_w[i1, j1] * Ix_w[i1, j1] * kernel[i1, j1]
				z2 += Ix_w[i1, j1] * Iy_w[i1, j1] * kernel[i1, j1]
				z3 += Ix_w[i1, j1] * Iy_w[i1, j1] * kernel[i1, j1]
				z4 += Iy_w[i1, j1] * Iy_w[i1, j1] * kernel[i1, j1]
				b1 += It_w[i1, j1] * Ix_w[i1, j1] * kernel[i1, j1]
				b2 += It_w[i1, j1] * Iy_w[i1, j1] * kernel[i1, j1]
		Z = np.array([[z1, z2], [z3, z4]])
		B = np.array([[b1], [b2]])
		try:
			v = np.dot(np.linalg.inv(Z), B)
			velocity[k, 0, 0] = v[0]
			velocity[k, 0, 1] = v[1]
		except np.linalg.linalg.LinAlgError:
			st[k] = 0
		
	return velocity, st
				
if __name__ == "__main__":
	cap = cv2.VideoCapture('../videos/bird.mp4')
	ret, old_frame = cap.read()
	old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
	width = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
	height = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))
	fourcc = cv2.cv.CV_FOURCC(*'XVID')
	writer = cv2.VideoWriter('../videos/bird_output.avi', fourcc, 20.0, (width, height))

	kernel = gaussWin(13).dot(gaussWin(13).transpose())
	c = 0
	for i in range(kernel.shape[0]):
		for j in range(kernel.shape[0]):
			c += kernel[i, j]
	lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
	feature_params = dict( maxCorners = Constant.NUM_FEATURE_TO_TRACK,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )
	
	# print("shape", p0.shape)
	color = np.random.randint(0,255,(Constant.NUM_FEATURE_TO_TRACK,3))
	mask = np.zeros_like(old_frame)
	counter = 0;
	while(1):
	    if counter % Constant.FEATURE_RESET_INTERVAL == 0:
	        p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
	        # p0 = find_good_features(old_gray)
	        mask = np.zeros_like(old_frame)
	    counter += 1
	    ret,frame = cap.read()
	    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	    cv2.imwrite("frameee1.jpg", frame_gray)
	    # calculate optical flow
	    # p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
	    p1, st = lk_pyr(old_gray, frame_gray, kernel, p0)
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
	    writer.write(img)
	    cv2.imshow('frame',img)
	    k = cv2.waitKey(30) & 0xff
	    if k == 27:
	        break
	    # Now update the previous frame and previous points
	    old_gray = frame_gray.copy()
	    p0 = good_new.reshape(-1,1,2)
	cv2.destroyAllWindows()
	cap.release()
	writer.release()
	