import numpy as np
import cv2

def findAffineTransformMatrix(srcPnts, destPnts):
	a = np.array([[srcPnts[0][0], srcPnts[0][1], 1], [srcPnts[1][0], srcPnts[1][1], 1], [srcPnts[2][0], srcPnts[2][1], 1]])
	b = np.array([destPnts[0][0], destPnts[1][0], destPnts[2][0]])
	c = np.array([destPnts[0][1], destPnts[1][1], destPnts[2][1]])
	x = np.linalg.solve(a, b)
	y = np.linalg.solve(a, c)
	return np.array([x, y])



def boundingRect(arr):
	x_coords = map(lambda pnts: pnts[0], arr[0])
	y_coords = map(lambda pnts: pnts[1], arr[0])
	x_coords = np.fromiter(x_coords, dtype=np.int)
	y_coords = np.fromiter(y_coords, dtype=np.int)
	smallest_x = min(x_coords)
	smallest_y = min(y_coords)
	largest_x = max(x_coords)
	largest_y = max(y_coords)
	x = smallest_x
	y = smallest_y
	w = largest_x - smallest_x
	h = largest_y - smallest_y
	return (x, y, w, h)

	
def check(valid, x, y, w, h):
	if(valid[x][y] != 0):
		return False
	elif (x-1 >= 0) and (y-1>=0) and (x+1 < h) and (y+1 < w):
		if(valid[x-1][y] !=0) and (valid[x][y-1] !=0) and (valid[x+1][y] !=0) and (valid[x][y+1] !=0):
			return False
	return True			


def warpAffine(src, warpMat, w, h):
	old_w = src.shape[1]
	old_h = src.shape[0]
	dst = np.copy(src)
	# dst = np.zeros((h, w, 3))
	dst = cv2.resize(dst, (w, h))
	cnt = np.zeros((h, w))
	valid = np.zeros((h, w))
	for x in range(old_h):
		for y in range(old_w):
			destination = np.matmul(warpMat, [[x], [y], [1]])
			new_x = int(destination[0][0])
			new_y = int(destination[1][0])
			print('value ', x, y, new_x, new_y)

			if new_x >= 0 and new_x < h and new_y >= 0 and new_y < w:
				old_cnt = cnt[new_x][new_y]
				value = src[x][y]
				value = ((np.array(dst[new_x][new_y]) * old_cnt + np.array(src[x][y])) / (old_cnt + 1))
				dst[new_x][new_y][0] = value[0]
				dst[new_x][new_y][1] = value[1]
				dst[new_x][new_y][2] = value[2]
				valid[new_x][new_y] = 1
				# dst[new_x][new_y] = src[x][y]
				cnt[new_x][new_y] += 1
				# print("new dst ")
				# print(dst[new_x][new_y])
				# print(value)
			
				# print(src[x][y])

	for x in range(h):
		for y in range(w):
			if(check(valid, x, y, w, h)):
				dst[x][y] = [0, 0, 0]

	# cv2.imshow('a', dst)
	# cv2.waitKey(500)

	return np.array(dst)


