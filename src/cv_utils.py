import numpy as np
import cv2
import matplotlib.pyplot as plt


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

def crossProduct(o, a, b):
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

def mapBack(origPnts, sortedPnts, hull):
	res = []
	for index, pnt in enumerate(origPnts):
		for i in range(len(hull)):
			p = sortedPnts[hull[i][0]]
			if (p[0] == pnt[0]) and (p[1] == pnt[1]):
				res.append([index])
	return res

def convexHull(points):
    if len(points) <= 1:
        return points

    x_coords = np.fromiter(map(lambda x: x[0], points), dtype=np.int)
    y_coords = np.fromiter(map(lambda x: x[1], points), dtype=np.int)
    print(x_coords)
    print(y_coords)
    plt.plot(x_coords, y_coords, 'ro')
    plt.axis([0, 600, 0, 600])

    # Build lower hull 
    lower = []
    for i in range(len(points)):
    	p = points[i]
    	while len(lower) >= 2 and crossProduct(points[lower[-2][0]], points[lower[-1][0]], p) <= 0:
    		lower.pop()
    	lower.append([i])

    # Build upper hull
    upper = []
    for i in range(len(points)):
    	p = points[len(points) - 1 - i]
    	while len(upper) >= 2 and crossProduct(points[upper[-2][0]], points[upper[-1][0]], p) <= 0:
    		upper.pop()
    	upper.append([len(points) - 1 - i])
    res = lower[:-1] + upper[:-1]
    return res

def warpAffine(src, warpMat, w, h):
	src_w = src.shape[1]
	src_h = src.shape[0]
	dst = np.copy(src)
	dst = cv2.resize(dst, (w, h))
	valid = np.zeros((h, w))
	for x in range(w):
		for y in range(h):
			destination = np.matmul(warpMat, [[x], [y], [1]])
			src_x = int(destination[0][0])
			src_y = int(destination[1][0])
			if src_x >= 0 and src_x < src_w and src_y >= 0 and src_y < src_h:
				value = src[src_y][src_x]
				dst[y][x] = value
				valid[y][x] = 1

	for x in range(w):
		for y in range(h):
			if(valid[y][x] == 0):
				dst[y][x] = [0, 0, 0]


	return np.array(dst)

