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


# points = [[175, 198],[182, 237],[191, 274],[197, 183],[198, 312],[213, 348],[220, 173],[230, 215],[237, 380],[245, 223],[246, 177],[247, 207],[265, 223],[267, 207],[267, 406],[270, 187],[275, 352],[283, 219],[285, 355],[293, 199],[294, 373],[296, 348],[301, 310],[301, 431],[312, 315],[312, 381],[313, 343],[313, 357],[313, 358],[320, 214],[322, 241],[323, 269],[324, 297],[325, 319],[325, 359],[326, 347],[326, 358],[326, 382],[333, 437],[337, 315],[338, 342],[338, 357],[339, 356],[340, 380],[344, 190],[348, 310],[356, 347],[358, 371],[365, 218],[366, 428],[368, 178],[368, 353],[378, 350],[380, 205],[384, 221],[393, 169],[400, 204],[400, 404],[403, 220],[419, 211],[421, 166],[430, 376],[448, 172],[454, 345],[468, 310],[474, 271],[477, 232],[479, 194]]
# hull = [[67],[66],[65],[64],[63],[61],[57],[49],[38],[23],[14],[ 8],[ 5],[ 4],[ 1],[ 0],[ 3],[ 6],[60],[62]]

# convexHull(points, hull)
