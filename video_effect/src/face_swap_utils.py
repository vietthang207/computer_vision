from __future__ import division
import numpy as np
import cv2
import matplotlib.pyplot as plt
from collections import namedtuple
from math import sqrt


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

Point = namedtuple('Point', ['x', 'y'])
Triangle = namedtuple('Triangle', ['a', 'b', 'c'])
Line = namedtuple('Line', ['slope', 'yintersect'])
Circle = namedtuple('Circle', ['center', 'radius'])
LineSegment = namedtuple('LineSegment', ['start', 'end'])


def create_line(p1, p2):
    if p1.x == p2.x:
        return None
    slope = (p2.y - p1.y)/(p2.x - p1.x)
    yintersect = (p1.y*p2.x - p2.y*p1.x)/(p2.x - p1.x)
    return Line(slope, yintersect)


def intersect(line1, line2):
    if line1.slope == line2.slope:
        return None
    x = (line2.yintersect - line1.yintersect) / (line1.slope - line2.slope)
    y = line1.slope * x + line1.yintersect
    return Point(x, y)


def perpendicular_slope(p1, p2):
    if p1.x == p2.x:
        return 0
    if p1.y == p2.y:
        return None
    return (p1.x - p2.x)/(p2.y - p1.y)


def dist(p1, p2):
    return sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)


def enclosed(points):
    xmin = min([p.x for p in points])
    xmax = max([p.x for p in points])
    ymin = min([p.y for p in points])
    ymax = max([p.y for p in points])

    line1 = Line(1, ymax-xmin) 
    line2 = Line(-1, ymax+xmax)
    line3 = Line(0, ymin)

    vertice1 = intersect(line1, line2)
    vertice2 = intersect(line1, line3)
    vertice3 = intersect(line2, line3)

    return Triangle(vertice1, vertice2, vertice3)


def is_collinear(a, b, c):
    det = b.x*c.y - b.y*c.x + a.x*b.y - a.x*c.y + a.y*c.x - a.y*b.x
    return det == 0


def circumcenter(triangle):
    """Assume that the triangle is valid"""
    midpoint1 = Point((triangle.a.x + triangle.b.x)/2, (triangle.a.y + triangle.b.y)/2)
    midpoint2 = Point((triangle.b.x + triangle.c.x)/2, (triangle.b.y + triangle.c.y)/2)
    
    if triangle.a.y == triangle.b.y:
        s2 = perpendicular_slope(triangle.b, triangle.c)
        l2 = Line(s2, midpoint2.y - s2*midpoint2.x)
        return Point(midpoint1.x, l2.slope*midpoint1.x + l2.yintersect)
    else:
        s1 = perpendicular_slope(triangle.a, triangle.b) 
        l1 = Line(s1, midpoint1.y - s1*midpoint1.x)

    if triangle.b.y == triangle.c.y:
        return Point(midpoint2.x, l1.slope*midpoint2.x + l1.yintersect)
    else:
        s2 = perpendicular_slope(triangle.b, triangle.c)
        l2 = Line(s2, midpoint2.y - s2*midpoint2.x)
        return intersect(l1, l2)


def circumcircle(triangle):
    if is_collinear(triangle.a, triangle.b, triangle.c):
        return None
    center = circumcenter(triangle)
    radius = dist(center, triangle.a)
    return Circle(center, radius)


def vertice_to_edge(triangle):
    return Triangle(LineSegment(triangle.a, triangle.b), LineSegment(triangle.b, triangle.c), LineSegment(triangle.c, triangle.a))


def triangle_from_edge_point(e, p):
    return Triangle(e.start, e.end, p)


def delaunay_triangulation(points):
    """points is a list of tuples"""
    
    pointList = []
    for p in points:
        pointList.append(Point(p[0], p[1]))

    #find a super triangle that contains all points
    super_tri = enclosed(pointList)
    tri_set = [(super_tri, circumcircle(super_tri))]
   
    #add all points one at a time
    for p in pointList:
        #find triangles that are no longer valid
        bad_triangle_edge = []
        bad_triangle_vertice = []
        for (t, c) in tri_set:
            if dist(p, c.center) <= c.radius:
                bad_triangle_edge.append(vertice_to_edge(t))
                bad_triangle_vertice.append((t, c))

        #find boundary of polygon hole
        polygon = []
        for t in bad_triangle_edge:
            for e in t:
                shared = False
                for t2 in bad_triangle_edge:
                    if t != t2 and (e in t2 or tuple(reversed(e)) in t2):
                        shared = True
                        break

                if not shared:
                    polygon.append(e)
    
        #remove bad triangles from triangulation
        for pair in bad_triangle_vertice:
            tri_set.remove(pair)

        #re-triangulate the polygon hole
        for e in polygon:
            if not is_collinear(e[0], e[1], p):
                new_tri = triangle_from_edge_point(e, p)
                tri_set.append((new_tri, circumcircle(new_tri)))

    out_range = []
    #clean up the super triangle
    for pair in tri_set:
        tri = pair[0]
        for v in tri:
            if v in super_tri:
                out_range.append(pair)
                break
    for pair in out_range:
        tri_set.remove(pair)

    return [(t[0][0][0], t[0][0][1], t[0][1][0], t[0][1][1], t[0][2][0], t[0][2][1]) for t in tri_set]