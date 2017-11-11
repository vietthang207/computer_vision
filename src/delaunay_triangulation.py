from __future__ import division
from collections import namedtuple
from math import sqrt

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
