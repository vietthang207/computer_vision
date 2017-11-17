#! /usr/bin/env python

import sys
import numpy as np
import cv2
import dlib
import math
from delaunay_triangulation import delaunay_triangulation
from cv_utils import findAffineTransformMatrix, warpAffine, boundingRect

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Read points from text file
def readPoints(path) :
    # Create an array of points.
    points = [];
    
    # Read points
    with open(path) as file :
        for line in file :
            x, y = line.split()
            points.append((int(x), int(y)))
    
    return points

# Apply affine transform calculated using srcTri and dstTri to src and
# output an image of size.
def applyAffineTransform(src, srcTri, dstTri, size) :
    
    # Given a pair of triangles, find the affine transform.
    # warpMat = cv2.getAffineTransform( np.float32(srcTri), np.float32(dstTri) )

    warpMat = findAffineTransformMatrix( np.float32(dstTri), np.float32(srcTri) )


    # Apply the Affine Transform just found to the src image
    #dst = cv2.warpAffine( src, warpMat, (size[0], size[1]) )
    dst = warpAffine( src, warpMat, size[0], size[1])
    # fakeDst = warpAffine(src, warpMat, size[0], size[1])
    # resizedDst = cv2.resize(fakeDst, (size[0], size[1]), interpolation = cv2.INTER_AREA)
    # resizedDst = resizedDst.astype(int)
  

    return dst


# Check if a point is inside a rectangle
def rectContains(rect, point) :
    if point[0] < rect[0] :
        return False
    elif point[1] < rect[1] :
        return False
    elif point[0] > rect[0] + rect[2] :
        return False
    elif point[1] > rect[1] + rect[3] :
        return False
    return True


#calculate delanauy triangle
def calculateDelaunayTriangles(rect, points):
    #create subdiv
    triangleList = delaunay_triangulation(points);
    
    delaunayTri = []
    
    pt = []    
    
    count= 0    
    
    for t in triangleList:        
        pt.append((t[0], t[1]))
        pt.append((t[2], t[3]))
        pt.append((t[4], t[5]))
        
        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])        
        

        if rectContains(rect, pt1) and rectContains(rect, pt2) and rectContains(rect, pt3):
        
            # cv2.line(img, pt1, pt2, (255, 255, 255), 1, cv2.LINE_AA, 0)
            # cv2.line(img, pt2, pt3, (255, 255, 255), 1, cv2.LINE_AA, 0)
            # cv2.line(img, pt3, pt1, (255, 255, 255), 1, cv2.LINE_AA, 0)   
            count = count + 1 
            ind = []
            for j in range(0, 3):
                for k in range(0, len(points)):                    
                    if(abs(pt[j][0] - points[k][0]) < 1.0 and abs(pt[j][1] - points[k][1]) < 1.0):
                        ind.append(k)                            
            if len(ind) == 3:                                                
                delaunayTri.append((ind[0], ind[1], ind[2]))
        pt = []        
    #print("result length ")
    #print(np.array(delaunayTri).shape)   
    #print(count)     
    
    return delaunayTri
        

# Warps and alpha blends triangular regions from img1 and img2 to img
def warpTriangle(img1, img2, t1, t2) :

    # Find bounding rectangle for each triangle
    r1 = boundingRect(np.float32([t1]))
    r2 = boundingRect(np.float32([t2]))
    # Offset points by left top corner of the respective rectangles
    t1Rect = [] 
    t2Rect = []
    t2RectInt = []

    for i in range(0, 3):
        t1Rect.append(((t1[i][0] - r1[0]),(t1[i][1] - r1[1])))
        t2Rect.append(((t2[i][0] - r2[0]),(t2[i][1] - r2[1])))
        t2RectInt.append(((t2[i][0] - r2[0]),(t2[i][1] - r2[1])))

    # Get mask by filling the triangle of t2, This mask when multiplied with the output 
    # image turns all pixels outside the triangle black while preserving the color of all 
    # pixels inside the triangle.
    mask = np.zeros((r2[3], r2[2], 3), dtype = np.float32)
    cv2.fillConvexPoly(mask, np.int32(t2RectInt), (1.0, 1.0, 1.0), 16, 0);

    # Apply warpImage to small rectangular patches
    img1Rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
    #img2Rect = np.zeros((r2[3], r2[2]), dtype = img1Rect.dtype)
    
    size = (r2[2], r2[3])

    img2Rect = applyAffineTransform(img1Rect, t1Rect, t2Rect, size)
    
    
    #print("mask")
    #print(mask.shape)
    # Preserve all inside, delete all outside
    img2Rect = img2Rect * mask

    # Preserve all outside, delete all insisde.
    # Copy triangular region of the rectangular patch to the output image
    # When multiply the destination image with (1 - mask), then we will preserve all pixels
    # outside the triangle, and turns all pixels inside triangle to black, after that we add with img2Rect.
    img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] * ( (1.0, 1.0, 1.0) - mask )
     
    img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] + img2Rect 
    

def get_landmarks(image):
    detections = detector(image, 1)
    #print(np.array(detections).shape)
    for k,d in enumerate(detections): #For all detected face instances individually
        res = []
        shape = predictor(image, d)
        for i in range(1,69): #Store X and Y coordinates in two lists
            x = (shape.part(i-1).x)
            y = (shape.part(i-1).y)
            res.append((int(x), int(y)))
        
        return res


def face_swap(img1, img2):
    
    img1Warped = np.copy(img2);
     
    # res = get_landmarks(img1)
    # print(res.shape)

    # Read array of corresponding points
    points1 = get_landmarks(img1)
    points2 = get_landmarks(img2)

    if points1 is None or points2 is None:
        return None
    
    
    # Find convex hull
    hull1 = []
    hull2 = []

    hullIndex = cv2.convexHull(np.array(points2), returnPoints = False)

    for i in range(0, len(hullIndex)):
        hull1.append(points1[int(hullIndex[i])])
        hull2.append(points2[int(hullIndex[i])])
    
    
    # Find delanauy triangulation for convex hull points
    sizeImg2 = img2.shape    
    #print(sizeImg2)
    rect = (0, 0, sizeImg2[1], sizeImg2[0])

    dt = calculateDelaunayTriangles(rect, hull2)
    
    if len(dt) == 0:
        return None
    
    # Apply affine transformation to Delaunay triangles
    for i in range(0, len(dt)):
        t1 = []
        t2 = []
        
        #get points for img1, img2 corresponding to the triangles
        for j in range(0, 3):
            t1.append(hull1[dt[i][j]])
            t2.append(hull2[dt[i][j]])
        
        warpTriangle(img1, img1Warped, t1, t2)

    #print("OKKKKK")
            
    # Calculate Mask
    hull8U = []
    for i in range(0, len(hull2)):
        hull8U.append((hull2[i][0], hull2[i][1]))
    
    mask = np.zeros(img2.shape, dtype = img2.dtype)  
    
    cv2.fillConvexPoly(mask, np.int32(hull8U), (255, 255, 255))
    
    r = cv2.boundingRect(np.float32([hull2]))    
    
    center = ((r[0]+int(r[2]/2), r[1]+int(r[3]/2)))
        
    
    # Clone seamlessly.
    output = cv2.seamlessClone(np.uint8(img1Warped), img2, mask, center, cv2.NORMAL_CLONE)
   
    return output
    
 
if __name__ == '__main__' :
    

    # Make sure OpenCV is version 3.0 or above
    #(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

    #if int(major_ver) < 3 :
    #    print >>sys.stderr, 'ERROR: Script needs OpenCV 3.0 or higher'
    #    sys.exit(1)

    
    # Read images
    filename2 = 'ntk.jpg'
    filename1 = 'taylor-swift-face.jpg'
    
    img1 = cv2.imread(filename1);
    img2 = cv2.imread(filename2);
    
    img1Warped = np.copy(img2);
     
    # res = get_landmarks(img1)
    # print(res.shape)

    # Read array of corresponding points
    points1 = get_landmarks(img1)
    points2 = get_landmarks(img2)
    
    
    # Find convex hull
    hull1 = []
    hull2 = []

    hullIndex = cv2.convexHull(np.array(points2), returnPoints = False)

    for i in range(0, len(hullIndex)):
        hull1.append(points1[int(hullIndex[i])])
        hull2.append(points2[int(hullIndex[i])])
    
    
    # Find delanauy triangulation for convex hull points
    sizeImg2 = img2.shape    
    print(sizeImg2)
    rect = (0, 0, sizeImg2[1], sizeImg2[0])

    dt = calculateDelaunayTriangles(rect, hull2)
    
    if len(dt) == 0:
        quit()
    
    # Apply affine transformation to Delaunay triangles
    for i in range(0, len(dt)):
        t1 = []
        t2 = []
        
        #get points for img1, img2 corresponding to the triangles
        for j in range(0, 3):
            t1.append(hull1[dt[i][j]])
            t2.append(hull2[dt[i][j]])
        
        warpTriangle(img1, img1Warped, t1, t2)

    print("OKKKKK")
            
    # Calculate Mask
    hull8U = []
    for i in range(0, len(hull2)):
        hull8U.append((hull2[i][0], hull2[i][1]))
    
    mask = np.zeros(img2.shape, dtype = img2.dtype)  
    
    cv2.fillConvexPoly(mask, np.int32(hull8U), (255, 255, 255))
    
    r = cv2.boundingRect(np.float32([hull2]))    
    
    center = ((r[0]+int(r[2]/2), r[1]+int(r[3]/2)))
        
    
    # Clone seamlessly.
    output = cv2.seamlessClone(np.uint8(img1Warped), img2, mask, center, cv2.NORMAL_CLONE)
    
    cv2.imshow("Face Swapped", output)
    cv2.waitKey(20000)
    
    cv2.destroyAllWindows()
        
