import cv2 as cv
import numpy as np
import math
import cv2.aruco as aruco



def augmentingImage(bbox,id,img,imgAug,draw=True):

    tl=bbox[0][0],bbox[0][1]
    tr=bbox[1][0],bbox[1][1]
    br=bbox[2][0],bbox[2][1]
    bl=bbox[3][0],bbox[3][1]

    h,w,c=imgAug.shape
    pts1=np.array([tl,tr,br,bl])
    pts2=np.float32([[0,0],[w,0],[w,h],[0,h]])
    matrix,_=cv.findHomography(pts2,pts1)
    imgOut=cv.warpPerspective(imgAug,matrix,(img.shape[1],img.shape[0]))
    cv.fillConvexPoly(img,pts1.astype(int),(0,0,0))
    imgOut=img+imgOut
    return imgOut

def rotatingImage(img,angle,rotPoint=None):

    (y,x)=img.shape[:2]
    if rotPoint==None:
        rotPoint=(x/2,y/2)
    rotMat=cv.getRotationMatrix2D(rotPoint,angle,1.0)
    dimension=(x,y)
    return cv.warpAffine(img,rotMat,dimension)

def findAruco(img,markerSize=5,totalMarker=250,draw=True):
    imgGray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    key=getattr(aruco,f'DICT_{markerSize}X{markerSize}_{totalMarker}')
    arucoDict=aruco.Dictionary_get(key)
    parameters=aruco.DetectorParameters_create()
    bboxs,ids,rejected=aruco.detectMarkers(imgGray,arucoDict,parameters=parameters)
    x1, y1 = (int(bboxs[0][0][0][0]), int(bboxs[0][0][0][1]))
    x2, y2 = (int(bboxs[0][0][1][0]), int(bboxs[0][0][1][1]))
    y_net = y2 - y1
    x_net = x2 - x1
    sideLength = ((y_net) ** 2 + (x_net) ** 2) ** (1 / 2)
    tanx = float(y_net) / x_net
    angle = math.atan(tanx) * (180 / 3.142)
    if draw:
        aruco.drawDetectedMarkers(img,bboxs)
    return bboxs,ids[0][0],angle,(x1,y1),sideLength



def findContour(img,color):
    _, thresh = cv.threshold(img, 245, 255, cv.THRESH_BINARY)
    contours, _ = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    i = 0
    for contour in contours:
        if i == 0:
            i = 1
            continue
        approx = cv.approxPolyDP(contour, 0.01 * cv.arcLength(contour, True), True)
        x=approx.ravel()[0]
        y=approx.ravel()[1]

        if len(approx) == 4:
            x1, y1, w, h = cv.boundingRect(approx)
            M=cv.moments(contour)
            if M['m00']!=0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
            aspect_ratio = float(w) / h
            if aspect_ratio>= 0.98 and aspect_ratio <= 1.02:
                rectangle = cv.minAreaRect(contour)
                box = cv.boxPoints(rectangle)
                print(len(approx))
                print(box)
                box1 = np.int0(box)
                cv.drawContours(img, [box1], 0, (255, 0, 0), 2)
                # cv.drawContours(final, [approx], 0, (0, 0, 255), 1)
                cv.putText(final,color, (cx,cy), cv.FONT_ITALIC, 0.5, (255, 0, 0), 1)
                return box

final=cv.imread("CVtask.jpg",1)
final=cv.resize(final,(int(final.shape[1]/2),int(final.shape[0]/2)),interpolation=cv.INTER_AREA)
gray=cv.cvtColor(final,cv.COLOR_BGR2GRAY)
        
         # green
hsv=cv.cvtColor(final,cv.COLOR_BGR2HSV)
lower_green=np.array([40,100,100])
upper_green=np.array([70,255,255])
green=cv.inRange(hsv,lower_green,upper_green)
        # Orange
lower_orange=np.array([10,40,40])
upper_orange=np.array([30,255,255])
orange=cv.inRange(hsv,lower_orange,upper_orange)
        # Black
lower_black=np.array([0,0,0])
upper_black=np.array([10,210,210])
black=cv.inRange(hsv,lower_black,upper_black)
black=cv.bilateralFilter(black,10,25,25)
black=cv.GaussianBlur(black,(3,3),0)

        # Pink Peach
lower_pink=np.array([0,8,54])
upper_pink=np.array([30,57,240])
pink=cv.inRange(hsv,lower_pink,upper_pink)

aruco_1=cv.imread("1.jpg")
aruco_2=cv.imread("2.jpg")
aruco_3=cv.imread("3.jpg")
aruco_4=cv.imread("4.jpg")

arucos=[aruco_1,aruco_2,aruco_3,aruco_4]
for arc in arucos:
    bbox,id,angle,point,L=findAruco(arc)
    rotated=rotatingImage(arc,angle,point)
    crop=rotated[point[1]:point[1]+int(L),point[0]:point[0]+int(L)]
    print(id)
    if id==1:
        box=findContour(green,str(id))
        final=augmentingImage(box,id,final,crop)
    elif id==2:
        box=findContour(orange,str(id))
        final = augmentingImage(box, id, final, crop)
    elif id==3:
        box=findContour(black,str(id))
        final = augmentingImage(box, id, final, crop)
    elif id==4:
        box=findContour(pink,str(id))
        final = augmentingImage(box, id, final, crop)


cv.imwrite("final.jpg",final)
cv.imshow("final",final)
cv.waitKey(0)
cv.destroyAllWindows()