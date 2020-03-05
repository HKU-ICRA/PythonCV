import cv2 as cv
import numpy as np
import math
'''
Helper functions begin
'''

def void_callback(x):
    pass

def DrawRotatedRect(img,rect,color=(0,255,0),thickness = 2):
    center = rect[0]
    angle  = rect[2]
    font   = cv.FONT_HERSHEY_COMPLEX
    cv.putText(img,str(angle),center,font,0.5,color,thickness,8,0)
    vertices = cv.boxPoints(rect)
    for i in range(4):
        cv.line(img,vertices[i],vertices[(i+1)%4],color,thickness)
    return img

def formatPrint(title,items,filename):
    try:
        file = open(filename,'w')
    except:
        print('cannot Open the file')
        return
    file.writelines(title+' {'+'\n')
    for i in items:
        file.writelines('  '+str(i)+'\n')
    file.writelines('}'+'\n')
    file.close()

'''
Helper functions end
'''

'''
Intermediate Classes Begin
'''
class LightBar:
    def __init__(self,vertices):
        # The length of edges
        edge1 = np.linalg.norm(vertices[0]-vertices[1])
        edge2 = np.linalg.norm(vertices[1]-vertices[2])
        if(edge1>edge2):
            self._width  = edge1
            self._height = edge2
            if(vertices[0][1]<vertices[1][1]):
                self._angle = math.atans(vertices[1][1]-vertices[0][1],vertices[1][0]-vertices[0][0])
            else:
                self._angle = math.atans(vertices[0][1]-vertices[1][1],vertices[0][0]-vertices[1][0])
        else:
            self._width  = edge2
            self._height = edge1
            if(vertices[2][1]<vertices[1][1]):
                self._angle = math.atans(vertices[1][1]-vertices[2][1],vertices[1][0]-vertices[2][0])
            else:
                self._angle = math.atans(vertices[2][1]-vertices[1][1],vertices[2][0]-vertices[1][0])
        # Convert to degree
        self.angle            = (self.angle*180)/math.pi
        self.area             = self._width * self._height
        self.aspect_ratio     = self._width / self._height
        self.center           = (vertices[1]-vertices[3])/2
        self.vertices         = vertices[:] # Create a copy instead of a reference
'''
Intermediate Classes End
'''

'''
Process Classes Begin Template Provided
'''

class GrayImageProc:
    def __call__(self,image):
        return cv.cvtColor(image,cv.COLOR_BGR2GRAY)

class HSVImageProc:
    def __init__(self,enable_debug=True,color = 'blue',ranges=None):
        self.enable_debug = enable_debug
        self._color = color
        if ranges == None:
            if (self._color=='blue'):
                self._ranges = [90,150,46,240,255,255]
            else:
                self._ranges = [170  ,43,46,  3,255,255]
        else:
            self._ranges = ranges
        if enable_debug:
            cv.namedWindow('image_proc')
            self.bars_name = ['h_low','s_low','v_low','h_high','s_high','v_high']
            self._bars = [cv.createTrackbar(self.bars_name[i],'image_proc',0 ,255 if i%3!=0 else 180,void_callback) for i in range(6)]

    def Update(self):
        if self.enable_debug:
            for i in range(6):
                self._ranges[i] = cv.getTrackbarPos(self.bars_name[i],'image_proc')
        else:
            print("Not On debug Mode!")    

    def __call__(self,img):
        self.Update()
        element = cv.getStructuringElement(cv.MORPH_RECT,(3,3))
        img = cv.dilate(img,element,anchor=(-1,-1),iterations=1)
        hsv_img = cv.cvtColor(img,cv.COLOR_BGR2HSV)
        lower = self._ranges[:3]
        upper = self._ranges[3:]
        if(lower[0]>upper[0]):
            thresh1_img = cv.inRange(hsv_img,[0]+lower[1:],upper)
            thresh2_img = cv.inRange(hsv_img,lower,[180]+upper[1:])
            thresh_img = thresh1_img | thresh2_img
        else:
            thresh_img = cv.inRange(hsv_img,lower,upper)
        if(self.enable_debug):
            cv.imshow('thresholded',thresh_img)
        return thresh_img

class BGRImageProc:
    def __init__(self,color='B',threshs=None,enable_debug=True):
        if threshs==None:
            self._threshs = [10,10]
        else:
            self._threshs = threshs
        self._color = color
        self.enable_debug = enable_debug
        if enable_debug:
            cv.createTrackbar('Thresh1','image_proc',0,255,void_callback)
            cv.createTrackbar('Thresh2','image_proc',0,255,void_callback)

    def Update(self):
        self._threshs[0] = cv.getTrackbarPos('Thresh1','image_proc')
        self._threshs[1] = cv.getTrackbarPos('Thresh2','image_proc')
    def __str__(self):
        return "rgb_threshold1: "+str(self._threshs[0])+'\n'+"rgb_threshold2: "+str(self._threshs[1])
    def __call__(self,img):
        # Feature enhance
        self.Update()
        element = cv.getStructuringElement(cv.MORPH_RECT,(3,3))
        img = cv.dilate(img,element,anchor=(-1,-1),iterations=1)
        if(self._color=='B'):
            b_r = cv.subtract(img[:,:,0].img[:,:,2])
            _,b_r = cv.threshold(img,self._threshs[0],255,cv.THRESH_BINARY)
            b_g = cv.subtract(img[:,:,0].img[:,:,1])
            _,b_g = cv.threshold(img,self._threshs[1],255,cv.THRESH_BINARY)
            thresh_img = b_g & b_r
        else:
            r_b = cv.subtract(img[:,:,2].img[:,:,0])
            _,r_b = cv.threshold(img,self._threshs[0],255,cv.THRESH_BINARY)
            r_g = cv.subtract(img[:,:,2].img[:,:,1])
            _,r_g = cv.threshold(img,self._threshs[1],255,cv.THRESH_BINARY)
            thresh_img = r_b & r_g
        if self.enable_debug:
            cv.imshow("Threshed Image",thresh_img)
        return thresh_img
        
class ScreenLightBars:
    def __init__(self,mode="hsv",enable_debug=False):
        # Create Rectangular 
        self._element = cv.getStructuringElement(cv.MORPH_RECT,(3,3))
        self._mode = mode
        cv.createTrackbar("Color","image_proc",0,255,void_callback)
        # Need to define file read action
        self._threshold    = 0
        self._enable_debug = enable_debug

    def Update(self):
        self._threshold = cv.getTrackbarPos('Color','image_proc')

    def __str__(self):
        return "color_thread: "+str(self._threshold)

    def __call__(self,thresh_img,gray_img,src):
        self.Update()
        src = src[:]
        light_bars = []
        brightness = cv.threshold(gray_img,self._threshold,255,cv.THRESH_BINARY)
        light_cnts,_ = cv.findContours(brightness,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
        color_cnts,_ = cv.findContours(thresh_img,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
        for i in light_cnts:
            for j in color_cnts:
                if(cv.pointPolygonTest(j,i[0],False)>=0.0):
                    single_light = cv.minAreaRect(i)
                    vertices = cv.boxPoints(single_light) # corner points
                    new_lb = LightBar(vertices)
                    single_light[2] = new_lb.angle # Modify the angle
                    light_bars.append(single_light)
                    if self._enable_debug:
                        src = DrawRotatedRect(src,single_light)
        if(self._enable_debug):
            cv.imshow('lightbars',src)
        return light_bars

if __name__ == "__main__":
    formatPrint('thresholds',['item1: 100','item2: 99','item3: 190'],'demo.prototxt')

            