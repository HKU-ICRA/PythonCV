import cv2 as cv
import numpy as np
import math

'''
Helper functions begin
'''


def void_callback(x):
    pass


def DrawRotatedRect(img, rect, color=(0, 255, 0), thickness=2):
    center = rect[0]
    angle = rect[2]
    font = cv.FONT_HERSHEY_COMPLEX
    cv.putText(img, str(angle), center, font, 0.5, color, thickness, 8, 0)
    vertices = cv.boxPoints(rect)
    for i in range(4):
        cv.line(img, vertices[i], vertices[(i + 1) % 4], color, thickness)
    return img


def formatPrint(title, items, filename):
    try:
        file = open(filename, 'w')
    except:
        print('cannot Open the file')
        return
    file.writelines(title + ' {' + '\n')
    for i in items:
        file.writelines('  ' + str(i) + '\n')
    file.writelines('}' + '\n')
    file.close()


def pointCmp(p1, p2):
    if p1[0] > p2[0]:
        return 1
    elif p1[0] == p2[0]:
        return 0
    else:
        return -1


def armorCmp(a1, a2):
    if a1.area > a2.area:
        return 1
    elif a1.area == a2.area:
        return 0
    else:
        return -1


def solveArmorCoordinate(width, height):
    return [(-width / 2, height / 2, 0.0), (width / 2, height / 2, 0.0), (width/2,  -height/2,  0.0), (-width/2, -height/2, 0.0)]


'''
Helper functions end
'''

'''
Intermediate Classes Begin
'''


class LightBar:
    def __init__(self, vertices):
        # The length of edges
        edge1 = np.linalg.norm(vertices[0] - vertices[1])
        edge2 = np.linalg.norm(vertices[1] - vertices[2])
        if edge1 > edge2:
            self._width = edge1
            self._height = edge2
            if vertices[0][1] < vertices[1][1]:
                self._angle = math.atans(vertices[1][1] - vertices[0][1], vertices[1][0] - vertices[0][0])
            else:
                self._angle = math.atans(vertices[0][1] - vertices[1][1], vertices[0][0] - vertices[1][0])
        else:
            self._width = edge2
            self._height = edge1
            if vertices[2][1] < vertices[1][1]:
                self._angle = math.atans(vertices[1][1] - vertices[2][1], vertices[1][0] - vertices[2][0])
            else:
                self._angle = math.atans(vertices[2][1] - vertices[1][1], vertices[2][0] - vertices[1][0])
        # Convert to degree
        self.angle = (self.angle * 180) / math.pi
        self.area = self._width * self._height
        self.aspect_ratio = self._width / self._height
        self.center = (vertices[1] - vertices[3]) / 2
        self.vertices = vertices[:]  # Create a copy instead of a reference


class Armor:
    def __init__(self, armor_rect, armor_vertex, armor_stddev=0.0):
        self.rect = armor_rect
        self.vertex = armor_vertex
        self.stddev = armor_stddev
        self.area = self.armor_rect[1][0] * self.armor_rect[1][1]


'''
Intermediate Classes End
'''

'''
Process Classes Begin Template Provided
'''


class GrayImageProc:
    def __call__(self, image):
        return cv.cvtColor(image, cv.COLOR_BGR2GRAY)


class HSVImageProc:
    def __init__(self, enable_debug=True, color='blue', ranges=None):
        self.enable_debug = enable_debug
        self._color = color
        if ranges is None:
            if self._color == 'blue':
                self._ranges = [90, 150, 46, 240, 255, 255]
            else:
                self._ranges = [170, 43, 46, 3, 255, 255]
        else:
            self._ranges = ranges
        if enable_debug:
            cv.namedWindow('image_proc')
            self.bars_name = ['h_low', 's_low', 'v_low', 'h_high', 's_high', 'v_high']
            self._bars = [
                cv.createTrackbar(self.bars_name[i], 'image_proc', 0, 255 if i % 3 != 0 else 180, void_callback) for i
                in range(6)]

    def Update(self):
        if self.enable_debug:
            for i in range(6):
                self._ranges[i] = cv.getTrackbarPos(self.bars_name[i], 'image_proc')
        else:
            print("Not On debug Mode!")

    def __call__(self, img):
        self.Update()
        element = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
        img = cv.dilate(img, element, anchor=(-1, -1), iterations=1)
        hsv_img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        lower = self._ranges[:3]
        upper = self._ranges[3:]
        if lower[0] > upper[0]:
            thresh1_img = cv.inRange(hsv_img, [0] + lower[1:], upper)
            thresh2_img = cv.inRange(hsv_img, lower, [180] + upper[1:])
            thresh_img = thresh1_img | thresh2_img
        else:
            thresh_img = cv.inRange(hsv_img, lower, upper)
        if self.enable_debug:
            cv.imshow('thresholded', thresh_img)
        return thresh_img


class BGRImageProc:
    def __init__(self, color='B', threshs=None, enable_debug=True):
        if threshs is None:
            self._threshs = [10, 10]
        else:
            self._threshs = threshs
        self._color = color
        self.enable_debug = enable_debug
        if enable_debug:
            cv.createTrackbar('Thresh1', 'image_proc', 0, 255, void_callback)
            cv.createTrackbar('Thresh2', 'image_proc', 0, 255, void_callback)

    def Update(self):
        self._threshs[0] = cv.getTrackbarPos('Thresh1', 'image_proc')
        self._threshs[1] = cv.getTrackbarPos('Thresh2', 'image_proc')

    def __str__(self):
        return "rgb_threshold1: " + str(self._threshs[0]) + '\n' + "rgb_threshold2: " + str(self._threshs[1])

    def __call__(self, img):
        # Feature enhance
        self.Update()
        element = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
        img = cv.dilate(img, element, anchor=(-1, -1), iterations=1)
        if self._color == 'B':
            b_r = cv.subtract(img[:, :, 0].img[:, :, 2])
            _, b_r = cv.threshold(img, self._threshs[0], 255, cv.THRESH_BINARY)
            b_g = cv.subtract(img[:, :, 0].img[:, :, 1])
            _, b_g = cv.threshold(img, self._threshs[1], 255, cv.THRESH_BINARY)
            thresh_img = b_g & b_r
        else:
            r_b = cv.subtract(img[:, :, 2].img[:, :, 0])
            _, r_b = cv.threshold(img, self._threshs[0], 255, cv.THRESH_BINARY)
            r_g = cv.subtract(img[:, :, 2].img[:, :, 1])
            _, r_g = cv.threshold(img, self._threshs[1], 255, cv.THRESH_BINARY)
            thresh_img = r_b & r_g
        if self.enable_debug:
            cv.imshow("Threshed Image", thresh_img)
        return thresh_img


class ScreenLightBars:
    def __init__(self, mode="hsv", enable_debug=False):
        # Create Rectangular 
        self._element = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
        self._mode = mode
        cv.createTrackbar("Color", "image_proc", 0, 255, void_callback)
        # Need to define file read action
        self._threshold = 0
        self._enable_debug = enable_debug

    def Update(self):
        self._threshold = cv.getTrackbarPos('Color', 'image_proc')

    def __str__(self):
        return "color_thread: " + str(self._threshold)

    def __call__(self, thresh_img, gray_img, src):
        self.Update()
        src = src[:]
        light_bars = []
        brightness = cv.threshold(gray_img, self._threshold, 255, cv.THRESH_BINARY)
        light_cnts, _ = cv.findContours(brightness, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        color_cnts, _ = cv.findContours(thresh_img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        for i in light_cnts:
            for j in color_cnts:
                if cv.pointPolygonTest(j, i[0], False) >= 0.0:
                    single_light = cv.minAreaRect(i)
                    vertices = cv.boxPoints(single_light)  # corner points
                    new_lb = LightBar(vertices)
                    single_light[2] = new_lb.angle  # Modify the angle
                    light_bars.append(single_light)
                    if self._enable_debug:
                        src = DrawRotatedRect(src, single_light)
        if self._enable_debug:
            cv.imshow('light_bars', src)
        return light_bars


class FilterLightBars:
    def __init__(self, light_max_aspect_ratio, light_min_area, enable_debug=False):
        self._light_max_aspect_ratio = light_max_aspect_ratio
        self._light_min_area = light_min_area
        self._enable_debug = enable_debug

    def __call__(self, light_bars, src):
        rects = []
        for light_bar in light_bars:
            vertices = cv.boxPoints(light_bar)
            new_lb = LightBar(vertices)
            area = new_lb.area
            width = light_bar[1][0]
            height = light_bar[1][1]
            light_aspect_ratio = max(width, height) / min(width, height)
            if light_aspect_ratio < self._light_max_aspect_ratio and area >= self._light_min_area:
                rects.append(light_bar)
                if self._enable_debug:
                    src = DrawRotatedRect(src, light_bar)
        if self._enable_debug:
            cv.imshow('light_bars_filtered', src)
        return rects


class PossibleArmors:
    def __init__(self, light_max_angle_diff, armor_max_aspect_ratio, armor_min_area, armor_max_pixel_val,
                 enable_debug=False):
        self._light_max_angle_diff = light_max_angle_diff
        self._armor_max_aspect_ratio = armor_max_aspect_ratio
        self._armor_min_area = armor_min_area
        self._armor_max_pixel_val = armor_max_pixel_val
        self._enable_debug = enable_debug

    def calcArmorInfo(self, left_light, right_light):
        armor_points = []
        left_points = cv.boxPoints(left_light)
        right_points = cv.boxPoints(right_light)
        left_points.sort(cmp=pointCmp)
        right_points.sort(cmp=pointCmp)
        if right_points[0][1] < right_points[1][1]:
            right_lu = right_points[0]
            right_ld = right_points[1]
        else:
            right_lu = right_points[1]
            right_ld = right_points[0]

        if left_points[2][1] < left_points[3][1]:
            lift_ru = left_points[2]
            lift_rd = left_points[3]
        else:
            lift_ru = left_points[3]
            lift_rd = left_points[2]
        armor_points.append(lift_ru)
        armor_points.append(right_lu)
        armor_points.append(right_ld)
        armor_points.append(lift_rd)
        return armor_points

    def __call__(self, rects, src):
        armors = []
        for i in range(len(rects)):
            for j in range(i + 1, len(rects)):
                rect1 = rects[i]
                rect2 = rects[j]
                edge1min = min(rect1[1][0], rect1[1][1])
                edge1max = max(rect1[1][0], rect1[1][1])
                edge2min = min(rect2[1][0], rect2[1][1])
                edge2max = max(rect2[1][0], rect2[1][1])
                lights_dis = math.sqrt(math.pow(rect1[0][0] - rect2[0][0], 2) + math.pow(rect1[0][1] - rect2[0][1], 2))
                center_angle = math.atan2(abs(rect1[0][1] - rect2[0][1]), abs(rect1[0][0] - rect2[0][0])) * 180 / np.pi
                if center_angle > 90:
                    center_angle = 180 - center_angle

                x = (rect1[0][0] + rect2[0][0]) / 2
                y = (rect1[0][1] + rect2[0][1]) / 2
                width = abs(lights_dis - max(edge1min, edge2min))
                height = max(edge1max, edge2max)
                rect_width = max(width, height)
                rect_height = min(width, height)
                rect = ((x, y), (rect_width, rect_height), center_angle)

                rect1_angle = rect1[2]
                rect2_angle = rect2[2]

                radio = max(edge1max, edge2max) / min(edge1max, edge2max)
                armor_aspect_ratio = rect_width / rect_height
                armor_area = rect_width * rect_height
                armor_pixel_val = src[y, x]
                if self._enable_debug:
                    print("*******************************")
                    print("light_angle_diff_:", abs(rect1_angle - rect2_angle))
                    print("radio:", radio)
                    print("armor_angle_:", abs(center_angle))
                    print("armor_aspect_ratio_:", armor_aspect_ratio)
                    print("armor_area_:", armor_area)
                    print("armor_pixel_val_:", src[y, x])
                    print("pixel_y", y)
                    print("pixel_x", x)

                angle_diff = abs(rect1_angle - rect2_angle)
                if angle_diff > 175:
                    angle_diff = 180 - angle_diff

                if angle_diff < self._light_max_angle_diff and radio < 2.0 and armor_aspect_ratio < self._armor_max_aspect_ratio and armor_area > self._armor_min_area and armor_pixel_val < self._armor_max_pixel_val:
                    if rect1[0][0] < rect2[0][0]:
                        armor_points = self.calcArmorInfo(rect1, rect2)
                        armors.append(Armor(rect, armor_points))
                        if self._enable_debug:
                            DrawRotatedRect(src, rect)
                    else:
                        armor_points = self.calcArmorInfo(rect2, rect1)
                        armors.append(Armor(rect, armor_points))
                        if self._enable_debug:
                            DrawRotatedRect(src, rect)
        if self._enable_debug:
            cv.imshow('armors', src)
        return armors


class FilterArmors:
    def __init__(self, armor_max_stddev, armor_max_mean, enable_debug=False):
        self._armor_max_stddev = armor_max_stddev
        self._armor_max_mean = armor_max_mean
        self._enable_debug = enable_debug

    def __call__(self, armors, src):
        filtered_armors = []
        mask = np.zeros_like(src, cv.CV_8UC1)
        for armor in armors:
            pts = []
            for i in range(4):
                pts.append(armor.vertex[i])
            cv.fillConvexPoly(mask, pts, (0, 255, 0))
            (mean, stddev) = cv.meanStdDev(src)
            if stddev <= self._armor_max_stddev and mean <= self._armor_max_mean:
                filtered_armors.append(armor)

        is_armor = [True for i in filtered_armors]
        for i in range(len(filtered_armors)):
            if is_armor[i]:
                for j in range(i + 1, len(filtered_armors)):
                    if is_armor[j]:
                        dx = filtered_armors[i].rect[0][0] - filtered_armors[j].rect[0][0]
                        dy = filtered_armors[i].rect[0][1] - filtered_armors[j].rect[0][1]
                        dis = math.sqrt(dx * dx + dy * dy)
                        if dis < filtered_armors[i].rect[1][0] + filtered_armors[j].rect[1][0]:
                            if filtered_armors[i].rect[2] > filtered_armors[j].rect[2]:
                                is_armor[i] = False
                            else:
                                is_armor[j] = False

        new_filtered_armors = []
        for i in range(len(filtered_armors)):
            if is_armor[i]:
                new_filtered_armors.append(filtered_armors[i])
                if self._enable_debug:
                    DrawRotatedRect(src, filtered_armors[i].rect)

        if self._enable_debug:
            cv.imshow('armors_filtered', src)

        return new_filtered_armors


class SelectFinalArmor:
    def __init__(self, enable_debug=False):
        self._enable_debug = enable_debug

    def __call__(self, armors, src):
        armors.sort(cmp=armorCmp)
        if self._enable_debug:
            DrawRotatedRect(src, armors[0].rect)
            cv.imshow('final_armor', src)
        return armors[0]


class CalcControlInfo:
    def __init__(self, armor_points, intrinsic_matrix, distortion_coeffs, enable_debug=False):
        self._armor_points = armor_points
        self._intrinsic_matrix = intrinsic_matrix
        self._distortion_coeffs = distortion_coeffs
        self._enable_debug = enable_debug

    def __call__(self, armor):
        _, rvec, tvec = cv.solvePnP(self._armor_points, armor.vertex, self._intrinsic_matrix, self._distortion_coeffs)
        if self._enable_debug:
            print("rotation vector:", rvec)
            print("translation vector:", tvec)
        return tvec


if __name__ == "__main__":
    formatPrint('thresholds', ['item1: 100', 'item2: 99', 'item3: 190'], 'demo.prototxt')
