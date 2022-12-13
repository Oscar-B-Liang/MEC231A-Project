import glob
import cv2
import numpy as np


THRESHOLD = 176
SAMPLE_POINTS = 32
WINDOW_SIZE = 15


def bilinear_probe(_gray_img, x0, y0, _theta, r):
    dist, direction = 0, -1
    while dist < WINDOW_SIZE:
        cur_dist = r + dist * direction
        _x, _y = int(x0 + np.cos(_theta) * cur_dist), int(y0 + np.sin(_theta) * cur_dist)
        if _gray_img[_y, _x] < THRESHOLD:
            return r + dist * direction
        if direction == -1:
            dist += 1
            direction = 1
        else:
            direction = -1
    raise ValueError


def binary_search(_gray_img, x0, y0, _theta, lo, hi, mode):
    c, s = np.cos(_theta), np.sin(_theta)
    lo, hi = int(lo), int(hi)
    assert lo <= hi

    while lo < hi:
        mi = (lo + hi) >> 1
        _x, _y = int(x0 + c * mi), int(y0 + s * mi)
        if (_gray_img[_y, _x] < THRESHOLD) ^ (mode == 'INF'):
            lo = mi + 1
        else:
            hi = mi
    return lo


def put_cross(cropped, _r):
    _cross_param = _r >> 2
    cv2.line(cropped, (_r - _cross_param, _r - _cross_param), (_r + _cross_param, _r + _cross_param),
             (255, 255, 255), 1)
    cv2.line(cropped, (_r + _cross_param, _r - _cross_param), (_r - _cross_param, _r + _cross_param),
             (255, 255, 255), 1)


def process_image():
    files = glob.glob('../data/circles/*/*.jpg')
    cv2.namedWindow('output', cv2.WINDOW_NORMAL)
    cv2.namedWindow('cropped', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('output', 800, 450)

    for file in files:
        img = cv2.imread(file)
        # rows, cols, _ = img.shape
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        circles = cv2.HoughCircles(gray_img, cv2.HOUGH_GRADIENT, 1, 20, param1=100, param2=60, minRadius=220,
                                   maxRadius=300)

        if circles is not None:
            # circles = circles.astype(int)
            x, y, r = np.mean(circles[0, :3, :], axis=0).astype(int)
            cv2.circle(gray_img, (x, y), 2, (0, 0, 255), 3)
            theta = np.linspace(start=-1 / 4 * np.pi, stop=1 / 2 * np.pi, num=SAMPLE_POINTS)
            sample_center = np.vstack((y + np.sin(theta) * r, x + np.cos(theta) * r)).T.astype(int)

            for _x, _y in sample_center:
                cropped = gray_img[_x - WINDOW_SIZE: _x + WINDOW_SIZE, _y - WINDOW_SIZE: _y + WINDOW_SIZE]
                # put_cross(cropped, _r)
                cv2.imshow("cropped", cropped)
                cv2.waitKey(10)

            try:
                rs = [bilinear_probe(gray_img, x, y, t, r) for t in theta]
            except ValueError:
                print(file, "Invalid Center. ")
                continue
            diff = np.asarray([binary_search(gray_img, x, y, t, r, r * 1.5, 'SUP')
                               - binary_search(gray_img, x, y, t, r * 0.75, r, 'INF') for t, r in zip(theta, rs)])
            width = np.mean(diff)
            std = np.std(diff)
            if std < width / 4:
                print(file, "Average Width: {} (std={})".format(width, std))
            else:
                print(file, "Noisy Data (mean={}, std={})".format(width, std))
        else:
            print(file, "Stroke Not Found")

        cv2.imshow('output', gray_img)
        cv2.waitKey(100)

    cv2.destroyAllWindows()


if __name__ == '__main__':
    process_image()
