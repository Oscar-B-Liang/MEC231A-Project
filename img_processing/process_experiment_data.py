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


def binary_search(img, x0, y0, theta, lo, hi, mode):
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    lo, hi = int(lo), int(hi)
    assert lo <= hi

    while lo < hi:
        mi = (lo + hi) >> 1
        _x, _y = int(x0 + cos_t * mi), int(y0 + sin_t * mi)
        if (img[_y, _x] < THRESHOLD) ^ (mode == 'INF'):
            lo = mi + 1
        else:
            hi = mi
    return lo


def _put_cross(cropped, r):
    size = r >> 2
    cv2.line(cropped, (r - size, r - size), (r + size, r + size), (255, 255, 255), 1)
    cv2.line(cropped, (r + size, r - size), (r - size, r + size), (255, 255, 255), 1)


def window_scan(gray_img, x0, y0, theta, r):
    # Sanity check: the area near the sampling center. Make sure the stroke is in the window!
    sample_center = np.vstack((y0 + np.sin(theta) * r, x0 + np.cos(theta) * r)).T.astype(int)
    for x, y in sample_center:
        cropped = gray_img[x - WINDOW_SIZE: x + WINDOW_SIZE, y - WINDOW_SIZE: y + WINDOW_SIZE]
        _put_cross(cropped, r)
        cv2.imshow("cropped", cropped)
        cv2.waitKey(10)


def process_circle_image():
    files = glob.glob('../data/new_data/*/*.jpg')
    cv2.namedWindow('output', cv2.WINDOW_NORMAL)
    cv2.namedWindow('cropped', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('output', 800, 450)

    log = {}

    for file in files:
        img = cv2.imread(file)

        # Using Hough Circles to determine the location of the arc.
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        circles = cv2.HoughCircles(gray_img, cv2.HOUGH_GRADIENT, 1, 20, param1=100, param2=60, minRadius=220,
                                   maxRadius=300)
        cv2.imshow('output', gray_img)
        cv2.waitKey(10)
        if circles is None:
            print(file, "Stroke Not Found")
            continue

        # Select the mean value of the 3 guess of the circle with the highest likelihood.
        x0, y0, r = np.mean(circles[0, :3, :], axis=0).astype(int)
        cv2.circle(gray_img, (x0, y0), 2, (0, 0, 255), 3)

        # Uniformly sample <SAMPLE_POINTS> points from the arc.
        theta = np.linspace(start=-1 / 4 * np.pi, stop=1 / 2 * np.pi, num=SAMPLE_POINTS)

        # Determine the stroke area using binary search with calibrated center line.
        try:
            rs = np.asarray([bilinear_probe(gray_img, x0, y0, t, r) for t in theta])
            window_scan(gray_img, x0, y0, theta, r)
        except ValueError:
            print(file, "Invalid Center. ")
            continue
        sup = np.asarray([binary_search(gray_img, x0, y0, t, lo=r, hi=r * 1.5, mode='SUP') for t, r in zip(theta, rs)])
        inf = np.asarray([binary_search(gray_img, x0, y0, t, lo=r * 0.75, hi=r, mode='INF') for t, r in zip(theta, rs)])

        # Derive the width of the stroke.
        diff = sup - inf
        width = np.mean(diff)
        std = np.std(diff)

        # Derive the shade of the stroke
        shades = np.empty(SAMPLE_POINTS, dtype=np.float32)
        for i, (t, lo, hi) in enumerate(zip(theta, inf, sup)):
            r = np.linspace(lo, hi, SAMPLE_POINTS)
            xs = (x0 + r * np.cos(t)).astype(int)
            ys = (y0 + r * np.sin(t)).astype(int)
            shades[i] = sum(gray_img[y, x] for x, y in zip(xs, ys)) / SAMPLE_POINTS
        shade = np.mean(shades)
        if std < width / 4:
            print(file, "\nAverage Width: {} (std={}), Average Shade: {}".format(width, std, shade))
        else:
            print(file, "\nNoisy Data (mean={}, std={}, shade={})".format(width, std, shade))

        log[file] = {
            'width': width,
            'std': std,
            'shade': shade
        }

    import pickle as pkl
    with open("new_data.pkl", "wb") as f:
        pkl.dump(log, f)

    cv2.destroyAllWindows()
    return log


if __name__ == '__main__':
    process_circle_image()
