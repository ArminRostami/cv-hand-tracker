from actions import move_mouse, scroll
import numpy as np
import cv2

HUE_MAX = 180
L = 256
ROWS_LEN = 20
COLS_LEN = 20
ROW_START = 0.5
COL_START = 0.8
motionPath = []


def main():

    calibrated = False
    mouseAction = True
    window = cv2.VideoCapture(0)
    print(
        """press:
        (C) to calibrate (move your hand to the blue rectangle)
        (S) to switch to scroll mode
        (M) to switch to mouse mode
        (X) to exit\n"""
    )
    while True:

        frame = cv2.flip(window.read()[1], 1)  # flip the image because it is mirrored in webcam
        key = cv2.waitKey(1)

        if pressed(key, "x"):
            print("X pressed. exiting...")
            cv2.destroyAllWindows()
            break
        elif pressed(key, "c"):
            hist = get_histogram(frame)
            calibrated = True
            print("Calibration complete. Moving mouse with hand motions...")
        elif pressed(key, "m"):
            mouseAction = True
            print("Switched to mouse mode.")
        elif pressed(key, "s"):
            mouseAction = False
            print("Switched to scroll mode.")

        if calibrated:
            cv2.destroyWindow("Window")
            run(mouseAction, hist, frame)
        else:
            draw_sampling_rectangle(frame)
            cv2.imshow("Window", frame)


def pressed(key, expected):
    return key == ord(expected)


def run(action, hist, frame):
    mask = get_mask(frame, hist)
    cv2.imshow("Mask", mask)
    coordinates = get_coordinates(mask)
    if coordinates is not None:
        add_to_path(coordinates)
        perform_action(coordinates, frame.shape, action)


def get_coordinates(mask):
    largestCnt = get_largest_cnt(mask)
    if largestCnt is None:
        return None
    coordinates = largestCnt[0][0]
    return reduce_noise(coordinates)


def perform_action(coordinates, shape, mouseAction):
    if mouseAction:
        move_mouse(coordinates, shape)
    else:
        scroll(motionPath)


def draw_sampling_rectangle(frame):
    rows, cols = frame.shape[0], frame.shape[1]
    irow, icol = int(ROW_START * rows), int(COL_START * cols)
    blue = (255, 0, 0)
    thickness = 3
    cv2.rectangle(frame, (icol, irow), (icol + COLS_LEN, irow + ROWS_LEN), blue, thickness)


def get_largest_cnt(mask):
    gray_Mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray_Mask, 0, L - 1, 0)[1]
    contours = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
    if len(contours) == 0:
        return None
    return max(contours, key=cv2.contourArea)


def reduce_noise(coordinates):
    tolerance = 10
    if len(motionPath) > 0:
        lastPoint = motionPath[-1]
        lastCol, lastRow = lastPoint[0], lastPoint[1]
        coordinates[0] = lastCol if abs(coordinates[0] - lastCol) < tolerance else coordinates[0]
        coordinates[1] = lastRow if abs(coordinates[1] - lastRow) < tolerance else coordinates[1]
    return coordinates


def add_to_path(coordinates):
    maxSize = 2
    if len(motionPath) >= maxSize:
        motionPath.pop(0)
    motionPath.append(coordinates)


def get_histogram(frame):
    rows, cols = frame.shape[0], frame.shape[1]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    irow = int(ROW_START * rows)
    icol = int(COL_START * cols)
    hsv = hsv[irow : irow + ROWS_LEN, icol : icol + COLS_LEN, :]

    hist = cv2.calcHist(
        [hsv], channels=[0, 1], mask=None, histSize=[HUE_MAX, L], ranges=[0, HUE_MAX, 0, L]
    )
    return cv2.normalize(hist, hist, 0, L - 1, cv2.NORM_MINMAX)


def get_mask(frame, hist):
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    proj = cv2.calcBackProject(
        [hsv_frame], channels=[0, 1], hist=hist, ranges=[0, HUE_MAX, 0, L], scale=1
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (29, 29))
    proj = cv2.filter2D(proj, -1, kernel)

    proj = cv2.threshold(proj, 150, L - 1, cv2.THRESH_BINARY)[1]

    kernel = np.ones((5, 5), np.uint8)
    proj = cv2.morphologyEx(proj, cv2.MORPH_CLOSE, kernel, iterations=5)

    proj_3d = cv2.merge((proj, proj, proj))
    mask = cv2.bitwise_and(frame, proj_3d)

    return mask


if __name__ == "__main__":
    main()
