import pyautogui
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
    pyautogui.PAUSE = 0  # seconds to pause after function calls
    pyautogui.FAILSAFE = False  # allows the mouse to exit the window
    calibrated = False
    mouseAction = True
    window = cv2.VideoCapture(0)
    print(
        """press:
        (C) to calibrate (move your hand to the green rectangle)
        (S) to switch to scroll mode
        (M) to switch to scroll mode
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
            hist = getHist(frame)
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
            drawSamplingRect(frame)
            cv2.imshow("Window", frame)


def pressed(key, expected):
    return key == ord(expected)


def run(action, hist, frame):
    mask = getMask(frame, hist)
    cv2.imshow("Mask", mask)
    coordinates = getCoordinates(mask)
    if coordinates is not None:
        doAction(coordinates, frame.shape, action)


def getCoordinates(mask):
    largestCnt = getLargestCnt(mask)
    if largestCnt is None:
        return None

    coordinates = largestCnt[largestCnt[:, :, 1].argmin()][0]
    return reduceNoise(coordinates)


def doAction(coordinates, shape, mouseAction):
    if mouseAction:
        moveMouse(coordinates, shape)
    else:
        scroll()


def moveMouse(coordinates, shape):
    screenCols, screenRows = pyautogui.size()
    col, row = coordinates[0], coordinates[1]
    pyautogui.moveTo(col * screenCols / shape[1], row * screenRows / shape[0])


def scroll():
    if len(motionPath) <= 1:
        return
    distance = motionPath[-1][1] - motionPath[-2][1]
    pyautogui.scroll(-distance / 2)


def drawSamplingRect(frame):
    rows, cols = frame.shape[0], frame.shape[1]
    irow, icol = int(ROW_START * rows), int(COL_START * cols)
    cv2.rectangle(frame, (icol, irow), (icol + COLS_LEN, irow + ROWS_LEN), (0, L - 1, 0), 1)


def getLargestCnt(mask):
    gray_Mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray_Mask, 0, L - 1, 0)[1]
    contours = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
    if len(contours) == 0:
        return None
    return max(contours, key=cv2.contourArea)


def reduceNoise(coordinates):
    tolerance = 10
    if len(motionPath) > 0:
        lastPoint = motionPath[-1]
        lastCol, lastRow = lastPoint[0], lastPoint[1]
        if abs(coordinates[0] - lastCol) < tolerance:
            coordinates[0] = lastCol
        if abs(coordinates[1] - lastRow) < tolerance:
            coordinates[1] = lastRow
    addToPath(coordinates)
    return coordinates


def addToPath(coordinates):
    maxSize = 2
    if len(motionPath) < maxSize:
        motionPath.append(coordinates)
    else:
        motionPath.pop(0)
        motionPath.append(coordinates)


def getHist(frame):
    rows, cols = frame.shape[0], frame.shape[1]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    roi = np.zeros([ROWS_LEN, COLS_LEN, 3], dtype=hsv.dtype)

    irow = int(ROW_START * rows)
    icol = int(COL_START * cols)
    roi = hsv[irow : irow + ROWS_LEN, icol : icol + COLS_LEN, :]

    hist = cv2.calcHist([roi], [0, 1], None, [HUE_MAX, L], [0, HUE_MAX, 0, L])
    return cv2.normalize(hist, hist, 0, L - 1, cv2.NORM_MINMAX)


def getMask(frame, hist):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    dst = cv2.calcBackProject([hsv], [0, 1], hist, [0, HUE_MAX, 0, L], 1)

    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    dst = cv2.filter2D(dst, -1, disc)

    thresh = cv2.threshold(dst, 150, L - 1, cv2.THRESH_BINARY)[1]

    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=5)

    merged = cv2.merge((thresh, thresh, thresh))
    return cv2.bitwise_and(frame, merged)


if __name__ == "__main__":
    main()
