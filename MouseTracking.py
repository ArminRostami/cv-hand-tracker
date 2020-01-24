import pyautogui
import numpy as np
import cv2

HUE_MAX = 180
L = 256
ROWS_LEN = 20
COLS_LEN = 20
ROW_START = 0.5
COL_START = 0.8
screenCols, screenRows = pyautogui.size()
pyautogui.PAUSE = 0
pyautogui.FAILSAFE = False
motionPath = []


def initCapture():
    histExists = False
    mouseAction = True
    window = cv2.VideoCapture(0)

    while window.isOpened():

        frame = cv2.flip(window.read()[1], 1)
        keyPressed = cv2.waitKey(1)

        if keyPressed == ord("c"):
            hist = getHist(frame)
            histExists = True
        elif keyPressed == ord("m"):
            mouseAction = True
        elif keyPressed == ord("s"):
            mouseAction = False
        elif keyPressed == ord("q"):
            window.release()
            cv2.destroyAllWindows()
            break

        if histExists:
            coordinates = getCoordinates(frame, hist)
            execAction(coordinates, frame.shape, mouseAction)

        else:
            drawSamplingRect(frame)

        cv2.imshow("Output", frame)


def getCoordinates(frame, hist):
    histMask = getHistMask(frame, hist)
    cv2.imshow("histMask", histMask)
    largestCnt = getLargestCnt(histMask)
    if largestCnt is None:
        return

    coordinates = largestCnt[largestCnt[:, :, 1].argmin()][0]
    return reduceNoise(coordinates)


def execAction(coordinates, shape, mouseAction):
    if mouseAction:
        if coordinates is None:
            return
        col, row = coordinates[0], coordinates[1]
        pyautogui.moveTo(col * screenCols / shape[1], row * screenRows / shape[0])
    else:
        if len(motionPath) >= 2:
            distance = motionPath[-1][1] - motionPath[-2][1]
            pyautogui.scroll(-distance / 2)


def drawSamplingRect(frame):
    rows, cols = frame.shape[0], frame.shape[1]
    irow, icol = int(ROW_START * rows), int(COL_START * cols)
    cv2.rectangle(frame, (icol, irow), (icol + COLS_LEN, irow + ROWS_LEN), (0, L - 1, 0), 1)


def getLargestCnt(histMask):
    grayHistMask = cv2.cvtColor(histMask, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(grayHistMask, 0, L - 1, 0)[1]
    contours = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
    if len(contours) == 0:
        return None
    return max(contours, key=cv2.contourArea)


def reduceNoise(coordinates):
    TOLERANCE = 10
    if len(motionPath) > 0:
        if abs(coordinates[0] - motionPath[-1][0]) < TOLERANCE:
            coordinates[0] = motionPath[-1][0]
        if abs(coordinates[1] - motionPath[-1][1]) < TOLERANCE:
            coordinates[1] = motionPath[-1][1]
    if len(motionPath) < 10:
        motionPath.append(coordinates)
    else:
        motionPath.pop(0)
        motionPath.append(coordinates)
    return coordinates


def getHist(frame):
    rows, cols = frame.shape[0], frame.shape[1]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    roi = np.zeros([ROWS_LEN, COLS_LEN, 3], dtype=hsv.dtype)

    irow = int(ROW_START * rows)
    icol = int(COL_START * cols)
    roi = hsv[irow : irow + ROWS_LEN, icol : icol + COLS_LEN, :]

    hist = cv2.calcHist([roi], [0, 1], None, [HUE_MAX, L], [0, HUE_MAX, 0, L])
    return cv2.normalize(hist, hist, 0, L - 1, cv2.NORM_MINMAX)


def getHistMask(frame, hist):
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
    initCapture()
