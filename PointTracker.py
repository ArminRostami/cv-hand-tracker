import pyautogui
import numpy as np
import cv2

HUE_MAX = 180
L = 256
ROWS_LEN = 20
COLS_LEN = 20
ROW_START = 0.5
COL_START = 0.8
mouseMode = True
histExists = False
screenCols, screenRows = pyautogui.size()
pyautogui.PAUSE = 0
pyautogui.FAILSAFE = False
traversePoints = []


def getHist(frame):
    global histExists
    histExists = True
    rows, cols = frame.shape[0], frame.shape[1]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    roi = np.zeros([ROWS_LEN, COLS_LEN, 3], dtype=hsv.dtype)

    irow = int(ROW_START * rows)
    icol = int(COL_START * cols)
    roi = hsv[irow : irow + ROWS_LEN, icol : icol + COLS_LEN, :]

    hist = cv2.calcHist([roi], [0, 1], None, [HUE_MAX, L], [0, HUE_MAX, 0, L])
    return cv2.normalize(hist, hist, 0, L - 1, cv2.NORM_MINMAX)


def drawSamplingRect(frame):
    rows, cols = frame.shape[0], frame.shape[1]
    irow, icol = int(ROW_START * rows), int(COL_START * cols)
    cv2.rectangle(frame, (icol, irow), (icol + COLS_LEN, irow + ROWS_LEN), (0, L - 1, 0), 1)


def detect(frame, hist):
    histMask = getHistMask(frame, hist)
    cv2.imshow("histMask", histMask)
    largestContour = getLargestContour(histMask)
    if largestContour is None:
        return

    farthestPoint = largestContour[largestContour[:, :, 1].argmin()][0]
    if len(traversePoints) > 0:
        if abs(farthestPoint[0] - traversePoints[-1][0]) < 10:
            farthestPoint[0] = traversePoints[-1][0]
        if abs(farthestPoint[1] - traversePoints[-1][1]) < 10:
            farthestPoint[1] = traversePoints[-1][1]
    farthestPoint = tuple(farthestPoint)

    cv2.circle(frame, farthestPoint, 5, [0, 0, L - 1], -1)

    if len(traversePoints) < 10:
        traversePoints.append(farthestPoint)
    else:
        traversePoints.pop(0)
        traversePoints.append(farthestPoint)

    drawPath(frame, traversePoints)
    execute(farthestPoint, frame)


def execute(farthestPoint, frame):
    if mouseMode:
        targetX = farthestPoint[0]
        targetY = farthestPoint[1]
        pyautogui.moveTo(
            targetX * screenCols / frame.shape[1], targetY * screenRows / frame.shape[0]
        )
    else:
        if len(traversePoints) >= 2:
            movedDistance = traversePoints[-1][1] - traversePoints[-2][1]
            pyautogui.scroll(-movedDistance / 2)


def drawPath(frame, traversePoints):
    for i in range(1, len(traversePoints)):
        thickness = int((i + 1) / 2)
        cv2.line(frame, traversePoints[i - 1], traversePoints[i], [0, 0, L - 1], thickness)


def getHistMask(frame, hist):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    dst = cv2.calcBackProject([hsv], [0, 1], hist, [0, HUE_MAX, 0, L], 1)

    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    cv2.filter2D(dst, -1, disc, dst)

    thresh = cv2.threshold(dst, 150, L - 1, cv2.THRESH_BINARY)[1]

    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=5)

    merged = cv2.merge((thresh, thresh, thresh))
    return cv2.bitwise_and(frame, merged)


def getLargestContour(histMask):
    grayHistMask = cv2.cvtColor(histMask, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(grayHistMask, 0, L - 1, 0)[1]
    contours = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
    if len(contours) == 0:
        return None
    return max(contours, key=cv2.contourArea)


def initCapture():
    global mouseMode, histExists
    capture = cv2.VideoCapture(0)

    while capture.isOpened():
        frame = capture.read()[1]
        frame = cv2.flip(frame, 1)
        k = cv2.waitKey(1) & 0xFF

        if k == ord("c"):
            hist = getHist(frame)
        elif k == ord("m"):
            mouseMode = True
        elif k == ord("s"):
            mouseMode = False
        elif k == ord("q"):
            capture.release()
            cv2.destroyAllWindows()
            break

        if histExists:
            detect(frame, hist)
        else:
            drawSamplingRect(frame)

        cv2.imshow("Output", frame)


if __name__ == "__main__":
    initCapture()
