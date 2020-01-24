import pyautogui
import numpy as np
import cv2

HUE_MAX = 180
Y_START = 20
X_START = 20
ROW_START = 0.5
COL_START = 0.8
L = 256
mouseMode = True
histExists = False
screenSizeX, screenSizeY = pyautogui.size()
pyautogui.PAUSE = 0
pyautogui.FAILSAFE = False
traversePoints = []


def createHistogram(frame):
    rows, cols, _ = frame.shape
    hsvFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    roi = np.zeros([Y_START, X_START, 3], dtype=hsvFrame.dtype)

    y0, x0 = int(ROW_START * rows), int(COL_START * cols)
    roi = hsvFrame[y0 : y0 + Y_START, x0 : x0 + X_START, :]

    hist = cv2.calcHist([roi], [0, 1], None, [HUE_MAX, L], [0, HUE_MAX, 0, L])
    return cv2.normalize(hist, hist, 0, L - 1, cv2.NORM_MINMAX)


def drawLocker(frame):
    rows, cols, _ = frame.shape

    y0, x0 = int(ROW_START * rows), int(COL_START * cols)
    cv2.rectangle(frame, (x0, y0), (x0 + X_START, y0 + Y_START), (0, L - 1, 0), 1)


def detect(frame, hist):
    # global traversePoints
    histMask = histMasking(frame, hist)
    cv2.imshow("histMask", histMask)
    contours = getContours(histMask)
    maxContour = getMaxContours(contours)

    centroid = getCentroid(maxContour)
    cv2.circle(frame, centroid, 5, [L - 1, 0, 0], -1)

    if maxContour is not None:
        farthestPoint = maxContour[maxContour[:, :, 1].argmin()][0]
        if farthestPoint is not None:
            # Reduce noise in farthestPoint
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
    global mouseMode, screenSizeX, screenSizeY
    if mouseMode:
        targetX = farthestPoint[0]
        targetY = farthestPoint[1]
        pyautogui.moveTo(
            targetX * screenSizeX / frame.shape[1], targetY * screenSizeY / frame.shape[0]
        )
    else:
        if len(traversePoints) >= 2:
            movedDistance = traversePoints[-1][1] - traversePoints[-2][1]
            pyautogui.scroll(-movedDistance / 2)


def drawPath(frame, traversePoints):
    for i in range(1, len(traversePoints)):
        thickness = int((i + 1) / 2)
        cv2.line(frame, traversePoints[i - 1], traversePoints[i], [0, 0, L - 1], thickness)


def histMasking(frame, hist):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    dst = cv2.calcBackProject([hsv], [0, 1], hist, [0, HUE_MAX, 0, L], 1)

    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    cv2.filter2D(dst, -1, disc, dst)

    ret, thresh = cv2.threshold(dst, 150, L - 1, cv2.THRESH_BINARY)

    kernel = np.ones((5, 5), np.uint8)
    # thresh = cv2.dilate(thresh, kernel, iterations=5)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=5)

    thresh = cv2.merge((thresh, thresh, thresh))
    return cv2.bitwise_and(frame, thresh)


def getCentroid(contour):
    moment = cv2.moments(contour)
    if moment["m00"] != 0:
        cx = int(moment["m10"] / moment["m00"])
        cy = int(moment["m01"] / moment["m00"])
        return cx, cy
    else:
        return None


def getMaxContours(contours):
    if len(contours) > 0:
        maxIndex = 0
        maxArea = 0

        for i in range(len(contours)):
            cnt = contours[i]
            area = cv2.contourArea(cnt)

            if area > maxArea:
                maxArea = area
                maxIndex = i
        return contours[maxIndex]


def getContours(histMask):
    grayHistMask = cv2.cvtColor(histMask, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(grayHistMask, 0, L - 1, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def initCapture():
    global mouseMode, histExists
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        k = cv2.waitKey(1) & 0xFF

        if k == ord("z"):
            histExists = True
            hist = createHistogram(frame)
        elif k == ord("m"):
            mouseMode = True
        elif k == ord("n"):
            mouseMode = False
        elif k == ord("q"):
            break
        if histExists:
            detect(frame, hist)
        else:
            drawLocker(frame)

        cv2.imshow("Output", frame)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    initCapture()
