import pyautogui

pyautogui.PAUSE = 0  # seconds to pause after function calls
pyautogui.FAILSAFE = False  # allows the mouse to exit the window


def move_mouse(coordinates, shape):
    screenCols, screenRows = pyautogui.size()
    col, row = coordinates[0], coordinates[1]
    pyautogui.moveTo(col * screenCols / shape[1], row * screenRows / shape[0])


def scroll(path):
    if len(path) <= 1:
        return
    distance = path[-1][1] - path[-2][1]
    pyautogui.scroll(-distance / 2)
