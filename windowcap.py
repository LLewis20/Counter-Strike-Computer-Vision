import numpy as np
import win32gui, win32ui, win32con
import mss
import numpy as np


class WindowCapture:

    # properties
    w = 0
    h = 0
    monitor = None

    # constructor
    def __init__(self, window_name=None):
        with mss.mss() as sct:
            if window_name is None:
                self.monitor = sct.monitors[0]
            else:
                window = {'top': 0, 'left': 0, 'width': 1920, 'height': 1080}
                self.monitor = window
                
            self.w = self.monitor['width']
            self.h = self.monitor['height']
            
    def get_screenshot(self):
        with mss.mss() as sct:
            screenshot = sct.grab(self.monitor)
        return np.array(screenshot)

    @staticmethod
    def list_window_names():
        def winEnumHandler(hwnd, ctx):
            if win32gui.IsWindowVisible(hwnd):
                print(hex(hwnd), win32gui.GetWindowText(hwnd))
        win32gui.EnumWindows(winEnumHandler, None)

    def get_screen_position(self, pos):
        return (pos[0] + self.offset_x, pos[1] + self.offset_y)