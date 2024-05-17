import os
import time
from ultralytics import YOLO
import cv2
import numpy as np
import win32gui, win32ui, win32con, win32api
import pydirectinput
import torch



class WindowCapture:

    # properties
    w = 0
    h = 0
    hwnd = None
    cropped_x = 0
    cropped_y = 0
    offset_x = 0
    offset_y = 0

    # constructor
    def __init__(self, window_name=None):
        # find the handle for the window we want to capture.
        # if no window name is given, capture the entire screen
        if window_name is None:
            self.hwnd = win32gui.GetDesktopWindow()
        else:
            self.hwnd = win32gui.FindWindow(None, window_name)
            if not self.hwnd:
                raise Exception('Window not found: {}'.format(window_name))

        # get the window size
        window_rect = win32gui.GetWindowRect(self.hwnd)
        self.w = window_rect[2] - window_rect[0]
        self.h = window_rect[3] - window_rect[1]

        # window border and titlebar and cut them off
        border_pixels = 8
        titlebar_pixels = 30
        self.w = self.w - (border_pixels * 2)
        self.h = self.h - titlebar_pixels - border_pixels
        self.cropped_x = border_pixels
        self.cropped_y = titlebar_pixels

        # set the cropped coordinates offset so we can translate screenshot
        # images into actual screen positions
        self.offset_x = window_rect[0] + self.cropped_x
        self.offset_y = window_rect[1] + self.cropped_y

    def get_screenshot(self):

        # get the window image data
        wDC = win32gui.GetWindowDC(self.hwnd)
        dcObj = win32ui.CreateDCFromHandle(wDC)
        cDC = dcObj.CreateCompatibleDC()
        dataBitMap = win32ui.CreateBitmap()
        dataBitMap.CreateCompatibleBitmap(dcObj, self.w, self.h)
        cDC.SelectObject(dataBitMap)
        cDC.BitBlt((0, 0), (self.w, self.h), dcObj, (self.cropped_x, self.cropped_y), win32con.SRCCOPY)

        # convert the raw data into a format opencv can read
        signedIntsArray = dataBitMap.GetBitmapBits(True)
        img = np.fromstring(signedIntsArray, dtype='uint8')
        img.shape = (self.h, self.w, 4)

        # free resources 
        dcObj.DeleteDC()
        cDC.DeleteDC()
        win32gui.ReleaseDC(self.hwnd, wDC)
        win32gui.DeleteObject(dataBitMap.GetHandle())

        img = img[...,:3]


       # https://github.com/opencv/opencv/issues/14866#issuecomment-580207109 
        img = np.ascontiguousarray(img)

        return img

    @staticmethod
    def list_window_names():
        def winEnumHandler(hwnd, ctx):
            if win32gui.IsWindowVisible(hwnd):
                print(hex(hwnd), win32gui.GetWindowText(hwnd))
        win32gui.EnumWindows(winEnumHandler, None)

    def get_screen_position(self, pos):
        return (pos[0] + self.offset_x, pos[1] + self.offset_y)
    
# code to train the model using yolov8 but I moved the training to google colab for faster results
def trainModel():
    model = YOLO("yolov8s.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    results = model.train(data='data.yaml', epochs=100)

def aimBot():
    wincap = WindowCapture() # <------------------------ window name needs to go here -------------------------------->
    model_path = os.path.join('.','runs','detect','train','weights','last.pt')
    model = YOLO(model_path)

    while True:

        frame = wincap.get_screenshot()

        H, W, _ = frame.shape 
        threshold = 0.5


        results = model(frame)[0]

        # drawing boxes around detected objects and printing the class name around the box
        for result in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = result
            print(f'You have spotted a {class_id}')
            if score > threshold : # change this around if the model is not detecting the objects or the correct objects
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
                cv2.putText(frame, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

                box_center_x = x1 + (x2 - x1) / 2
                box_center_y = (y1 + (y2 - y1) / 2) - y2 * .1
                screen_width = win32api.GetSystemMetrics(0)
                screen_height = win32api.GetSystemMetrics(1)

                # moving crosshair to model box
                target_x = int(screen_width * box_center_x / W)
                target_y = int(screen_height * box_center_y / H)
                pydirectinput.moveTo(target_x, target_y)
        
        cv2.imshow("COMPUTER VISION",frame)

        if cv2.waitKey(1) == ord('q'):
            cv2.destroyAllWindows()
            print("FILE DONE")
         
if __name__ == '__main__':
    aimBot()
    

    