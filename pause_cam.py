import numpy as np
import imutils
import cv2
from helper import FPS2, WebcamVideoStream
from skimage import measure
from random import randint
import easygui
import pyfakewebcam

def resize(img):
    global target_size
    return cv2.resize(img, target_size)

def toggleCam(event, x, y, flags, param):
    global showCam
    if event == cv2.EVENT_RBUTTONUP:
        showCam=not showCam

def segmentation():
    global target_size

    w=640
    h=480
    vs = WebcamVideoStream(0, w, h).start()
    fake2 = pyfakewebcam.FakeWebcam('/dev/video2', w,h)

    resize_ratio = 1.0 * 513 / max(vs.real_width, vs.real_height)
    target_size = (int(resize_ratio * vs.real_width),
                   int(resize_ratio * vs.real_height))
    fps = FPS2(5).start()

    print("Starting...")

    name="Webcam-Replacement"
    cv2.namedWindow(name,16)# 16 = WINDOW_GUI_NORMAL, disable right click

    cv2.setMouseCallback(name, toggleCam)

    global showCam
    showCam=True

    while vs.isActive():
        if showCam:
            image = cv2.resize(vs.read(), target_size)

        ir = cv2.resize(image, (vs.real_width, vs.real_height))

        ir2 = cv2.cvtColor(ir, cv2.COLOR_BGR2RGB)
        fake2.schedule_frame(ir2)

        cv2.imshow(name, ir)


        key=cv2.waitKey(1) 
        if key & 0xFF == ord('q'):
            break
        if key & 0xFF == 27:
            break
        if key & 0xFF == ord('c'):
            showCam=not showCam
        if cv2.getWindowProperty(name, 0) < 0:
            break
        if cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE) < 1:
            break
        fps.update()

    fps.stop()
    vs.stop()

    cv2.destroyAllWindows()


if __name__ == '__main__':
    segmentation()
