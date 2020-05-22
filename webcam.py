import numpy as np
# import tensorflow as tf # for tf1
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import imutils
import os
import cv2
from helper import FPS2, WebcamVideoStream
from skimage import measure
from random import randint
import easygui
import pyfakewebcam

img_num=0
bg_img=None

def load_model():
    print('Loading model...')
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        seg_graph_def = tf.GraphDef()
        with tf.gfile.GFile('models/deeplabv3_mnv2_pascal_train_aug/frozen_inference_graph.pb', 'rb') as fid:
            serialized_graph = fid.read()
            seg_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(seg_graph_def, name='')
    return detection_graph


def nextBg(event, x, y, flags, param):
    global img_num
    global bg_img
    global bg_imgs
    if event == cv2.EVENT_RBUTTONUP:
        img_num+=1
        bg_img=bg_imgs[img_num%len(bg_imgs)]

def resize(img):
    global target_size
    return cv2.resize(img, target_size)

def loadBg(directory):
    filelist = [file for file in os.listdir(directory) if file.endswith('.jpg')]

    bg_imgs= []
    for x in filelist:
        print("Load BG",x)
        bg_imgs.append(resize(cv2.imread(x)))
    return bg_imgs

def openImg():
    global bg_img
    filename=easygui.fileopenbox("Bitte Bild auswählen", "Bildauswahl", "./*.jpg",["*.jpg"])
    if filename != None:
        print("Load ",filename)
        bg_img=resize(cv2.imread(filename))


def segmentation(detection_graph):
    global bg_img
    global bg_imgs
    global target_size

    w=640
    h=480
    vs = WebcamVideoStream(0, w, h).start()
    fake2 = pyfakewebcam.FakeWebcam('/dev/video2', w,h)

    resize_ratio = 1.0 * 513 / max(vs.real_width, vs.real_height)
    target_size = (int(resize_ratio * vs.real_width),
                   int(resize_ratio * vs.real_height))
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    fps = FPS2(5).start()

    bg_imgs=loadBg(".")
    bg_img=bg_imgs[0]

    print("Starting...")

    name="Webcam-Replacement"
    cv2.namedWindow(name,16)# 16 = WINDOW_GUI_NORMAL, disable right click

    cv2.setMouseCallback(name, nextBg)

    global img_num
    img_num=0

    showCam=False

    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            while vs.isActive():
                image = cv2.resize(vs.read(), target_size)
                batch_seg_map = sess.run('SemanticPredictions:0',
                                         feed_dict={'ImageTensor:0': [cv2.cvtColor(image, cv2.COLOR_BGR2RGB)]})

                seg_map = batch_seg_map[0]
                seg_map[seg_map != 15] = 0

                bg_copy=bg_img.copy()

                mask = (seg_map == 15)
                if showCam:
                    bg_copy=image
                else:
                    bg_copy[mask] = image[mask]


                seg_image = np.stack(
                    (seg_map, seg_map, seg_map), axis=-1).astype(np.uint8)
                gray = cv2.cvtColor(seg_image, cv2.COLOR_BGR2GRAY)

                ir = cv2.resize(bg_copy, (vs.real_width, vs.real_height))

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
                if key & 0xFF == ord('o'):
                    openImg()
                if cv2.getWindowProperty(name, 0) < 0:
                    break
                if cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE) < 1:
                    break
                fps.update()

    fps.stop()
    vs.stop()

    cv2.destroyAllWindows()


if __name__ == '__main__':

    graph = load_model()
    segmentation(graph)
