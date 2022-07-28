from __future__ import print_function
from imageai.Detection import ObjectDetection
from PIL import Image
from PIL import ImageTk
import tkinter as tki
import threading
import datetime
import imutils
import cv2
import os
import numpy
import RPi.GPIO as GPIO
GPIO.setmode(GPIO.BOARD)
GPIO.setup(7,GPIO.OUT)
GPIO.setup(11,GPIO.OUT)
GPIO.setup(13,GPIO.OUT)
GPIO.setup(15,GPIO.OUT)
GPIO.setwarnings(False)
class PhotoBoothApp:
        def __init__(self, vs, outputPath):
                # store the video stream object and output path, then initialize
                # the most recently read frame, thread for reading frames, and
                # the thread stop event
                self.vs = vs
                self.outputPath = outputPath
                self.frame = None
                self.thread = None
                self.stopEvent = None
                # initialize the root window and image panel
                self.root = tki.Tk()
                self.panel = None
                                # create a button, that when pressed, will take the current
                # frame and save it to file
                self.execution_path = os.getcwd()
                self.detector = ObjectDetection()
                self.detector.setModelTypeAsRetinaNet()
                self.detector.setModelPath( os.path.join(self.execution_path , "resnet50_coco_best_v2.0.1.h5"))
                self.detector.loadModel()
                btn = tki.Button(self.root, text="Snapshot!", command=self.takeSnapshot)
                btn.pack(side="top", fill="x", expand="yes", padx=10,pady=10)
                b2 = tki.Button(self.root, text="backward", command=self.backward)
                b2.pack(side="bottom", fill="x", expand="no", padx=10,pady=10)
                b1 = tki.Button(self.root, text="forward", command=self.forward)
                b1.pack(side="bottom", fill="x", expand="no", padx=10,pady=10)
                b3 = tki.Button(self.root, text="right", command=self.right)
                b3.pack(side="right", fill="y", expand="no", padx=5,pady=5)
                b4 = tki.Button(self.root, text="left", command=self.left)
                b4.pack(side="left", fill="y", expand="no", padx=5,pady=5)
                b5 = tki.Button(self.root, text="stop", command=self.stop)
                b5.pack(side="bottom", fill="y", expand="no", padx=10,pady=10)
                # start a thread that constantly pools the video sensor for
                # the most recently read frame
                self.stopEvent = threading.Event()
                self.thread = threading.Thread(target=self.videoLoop, args=())
                self.thread.start()
                # set a callback to handle when the window is closed
                self.root.wm_title("PyImageSearch PhotoBooth")
                self.root.wm_protocol("WM_DELETE_WINDOW", self.onClose)
                
        def forward(self):
                print("Forward")
                GPIO.output(7,False)
                GPIO.output(11,True)
                GPIO.output(13,False)
                GPIO.output(15,True)
        def backward(self):
                GPIO.output(7,True)
                GPIO.output(11,False)
                GPIO.output(13,True)
                GPIO.output(15,False)
                print("Backward")
        def right(self):
                GPIO.output(7,True)
                GPIO.output(11,False)
                GPIO.output(13,False)
                GPIO.output(15,True)
                print("Right")
        def left(self):
                GPIO.output(7,False)
                GPIO.output(11,True)
                GPIO.output(13,True)
                GPIO.output(15,False)
                print("Left")
        def stop(self):
                GPIO.output(7,False)
                GPIO.output(11,False)
                GPIO.output(13,False)
                GPIO.output(15,False)
                print("stop")
        def videoLoop(self):
                # DISCLAIMER:
                # I'm not a GUI developer, nor do I even pretend to be. This
                # try/except statement is a pretty ugly hack to get around
                # a RunTime error that Tkinter throws due to threading
                try:
                        # keep looping over frames until we are instructed to stop
                        while not self.stopEvent.is_set():
                                # grab the frame from the video stream and resize it to
                                # have a maximum width of 300 pixels
                                self.frame = self.vs.read()
                                self.frame = imutils.resize(self.frame, width=300)
                
                                # OpenCV represents images in BGR order; however PIL
                                # represents images in RGB order, so we need to swap
                                # the channels, then convert to PIL and ImageTk format
                                image = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
                                image = Image.fromarray(image)
                                image = ImageTk.PhotoImage(image)
                
                                # if the panel is not None, we need to initialize it
                                if self.panel is None:
                                        self.panel = tki.Label(image=image)
                                        self.panel.image = image
                                        self.panel.pack(side="left", padx=10, pady=10)
                
                                # otherwise, simply update the panel
                                else:
                                        self.panel.configure(image=image)
                                        self.panel.image = image
                except RuntimeError:
                        print("[INFO] caught a RuntimeError")
        def takeSnapshot(self):
                # grab the current timestamp and use it to construct the
                # output path
                ts = datetime.datetime.now()
                filename = "{}.jpg".format(ts.strftime("%Y-%m-%d_%H-%M-%S"))
                p = os.path.sep.join((self.outputPath, filename))
                # save the file
                cv2.imwrite(p, self.frame.copy())
                print("[INFO] saved {}".format(filename))
                filename1 = "{}.jpg".format(ts.strftime("%Y-%m-%d_%H-%M-%S")+"detected")

                detections = self.detector.detectObjectsFromImage(input_image=os.path.join(self.execution_path ,filename), output_image_path=os.path.join(self.execution_path ,filename1))
                for eachObject in detections:
                    print(eachObject["name"] , " : " , eachObject["percentage_probability"] )
        def onClose(self):
                # set the stop event, cleanup the camera, and allow the rest of
                # the quit process to continue
                print("[INFO] closing...")
                self.stopEvent.set()
                self.vs.stop()
                self.root.quit()
