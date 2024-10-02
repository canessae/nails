import csv
import os
import sys
import threading
import time
import tkinter
from signal import alarm
from tkinter import ttk
import cv2
import PIL
import numpy as np

from tkinter import *

from matplotlib.axis import Axis
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,
                                               NavigationToolbar2Tk)

from alertplot import AlertPlot
from camerasearch import CameraSearch
from myvideocapture import MyVideoCapture
from myprocessing import MyProcessing



class App:

    SIZE_WHITE = 40
    SIZE_BLACK = 20

    START_AUTOSTART = "Start autosnap"
    STOP_AUTOSTART = "Stop autosnap"

    DEFAULT_INTERVAL = 10
    DEFAULT_REPETITIONS = 5
    
    DATAFILE = 'data.csv'

    EMPTYRECT = [0, 0, 0, 0]

    def __init__(self, window, window_title, video_source=0):
        self.flagSnaphot = False
        self.window = window
        self.window.title(window_title)
        self.video_source = video_source
        self.proc = MyProcessing()
        self.nn_threshold = tkinter.IntVar()
        self.timeposition = -15 #this variable to manage the position of plots
        self.updateViaTimer = False

        self.reset_nails_images()

        self.vid = MyVideoCapture()

        window.bind('<space>', self.spazio)
        window.bind('<Return>', self.enter)

        ico = PIL.Image.open('logo.jpg')
        resized_image= ico.resize((256,256), PIL.Image.LANCZOS)
        photo = PIL.ImageTk.PhotoImage(resized_image)
        window.wm_iconphoto(False, photo)

        toolbar = tkinter.Frame(window, bd=1, relief=tkinter.RAISED)
        label = tkinter.Label(toolbar, text='Webcam ')
        label.pack(side=tkinter.LEFT)
        selected_device = tkinter.StringVar()
        self.combobox = ttk.Combobox(toolbar, textvariable=selected_device)
        self.combobox.pack(side=tkinter.LEFT)
        toolbar.pack(side=tkinter.TOP, fill=tkinter.X)
        self.combobox.bind('<<ComboboxSelected>>', self.camera_changed)
        self.combobox['state'] = 'readonly'

        frame = tkinter.Frame(window)
        # Create a canvas that can fit the above video source size
        self.canvas = tkinter.Canvas(frame, width=self.vid.width, height=self.vid.height)
        self.canvas.pack(side=tkinter.LEFT)

        # Canvas for neural network
        self.canvas2 = tkinter.Canvas(frame, width=self.vid.width, height=self.vid.height)
        self.canvas2.pack(side=tkinter.LEFT)

        # slider for threshold
        self.sliderthreshold = tkinter.Scale(frame, orient=tkinter.VERTICAL, variable=self.nn_threshold, command=self.threshold_changed)
        self.sliderthreshold.pack(side=tkinter.LEFT, fill='y')
        self.sliderthreshold.set(70)
        frame.pack()

        # Frame for nails area left
        self.framenail = tkinter.Frame(window)
        #framenail.grid_configure(0, weight=1, uniform="fred")
        w = self.vid.width
        h = self.vid.height
        image = PIL.Image.open('current.jpg') #TODO
        image = image.resize((int(w/5), int(h/5)))
        self.snapshot_size = (int(w/5), int(h/5))
        logo_for_nails = PIL.ImageTk.PhotoImage(image)

        # pictures inside the frame, 7 images
        self.measvar = [tkinter.StringVar() for i in range(0,7)]
        self.meas = []
        self.label = []
        for i in range(0,7):
            dummy = tkinter.Frame(self.framenail)
            self.label.append( tkinter.Label(dummy, image=logo_for_nails, height=100) )
            self.label[i].pack(side=tkinter.TOP)
            self.meas.append(tkinter.Label(dummy, textvariable=self.measvar[i]))
            self.meas[i].pack(side=tkinter.TOP)
            dummy.pack(side=tkinter.LEFT, fill='x', ipadx=20)
        self.framenail.pack(expand=True)
        empty = np.zeros((int(logo_for_nails.height()/2), int(logo_for_nails.width()/2)))
        photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(empty))
        self.label[5].configure(image=photo)
        self.label[5].image = photo
        self.measvar[5].set("Black")
        self.label[6].configure(image=photo)
        self.label[6].image = photo
        self.measvar[6].set("White")

        # Button that lets the user take a snapshot
        framebutton = tkinter.Frame(window)
        self.btn_snapshot = tkinter.Button(framebutton, text="Snapshot", width=50, command=self.snapshot)
        self.btn_snapshot.pack(anchor=tkinter.CENTER, side=tkinter.LEFT, expand=True)
        self.btn_snapshotadd = tkinter.Button(framebutton, text="Add sample", width=50, command=self.addsnap)
        self.btn_snapshotadd.pack(anchor=tkinter.CENTER, side=tkinter.LEFT, expand=True)
        self.btn_deletedata = tkinter.Button(framebutton, text="Delete all", width=50, command=self.delete_samples)
        self.btn_deletedata.pack(anchor=tkinter.CENTER, side=tkinter.LEFT, expand=True)
        framebutton.pack()

        # Auto snapshot
        mysize = "00000"
        frameauto = tkinter.Frame(window)
        label = Label(frameauto, text="Interval [s]")
        label.pack(side=LEFT, padx=5)
        self.interval = IntVar()
        self.interval.set(self.DEFAULT_INTERVAL)
        self.entry_interval = tkinter.Entry(frameauto, bd=2, textvariable=self.interval, width=len(mysize))
        self.entry_interval.pack(side=tkinter.LEFT)
        #repetitions
        label = Label(frameauto, text="Repetitions")
        label.pack(side=LEFT, padx=5)
        self.snapnumber = IntVar()
        self.snapnumber.set(0)
        label = tkinter.Label(frameauto, textvariable=self.snapnumber)
        label.pack(side=LEFT)
        label = tkinter.Label(frameauto, text="/")
        label.pack(side=LEFT)
        self.repetitions = IntVar()
        self.repetitions.set(self.DEFAULT_REPETITIONS)
        mysize = "00"
        self.entry_repetitions = tkinter.Entry(frameauto, bd=2, textvariable=self.repetitions, width=len(mysize))
        self.entry_repetitions.pack(side=tkinter.LEFT)
        self.text_btn_autosnap = StringVar()
        self.text_btn_autosnap.set(self.START_AUTOSTART)
        self.btn_autostart = tkinter.Button(frameauto, textvariable=self.text_btn_autosnap, width=50, command=self.autosnap)
        self.btn_autostart.pack(anchor=tkinter.CENTER, side=tkinter.LEFT, expand=True)
        frameauto.pack()

        # history diagram
        self.history = tkinter.Frame(window)
        self.btn_backward = tkinter.Button(self.history, text="BACK", command=self.backward)
        self.btn_backward.pack(side=tkinter.LEFT, fill=tkinter.Y)

        #-----------plot
        self.fig = Figure(figsize=(3, 3), dpi=100)
        self.histcanvas = FigureCanvasTkAgg(self.fig, master=self.history)
        self.histcanvas.draw()
        self.histcanvas.get_tk_widget().pack(side=tkinter.LEFT, fill=tkinter.X, expand=True)
        # -----------plot

        #alert plot
        self.alerthistory = AlertPlot(self.history)
        self.alerthistory.pack(side=tkinter.LEFT, fill=tkinter.X)

        self.btn_forward = tkinter.Button(self.history, text="FORW", command=self.forward)
        self.btn_forward.pack(side=tkinter.LEFT, fill=tkinter.Y)

        self.history.pack(fill=tkinter.X)


        self.get_available_cameras()
        # open video source (by default this will try to open the computer webcam)
        del self.vid
        self.vid = MyVideoCapture(self.camera_find_index(self.combobox.current()))

        # After it is called once, the update method will be automatically called every delay milliseconds
        self.delay = 15
        self.flag_update = True
        self.onDrawWebcam()
        self.refresh_data()

        window.bind("<<event1>>", self.eventhandler)

        self.window.mainloop()


    def reset_nails_images(self):
        os.system('cp original.jpg current.jpg')
        if hasattr(self, 'label'):
            w = self.vid.width
            h = self.vid.height
            image = PIL.Image.open('current.jpg')
            image = image.resize((int(w / 5), int(h / 5)))
            photo = PIL.ImageTk.PhotoImage(image)
            for idx in range(0,5):
                self.label[idx].configure(image=photo)
                self.label[idx].image = photo
                self.measvar[idx].set("")

    def eventhandler(self, evt):
        print('Event Thread', threading.get_ident())  # event thread id (same as main)
        print(evt.state)  # 123, data from event

    def spazio(self, event):
        print("spazio")
        self.addsnap()

    def enter(self, event):
        print('enter')
        self.snapshot()


    def camera_restart(self, index):
        self.flag_update = False

        print(index)
        self.video_source = index
        if hasattr(self, "vid"):
            pass
            #self.vid.releaseCamera()
            #del self.vid

        self.vid = MyVideoCapture(self.video_source)

        if self.vid is not None:
            self.flag_update = True
            #self.update()


    def camera_changed(self, event):
        print("called")
        idx = self.camera_find_index(self.combobox.current())
        print(f'indice {idx}')
        self.camera_restart(idx)


    def camera_find_index(self, combovalue):
        return self.available[combovalue]['devid']


    def get_available_cameras(self):
        cameras = CameraSearch()
        cameras.detect()
        self.available = cameras.getAllCameras()

        list = []
        for camera in self.available:
            list.append(camera['name'])

        if len(list) > 0:
            self.combobox['values'] = list
            self.combobox.current(0)


    def threshold_changed(self, val):
        if not hasattr(self.proc, "unghie"):
            return
        drawing, centroids, labels, boundinrrect, base = self.proc.apply(nn_threshold=float(val)/100)

        frame = base.copy()
        self.order = self.add_labels_to_drawing(frame, labels, drawing, centroids)
        self.update_zoomed_version(frame, drawing, boundinrrect, labels)
        dsize = (int(self.vid.width), int(self.vid.height))
        toshow = cv2.resize(drawing, dsize)
        self.update_right_image(toshow)


    def append_data(self):
        timestamp = int(time.time())

        if not hasattr(self, 'black') or not hasattr(self, 'white'):
            print("skipped image due to white or black")
            return

        tosave = {'timestamp': timestamp,
                  'finger1': self.measvar[0].get(),
                  'finger2': self.measvar[1].get(),
                  'finger3': self.measvar[2].get(),
                  'finger4': self.measvar[3].get(),
                  'finger5': self.measvar[4].get(),
                  'black' : self.black,
                  'white' : self.white,
                  }

        exists = os.path.exists(self.DATAFILE)

        with open(self.DATAFILE, 'a', newline='') as csvfile:
            fieldnames = ['timestamp', 'finger1', 'finger2', 'finger3', 'finger4', 'finger5', 'black', 'white']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if not exists:
                writer.writeheader()
            writer.writerow(tosave)
        self.refresh_data()


    def load_data(self) -> bool:
        self.current_data = []
        if not os.path.exists(self.DATAFILE):
            return False

        with open(self.DATAFILE, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                self.current_data.append(row)
                #print(row)
        return True


    def divide_data(self, keys = ['finger1', 'finger2','finger3','finger4','finger5']):
        fingers = []
        for key in keys:#repeat for every finger in keys
            first = True
            finger = np.array([])
            for idx, item in enumerate(self.current_data):
                if item[key] != '':
                    #print(f' item = {item[key]}')
                    values = list(item[key][1:-1].split()) #remove [ and ]
                    tmp = np.array(float(item['timestamp'])) #extract timestamp from csv
                    #get black and white reference
                    bvalues = list(item['black'][1:-1].split())  # remove [ and ]
                    wvalues = list(item['white'][1:-1].split())  # remove [ and ]

                    sample = np.hstack((idx, tmp, np.array([float(values[0]), float(values[1]), float(values[2])]),
                                                np.array([float(bvalues[0]), float(bvalues[1]), float(bvalues[2])]),
                                                np.array([float(wvalues[0]), float(wvalues[1]), float(wvalues[2])])))

                    #print(f'sample = {sample}')
                    if first:
                        first = False
                        finger = sample
                    else:
                        finger = np.vstack((finger, sample))

            # found or not found finger
            if finger.size > 0 and len(finger.shape) == 1:
                finger = np.array([finger])

            fingers.append(finger)

        return fingers


    def refresh_data(self):

        self.fig.clear()

        self.plotR = self.fig.add_subplot(131)
        self.plotR.set_title("Red")
        self.plotG = self.fig.add_subplot(132)
        self.plotG.set_title("Green")
        self.plotB = self.fig.add_subplot(133)
        self.plotB.set_title("Blue")

        #if data.csv is not present
        if not self.load_data():
            self.histcanvas.draw_idle() #refresh if called after delete
            return

        fingers = self.divide_data()

        self.maxtimeposition = -100
        for i in range(0, 5):
            print(f'calcolo maxtimeposition {fingers[i].shape[0]}')
            if fingers[i].shape[0] > self.maxtimeposition:
                self.maxtimeposition = fingers[i].shape[0]

        #normalize finger values with respect to black and white
        self.livius_normalize(fingers)

        pos = self.timeposition
        symbols = ['^g-', 'xr-', 'ok-', '*r-', 'sg-']
        for i in range(0, 5):
            if fingers[i].size == 0:
                continue
            self.plotR.plot(fingers[i][pos:, 0][:15], fingers[i][pos:, 2][:15], symbols[i])
        self.plotR.grid(True)
        self.plotR.legend(labels=['1','2','3','4','5'])
        self.plotR.set_ylim([0, 1])
        self.plotR.set_ylabel('dark          clear')
        l, r = self.plotR.get_xlim()
        self.plotR.set_xlim(r-15, r)

        for i in range(0, 5):
            if fingers[i].size == 0:
                continue
            self.plotG.plot(fingers[i][pos:, 0][:15], fingers[i][pos:, 3][:15], symbols[i])
        self.plotG.grid(True)
        self.plotG.legend(labels=['1', '2', '3', '4', '5'])
        self.plotG.set_ylim([0, 1])
        self.plotG.set_ylabel('dark          clear')
        l, r = self.plotG.get_xlim()
        self.plotG.set_xlim(r-15, r)

        for i in range(0, 5):
            if fingers[i].size == 0:
                continue
            self.plotB.plot(fingers[i][pos:, 0][:15], fingers[i][pos:, 4][:15], symbols[i])
        self.plotB.grid(True)
        self.plotB.legend(labels=['1', '2', '3', '4', '5'])
        self.plotB.set_ylim([0, 1])
        self.plotB.set_ylabel('dark          clear')
        l, r = self.plotB.get_xlim()
        self.plotB.set_xlim(r-15, r)

        self.histcanvas.draw_idle()

        #update alert box
        self.alerthistory.clearGridData()
        keys = ['finger1', 'finger2', 'finger3', 'finger4', 'finger5']
        print(f'pos = {pos} {len(self.current_data[pos:])}')
        for item in self.current_data[pos:][0:15]:
            print(item)
            elem = []
            for key in keys:
                if item[key] == '':
                    elem.append(self.EMPTYRECT)
                else:
                    tmp = item['black'][1:-1].split()
                    black = np.array([float(tmp[0]), float(tmp[1]), float(tmp[2])])
                    tmp = item['white'][1:-1].split()
                    white = np.array([float(tmp[0]), float(tmp[1]), float(tmp[2])])
                    tmp = item[key][1:-1].split()
                    item_n = np.array([float(tmp[0]), float(tmp[1]), float(tmp[2])])
                    item_n = ((item_n - black) / (white - black)).clip(0,1)
                    elem.append(item_n)
            self.alerthistory.appendColumn(elem)

        # for k in range(1, 16):
        #     elem = []
        #     for i in range(0, 5):
        #         if fingers[i].size == 0:
        #             elem.append(self.EMPTYRECT)
        #         else:
        #             m = fingers[i][pos:, 2:5].shape[0]
        #             print(f'k = {k}')
        #             if k <= m:
        #                 elem.append(fingers[i][pos:, 2:5][-k])
        #                 print(f'{k} - {i}')
        #                 print(fingers[i][pos:, 2:5][-k])
        #             else:
        #                 elem.append(self.EMPTYRECT)
        #     print(f'elem = {elem}')
        #     self.alarm.appendColumn(elem)

    def livius_normalize(self, fingers):
        for i in range(0, 5):
            if fingers[i].size == 0:
                continue
            #R
            black = fingers[i][:, -6]
            white = fingers[i][:, -3]
            fingers[i][:, 2] = ((fingers[i][:,2] - black) / (white - black)).clip(0,1)
            #G
            black = fingers[i][:, -5]
            white = fingers[i][:, -2]
            fingers[i][:, 3] = ((fingers[i][:, 3] - black) / (white - black)).clip(0, 1)
            #B
            black = fingers[i][:, -4]
            white = fingers[i][:, -1]
            fingers[i][:, 4] = ((fingers[i][:, 4] - black) / (white - black)).clip(0, 1)

        print('livius_normalize')


    def addsnap(self):
        #img = cv2.imread("current.jpg")
        # for i in [0,1,2]:
        #     img[:,:,i] = ((img[:,:,i] - self.black[i])/(self.white[i] - self.black[i])).clip(0,1)*255
        #     img[:,:,i] = img[:,:,i].astype('uint8')
        #     #img[:,:,i] = (img[:,:,i]/self.white[i]).clip(0,1)*255
        # #img = cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        # #cv2.imwrite("nuovo.jpg", img)
        #
        # dsize = (int(self.vid.width), int(self.vid.height))
        # toshow = cv2.resize(img, dsize)
        # self.update_right_image(toshow)

        # frame2 = np.zeros(frame.shape, frame.dtype)
        self.append_data()


    def delete_samples(self):
        if os.path.exists(self.DATAFILE):
            os.remove(self.DATAFILE)
        self.reset_nails_images()
        self.alerthistory.clearGridData()
        self.alerthistory.display([])
        self.refresh_data()


    def enableButtons(self, en = False):
        if not en:
            self.btn_snapshotadd['state'] = tkinter.DISABLED
            self.btn_snapshot['state'] = tkinter.DISABLED
            self.btn_deletedata['state'] = tkinter.DISABLED
            #self.btn_autostart['state'] = tkinter.DISABLED
        else:
            self.btn_snapshotadd['state'] = tkinter.NORMAL
            self.btn_snapshot['state'] = tkinter.NORMAL
            self.btn_deletedata['state'] = tkinter.NORMAL
            #self.btn_autostart['state'] = tkinter.NORMAL

    def autosnapTimer(self):
        self.counter += 1
        self.window.event_generate("<<event1>>", when="tail", state=123)
        self.snapnumber.set(self.counter)

        # call to acquire image and add sample
        self.snapshot()
        self.updateViaTimer = True
        print("def autosnapTimer(self):")

        if self.counter == self.repetitions.get():
            self.enableButtons(True)
            self.snapnumber.set(0)
            self.text_btn_autosnap.set(self.START_AUTOSTART)
            return

        # restart timer
        self.timer1 = threading.Timer(interval=self.interval.get(), function=self.autosnapTimer)
        self.timer1.start()

    def autosnap(self):
        self.counter = 0
        if self.text_btn_autosnap.get() == self.START_AUTOSTART:
            self.text_btn_autosnap.set(self.STOP_AUTOSTART)
            self.enableButtons(False)
            # call the timer callback
            self.autosnapTimer()
        else:
            if hasattr(self, 'timer1') and self.timer1 is not None:
                self.timer1.cancel()
            self.text_btn_autosnap.set(self.START_AUTOSTART)
            self.enableButtons(True)

    def forward(self):
        self.timeposition += 5
        print(f'timeposition {self.timeposition}')
        if self.timeposition >= 0:
            self.timeposition -= 5
        self.refresh_data()

    def backward(self):
        self.timeposition -= 5
        print(f'timeposition {self.timeposition}/{self.maxtimeposition}')
        if self.timeposition < -self.maxtimeposition:
            self.timeposition = -self.maxtimeposition
        self.refresh_data()

    #this function saves the image in current.jpg
    def snapshot(self):
        # Get a frame from the video source
        ret, frame = self.vid.get_frame()

        if ret:
            self.flagSnaphot = True
            #cv2.imwrite("frame-" + time.strftime("%d-%m-%Y-%H-%M-%S") + ".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            cv2.imwrite("current.jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            print(f'immagine salvata {frame.shape}')


    def drawText(self, image, text, position):
        position = [int(item) for item in position]
        # font
        font = cv2.FONT_HERSHEY_SIMPLEX
        # fontScale
        fontScale = 1.6
        # Blue color in BGR
        color = (255, 0, 0)
        # Line thickness of 2 px
        thickness = 2
        # Using cv2.putText() method
        image = cv2.putText(image, text, position, font,
                            fontScale, color, thickness, cv2.LINE_AA)


    def add_labels_to_drawing(self, frame, labels, drawing, centroids):
        fact = frame.shape[1]/drawing.shape[1]
        order = []
        for i in range(0, len(labels)):
            id = self.whereIsPoint(centroids[i], self.regions, fact)
            order.append(id)
            self.drawText(drawing, str(id), centroids[i])
        return order

    def __mymax(self, img):
        ret = []
        ret.append(np.max(img[:, :, 0]))
        ret.append(np.max(img[:, :, 1]))
        ret.append(np.max(img[:, :, 2]))
        return ret

    def __mymin(self, img):
        ret = []
        ret.append(np.min(img[:, :, 0]))
        ret.append(np.min(img[:, :, 1]))
        ret.append(np.min(img[:, :, 2]))
        return ret

    def add_boxes(self, img : np.ndarray):
        h, w, _ = img.shape
        m_h = int(h / 2)
        m_w = int(w / 2)
        #print(f'{m_h} {m_w}')

        # white region
        size = self.SIZE_BLACK
        # top
        ret = self.__mymax(img[0:int(size), m_w-int(size/2):m_w+int(size/2), :])
        white = np.array([ret])
        #plot black rectangle for white region
        cv2.rectangle(img, (m_w-int(size/2), 0), (m_w+int(size/2), int(size)), (0,0,0), 2)

        # left
        ret = self.__mymax(img[m_h-int(size/2):m_h+int(size/2), 0:size, :])
        white = np.concatenate((white, np.array([ret])))
        # plot black rectangle for white region
        cv2.rectangle(img, (0, m_h-int(size/2)), (size, m_h+int(size/2)), (0, 0, 0), 2)

        # right
        ret = self.__mymax(img[m_h-int(size/2):m_h+int(size/2), w - size:w, :])
        white = np.concatenate((white, np.array([ret])))
        # plot black rectangle for white region
        cv2.rectangle(img, (w - size, m_h-int(size/2)), (w, m_h+int(size/2)), (0, 0, 0), 2)
        #print(f'white {white}')
        white = np.max(white, axis=0)
        #print(f'white {white}')

        # black region
        size = self.SIZE_BLACK
        ret = self.__mymin(img[0:size, 0:size, :])
        black = np.array([ret])
        # plot white rectangle for black region
        cv2.rectangle(img, (0, 0), (size, size), (255, 255, 255), 2) #tl

        ret = self.__mymin(img[h-size:h, w-size:w, :])
        black = np.concatenate((black, np.array([ret])))
        # plot white rectangle for black region
        cv2.rectangle(img, (w-size, h-size), (w, h), (255, 255, 255), 2) #br

        ret = self.__mymin(img[0:size, w - size:w, :])
        black = np.concatenate((black, np.array([ret])))
        # plot white rectangle for black region
        cv2.rectangle(img, (w - size, 0), (w, size), (255, 255, 255), 2) #tr

        ret = self.__mymin(img[h-size:h, 0:size, :])
        black = np.concatenate((black, np.array([ret])))
        cv2.rectangle(img, (0, h - size), (size, h), (255, 255, 255), 2) #bl

        black = np.min(black, axis=0)

        return white, black


    def addVerticalGrid(self, frame : np.ndarray):
        h, w, _ = frame.shape
        step = int(w/5)
        regions = []

        for i in range(0,5):
            cv2.rectangle(frame, (i*step, 0), (i*step+step, h), (0,0,255), 3)
            regions.append(((i*step, 0), (i*step+step, h)))

        return regions

    def onDrawWebcam(self):
        # Get a frame from the video source
        if self.vid is not None:
            ret, frame = self.vid.get_frame()
        else:
            ret = False

        if ret:
            self.add_boxes(frame)
            # print(f'white {self.white} and black {self.black}')
            self.regions = self.addVerticalGrid(frame)
            self.photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(frame))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tkinter.NW)

        if self.flag_update:
            self.window.after(self.delay, self.onDrawWebcam)

        if self.flagSnaphot is True:
            self.flagSnaphot = False
            self.white, self.black = self.add_boxes(frame)
            drawing, centroids, labels, boundingrect, base = self.proc.apply(nn_threshold=float(self.nn_threshold.get())/100, force=True)

            #show labels
            self.order = self.add_labels_to_drawing(frame, labels, drawing, centroids)

            #show rect in snapshot bar
            self.update_zoomed_version(base, drawing, boundingrect, labels)

            #show white and black
            self.update_white_black(self.white, self.black)

            #update right image
            dsize = (int(self.vid.width), int(self.vid.height))
            toshow = cv2.resize(drawing, dsize)
            self.update_right_image(toshow)

        if self.updateViaTimer:
            self.updateViaTimer = False
            self.addsnap()


    def clear_zoomed_version(self):
        for idx in range(0,5):
            #if idx in [5,6]:
            #    empty = np.zeros((int(self.snapshot_size[0]/2), int(self.snapshot_size[1]/2)))
            #else:
            w, h = self.snapshot_size
            empty = np.ones((h, w, 3))*220 #220 is interface gray!

            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 3
            color = (255, 0, 0)
            thickness = 5
            empty = empty.astype('uint8')
            print(empty.shape)
            h, w, _ = empty.shape
            # Using cv2.putText() method
            empty = cv2.putText(empty, str(idx+1), (0, h-10), font,
                                fontScale, color, thickness, cv2.LINE_AA)

            photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(empty))
            self.label[idx].configure(image=photo)
            self.label[idx].image = photo
            self.measvar[idx].set("")


    def update_white_black(self, white, black):
        empty = np.ones((int(self.snapshot_size[0]/2), int(self.snapshot_size[1]/2), 3))
        empty[:, :, 0] = np.ones((int(self.snapshot_size[0]/2), int(self.snapshot_size[1]/2))) * black[0]
        empty[:, :, 1] = np.ones((int(self.snapshot_size[0]/2), int(self.snapshot_size[1]/2))) * black[1]
        empty[:, :, 2] = np.ones((int(self.snapshot_size[0]/2), int(self.snapshot_size[1]/2))) * black[2]
        empty = empty.astype('uint8')
        photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(empty))
        self.label[5].configure(image=photo)
        self.label[5].image = photo
        self.measvar[5].set(black)

        empty = np.ones((int(self.snapshot_size[0]/2), int(self.snapshot_size[1]/2), 3))
        empty[:, :, 0] = np.ones((int(self.snapshot_size[0]/2), int(self.snapshot_size[1]/2))) * white[0]
        empty[:, :, 1] = np.ones((int(self.snapshot_size[0]/2), int(self.snapshot_size[1]/2))) * white[1]
        empty[:, :, 2] = np.ones((int(self.snapshot_size[0]/2), int(self.snapshot_size[1]/2))) * white[2]
        empty = empty.astype('uint8')
        photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(empty))
        self.label[6].configure(image=photo)
        self.label[6].image = photo
        self.measvar[6].set(white)


    def whereIsPoint(self, point, regions, fact) -> int:
        x = point[0]*fact
        for idx, item in enumerate(regions):
            minx = item[0][0]
            maxx = item[1][0]
            if x<maxx and x>minx:
                #print(f'{x} -> {idx}')
                return idx+1
        return -1

    def update_zoomed_version(self, frame, drawing, boundinrrect, labels):
        tosave = frame.copy()
        factx = tosave.shape[1] / drawing.shape[1]
        facty = tosave.shape[0] / drawing.shape[0]

        self.clear_zoomed_version()

        for idx, (x, y, w, h) in enumerate(boundinrrect):
            if idx == 5:
                break
            # print(f'{idx} {x} {y} {w} {h}')
            x *= factx
            w *= factx
            y *= facty
            h *= facty
            x = int(x)
            y = int(y)
            w = int(w)
            h = int(h)
            selection = frame[y:y + h, x:x + w, :]
            # tosave = cv2.rectangle(tosave, (x,y), (x+w,y+h), (0,255,0), 3);
            selection = cv2.resize(selection, self.snapshot_size)
            photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(selection))
            #reorder
            id = self.order[idx]-1
            self.label[id].configure(image=photo)
            self.label[id].image = photo
            self.measvar[id].set(labels[idx])


    def update_right_image(self, toshow):
        self.toshow = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(toshow))
        self.canvas2.create_image(0, 0, image=self.toshow, anchor=tkinter.NW)

    # Create a window and pass it to the Application object

if __name__ == "__main__":
    App(tkinter.Tk(), "NAILS")
