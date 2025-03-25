import time
import threading
import cv2
from locker import condition ,lock
from plot import Plot
from yolox.tracking_utils.timer import Timer

class Player:
    frame = None
    frame_count = 0
    class_ids = []
    confidences = []
    boxes = []
    gap = -1

    def __init__(self,args):
        #self.cap =cv2.VideoCapture(args.path)
        self.cap = cv2.VideoCapture(args.path if args.demo == "video" else args.camid)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        self.width =self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height =self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.pre_fps =self.cap.get(cv2.CAP_PROP_FPS)
        self.boxes = None
        self.class_ids = None
        self.confidences = None
        self.frame_id=0
        self.start_time =None
        self.plot=Plot()

    def showFrame(self,timer):
        cv2.namedWindow('frame')
        cv2.resizeWindow('frame',1920,1080)
        self.start_time = time.time()
        timer.tic()
        pre_frame_time = 1.0 / self.pre_fps
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            Player.update_frame(frame)
            self.frame_id+=1

            with lock:
                self.boxes = Player.boxes
                self.class_ids = Player.class_ids
                self.confidences = Player.confidences

            elapsed = timer.toc()
            timer.tic()

            expected_time = self.start_time + self.frame_id * pre_frame_time
            now = time.time()
            delay = expected_time - now
            if delay > 0:
                time.sleep(delay)

            fps = 1. / max(1e-5, timer.average_time)
            fra = self.plot.plot_tracking(
                frame, self.boxes, self.class_ids, frame_id=self.frame_id, fps=fps, gap=self.gap
            )

            cv2.imshow("Video", fra)
            cv2.waitKey(1)
            if cv2.waitKey(1) in [27, ord("q"), ord("Q")]:
                break


        self.cap.release()
        cv2.destroyAllWindows()

    @staticmethod
    def update_frame(fra):
        global condition
        with condition:
            Player.frame = fra
            Player.frame_count += 1
            condition.notify()  # 通知等待线程

    @staticmethod
    def get_frame():
        global condition
        with condition:
            while Player.frame is None:
                condition.wait()  # 等待新的帧数据
            frame = Player.frame.copy()
            frame_id = Player.frame_count
            Player.frame = None  # 清空帧
            return frame,frame_id