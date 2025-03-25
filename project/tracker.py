from yolox.tracker.byte_tracker import BYTETracker
from player import Player
from locker import lock

class Tracker:
    def __init__(self,predictor,args,test_size):
        self.predictor = predictor
        self.args = args
        self.tracker = BYTETracker(self.args, frame_rate=30)
        self.test_size = test_size
        self.frame= None
        self.last_frame_id=-1
        self.current_frame_id=-1
        self.gap=0

    def _update(self,online_tlwhs, online_ids, online_scores):
        with lock:
            Player.class_ids = online_ids
            Player.confidences = online_scores
            Player.boxes =  online_tlwhs
            Player.gap = self.gap

    def pre(self,timer):
        while True:
            self.frame, self.current_frame_id = Player.get_frame()

            if self.frame is None:
                continue

            if self.last_frame_id != -1:
                self.gap = self.current_frame_id - self.last_frame_id
            self.last_frame_id = self.current_frame_id

            outputs, img_info = self.predictor.inference(self.frame)
            if outputs[0] is not None:
                online_targets = self.tracker.update(outputs[0], [img_info['height'], img_info['width']], self.test_size)
                online_tlwhs = []
                online_ids = []
                online_scores = []
                for t in online_targets:
                    tlwh = t.tlwh
                    tid = t.track_id
                    vertical = tlwh[2] / tlwh[3] > self.args.aspect_ratio_thresh
                    if tlwh[2] * tlwh[3] > self.args.min_box_area and not vertical:
                        online_tlwhs.append(tlwh)
                        online_ids.append(tid)
                        online_scores.append(t.score)
                self._update(online_tlwhs, online_ids, online_scores)
            else:
                continue
