import cv2
import numpy as np

from yolox.utils.visualize import get_color

class Plot:
    def __init__(self):
        self.track_history = {}
        self.counted_ids = set()
        self.total_in = 0
        self.total_out = 0
        self.center_line = 0

    def plot_tracking(self,image, tlwhs, obj_ids, scores=None, frame_id=0, fps=0., gap=0, ids2=None):
        im = np.ascontiguousarray(np.copy(image))
        im_h, im_w = im.shape[:2]
        self.center_line = image.shape[0] // 2

        top_view = np.zeros([im_w, im_w, 3], dtype=np.uint8) + 255

        # text_scale = max(1, image.shape[1] / 1600.)
        # text_thickness = 2
        # line_thickness = max(1, int(image.shape[1] / 500.))
        text_scale = 2
        text_thickness = 2
        line_thickness = 3

        radius = max(5, int(im_w / 140.))
        cv2.putText(im, f'frame: {frame_id} fps: {fps:.2f} num: {len(tlwhs)} gap:{gap}',
                    (0, int(15 * text_scale)), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), thickness=2)

        cv2.line(im, (0, self.center_line), (im_w, self.center_line), (0, 255, 255), 2)

        for i, tlwh in enumerate(tlwhs):
            x1, y1, w, h = tlwh
            cx = int(x1 + w / 2)
            cy = int(y1 + h / 2)
            obj_id = int(obj_ids[i])
            id_text = str(obj_id)
            if ids2 is not None:
                id_text += f", {int(ids2[i])}"

            color = get_color(abs(obj_id))
            intbox = tuple(map(int, (x1, y1, x1 + w, y1 + h)))
            cv2.rectangle(im, intbox[0:2], intbox[2:4], color=color, thickness=line_thickness)
            cv2.putText(im, id_text, (intbox[0], intbox[1]), cv2.FONT_HERSHEY_PLAIN,
                        text_scale, (0, 0, 255), thickness=text_thickness)

            if obj_id not in self.track_history:
                self.track_history[obj_id] = []
            self.track_history[obj_id].append((cx, cy))
            if len(self.track_history[obj_id]) > 2:
                self.track_history[obj_id].pop(0)

            if obj_id not in self.counted_ids and len(self.track_history[obj_id]) == 2:
                y1 = self.track_history[obj_id][0][1]
                y2 = self.track_history[obj_id][1][1]
                if (y1 - self.center_line) * (y2 - self.center_line) < 0:  # 穿越了
                    if y2 < y1:
                        self.total_in += 1
                    else:
                        self.total_out += 1
                    self.counted_ids.add(obj_id)

            if len(self.track_history[obj_id]) == 2:
                pt1 = self.track_history[obj_id][0]
                pt2 = self.track_history[obj_id][1]
                cv2.line(im, pt1, pt2, color=color, thickness=4)

        cv2.putText(im, f"In: {self.total_in}  Out: {self.total_out}",
                    (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 2)

        return im
