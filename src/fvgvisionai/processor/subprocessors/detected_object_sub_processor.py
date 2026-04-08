import logging
from typing import List, Dict

import cv2
from numpy import ndarray

from fvgvisionai.common.video_utils import draw_icon
from fvgvisionai.processor.categories import categories_dict, ModelCategory
from fvgvisionai.processor.detected_object import DetectedObject

RAISED_HAND_COLOR = (0, 0, 255)
# Definire il colore del rettangolo (in formato BGR)
yellow_color = (0, 255, 255)


def format_duration(seconds):
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    # Formattare il risultato come stringa
    formatted_duration = "{:02}:{:02}:{:02}".format(int(hours), int(minutes), int(seconds))

    return formatted_duration


class DetectedObjectSubProcessor:
    def __init__(self, tracking_enabled: bool,
                 show_time_people_in_zone: bool,
                 scenario_in_zone: bool,
                 scenario_door: bool,
                 show_category: List[ModelCategory],
                 draw_circle_enabled: bool):
        self._tracking_enabled = tracking_enabled
        self._draw_circle_enabled = draw_circle_enabled

        self._scenario_in_zone = scenario_in_zone
        self._scenario_door = scenario_door
        self._show_category_set = set(obj.model_class for obj in show_category)
        self._show_time_object_in_zone = show_time_people_in_zone

        self.categories = categories_dict

        self.clock_image = cv2.imread('assets/images/icon_timer16.png', -1)
        self.icons: Dict[str, ndarray] = {"person": cv2.imread('assets/images/icon_people16.png', -1),
                                          "bicycle": cv2.imread('assets/images/icon_bicycle16.png', -1),
                                          "car": cv2.imread('assets/images/icon_car16.png', -1),
                                          "motorcycle": cv2.imread('assets/images/icon_motorbike16.png', -1),
                                          "bus": cv2.imread('assets/images/icon_bus16.png', -1),
                                          "truck": cv2.imread('assets/images/icon_truck16.png', -1),
                                          "electric_scooter": cv2.imread('assets/images/icon_scooter16.png', -1)}

        self._logger = logging.getLogger(__name__)

    def draw(self, frame: ndarray, detected_objects: List[DetectedObject]):
        for item in detected_objects:
            if item.label_num in self._show_category_set:
                self._draw_detected_object_box(item, frame)

    def _draw_detected_object_box(self, detected_object: DetectedObject, frame: ndarray):
        """
        draw each bounding box with the color provided in "labels_dict", uses self.bbox_xy and self.bbox_wh

        params:
            frame: current frame given by opencv
            labels_dict: dictionary with corresponding labels for categories of objects
        """

        if self._tracking_enabled:
            text = f"  {detected_object.label_str} id:{hex(detected_object.id % 256)[2:].upper().zfill(2)} ({detected_object.conf})"
        else:
            text = f"  {detected_object.label_str} ({detected_object.conf})"

        box_color = detected_object.color
        if detected_object.raised_hands:
            box_color = RAISED_HAND_COLOR

        txt_size = cv2.getTextSize(text=text, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=.5, thickness=1)[0]

        # bbox rectangle
        cv2.rectangle(img=frame,
                      pt1=(int(detected_object.bbox_x1), int(detected_object.bbox_y1)),
                      pt2=(int(detected_object.bbox_x2), int(detected_object.bbox_y2)),
                      color=box_color,
                      thickness=2)

        # header of bbox to put info into
        cv2.rectangle(img=frame,
                      pt1=(int((detected_object.bbox_x1 - 1)),
                           int(detected_object.bbox_y1 - int(2 * txt_size[1]))),
                      pt2=(int((detected_object.bbox_x1 + txt_size[0] + 2)),
                           int(detected_object.bbox_y1)),
                      color=detected_object.color,
                      thickness=-1)

        if detected_object.is_in_zone and self._show_time_object_in_zone is True:
            txt_time_size = cv2.getTextSize(text=f"   {format_duration(detected_object.time_in_zone)}",
                                            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=.5, thickness=1)[0]

            # Disegnare il rettangolo sull'immagine
            cv2.rectangle(img=frame,
                          pt1=(int((detected_object.bbox_x1 - 1)),
                               int(detected_object.bbox_y2 + int(2 * txt_time_size[1]) + 2)),
                          pt2=(int((detected_object.bbox_x1 + txt_time_size[0] + 2)),
                               int(detected_object.bbox_y2)),
                          color=yellow_color,
                          thickness=-1)
            cv2.putText(img=frame,
                        text=f"   {format_duration(detected_object.time_in_zone)}",
                        org=(int(detected_object.bbox_x1),
                             int((detected_object.bbox_y2 + int(2 * txt_time_size[1])))),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=.5,
                        color=(0, 0, 0),
                        thickness=1)

            draw_icon(frame, int(detected_object.bbox_x1),
                      int(detected_object.bbox_y2 + txt_time_size[1]) - 2, self.clock_image)

        if self._draw_circle_enabled:
            point_color = detected_object.color

            if self._scenario_in_zone:
                if detected_object.is_in_zone:
                    point_color = (0, 0, 255)
            elif self._scenario_door:
                if detected_object.is_door_entering_zone:
                    point_color = (0, 255, 0)
                elif detected_object.is_door_leaving_zone:
                    point_color = (0, 0, 255)

            cv2.circle(img=frame,
                       center=(int(detected_object.bbox_x), int(detected_object.bbox_y + (detected_object.bbox_h / 2))),
                       radius=8, color=point_color, thickness=-1)

        try:
            if int(detected_object.bbox_y1) - int(1 * txt_size[1]) - 6 >= 0:
                draw_icon(frame, int(detected_object.bbox_x1),
                          int(detected_object.bbox_y1) - int(1 * txt_size[1]) - 6,
                          self.icons[detected_object.label_str])
        except Exception as e:
            self._logger.error(f"{detected_object.label_str} with exception: {e}")

        cv2.putText(img=frame,
                    text=text,
                    org=(int(detected_object.bbox_x1),
                         int((detected_object.bbox_y1 - int(.5 * txt_size[1])))),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=.5,
                    color=(0, 0, 0),
                    thickness=1)
