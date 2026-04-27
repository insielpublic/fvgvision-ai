import traceback
from abc import ABC
from typing import List, Optional

import cv2
import numpy as np
import torch
from numpy import ndarray
from ultralytics import YOLO

from fvgvisionai.benchmark.benchmark_monitor import BenchmarkMonitor
from fvgvisionai.common.app_timer import AppTimer
from fvgvisionai.common.triple_buffer import TripleBuffer
from fvgvisionai.config.app_settings import AppSettings
from fvgvisionai.notify.notification_client import NotificationClient
from fvgvisionai.processor.abstract_frame_processor import AbstractFrameProcessor
from fvgvisionai.processor.categories import categories_dict
from fvgvisionai.processor.data_aggregator import DataAggregator
from fvgvisionai.processor.decorators.alert_decorator import AlertFrameDecorator
from fvgvisionai.processor.decorators.fps_frame_decorator import FpsFrameDecorator
from fvgvisionai.processor.decorators.object_counter_decorator import ObjectCounterDecorator
from fvgvisionai.processor.decorators.parking_decorator import ParkingDecorator
from fvgvisionai.processor.decorators.video_info_decorator import VideoInfoDecorator
from fvgvisionai.processor.detected_object import DetectedObject
from fvgvisionai.processor.subprocessors.detected_object_sub_processor import DetectedObjectSubProcessor
from fvgvisionai.processor.subprocessors.door_sub_processor import DoorSubProcessor
from fvgvisionai.processor.subprocessors.parking_sub_processor import ParkingSubProcessor
from fvgvisionai.processor.subprocessors.raise_your_hand_sub_processor import RaiseYourHandSubProcessor
from fvgvisionai.processor.subprocessors.in_zone_sub_processor import InZoneSubProcessor, AlarmStatus

MODELS_TRACKER_YAML = "assets/models/tracker.yaml"
MIN_HEIGHT_FOR_MODEL = 448


class UltralyticsFrameProcessor(AbstractFrameProcessor, ABC):
    def __init__(self, buffer: TripleBuffer,
                 notification_client: Optional[NotificationClient],
                 benchmark_monitor: Optional[BenchmarkMonitor],
                 app_settings: AppSettings):
        super().__init__(buffer=buffer,
                         notification_client=notification_client,
                         benchmark_monitor=benchmark_monitor,
                         app_settings=app_settings)

        if self._app_settings.model_filename is not None:
            self._model_file_name = self._app_settings.model_filename.split("/")[-1]
            self._model = YOLO(f"./assets/models/{self._model_file_name}")
        else:
            if self._app_settings.scenario_pose_enabled:
                if self._app_settings.model_use_tensort:
                    self._model_file_name = f"{app_settings.model_id.value['prefix']}{app_settings.model_size.value['suffix']}-pose-{app_settings.model_resolution.value['suffix']}-{app_settings.model_precision.value['suffix']}.engine"
                else:
                    self._model_file_name = f"{app_settings.model_id.value['prefix']}{app_settings.model_size.value['suffix']}-pose.pt"
            else:
                if self._app_settings.model_use_tensort:
                    self._model_file_name = f"{app_settings.model_id.value['prefix']}{app_settings.model_size.value['suffix']}-{app_settings.model_resolution.value['suffix']}-{app_settings.model_precision.value['suffix']}.engine"
                else:
                    self._model_file_name = f"{app_settings.model_id.value['prefix']}{app_settings.model_size.value['suffix']}.pt"
            self._model = YOLO(f"./assets/models/{self._model_file_name}")
        self._logger.info(f"Used model is {self._model_file_name}")

        self._image_source_width = 0
        self._image_source_height = 0

        self._model_width = app_settings.model_width
        self._model_height = app_settings.model_height
        self._model_skip_frames_enabled = app_settings.model_skip_frames_enabled
        self._model_skip_frames_mask = app_settings.model_skip_frames_mask
        self._model_skip_frames_mask_size = len(app_settings.model_skip_frames_mask)

        # if door is enabled, tracking have to be True
        self._tracking_enabled = (app_settings.model_tracking_enabled or
                                  app_settings.scenario_door_enabled or
                                  (app_settings.show_time_people_in_zone and app_settings.scenario_zone_enabled))

        if self._tracking_enabled:
            self._logger.info(f"Used model tracker is {MODELS_TRACKER_YAML}")

        self._scenario_pose_enabled = app_settings.scenario_pose_enabled
        self._scenario_zone_enabled = app_settings.scenario_zone_enabled
        self._scenario_door_enabled = app_settings.scenario_door_enabled
        self._scenario_parking_enabled = app_settings.scenario_parking_enabled

        self._categories = categories_dict

        self._categories_list: List[int] = []
        for item in app_settings.model_categories:
            self._categories_list.append(item.model_class)

        self._detected_objects_processor = DetectedObjectSubProcessor(tracking_enabled=self._tracking_enabled,
                                                                      show_time_people_in_zone=app_settings.show_time_people_in_zone,
                                                                      scenario_in_zone=self._scenario_zone_enabled,
                                                                      scenario_door=self._scenario_door_enabled,
                                                                      show_category=self._app_settings.show_categories,
                                                                      draw_circle_enabled=self._scenario_zone_enabled or self._scenario_door_enabled)
        self._zone_processor = InZoneSubProcessor(app_settings)
        self._door_processor = DoorSubProcessor(app_settings)
        self._raise_your_hand_processor = RaiseYourHandSubProcessor(app_settings)
        self._parking_processor = ParkingSubProcessor(app_settings)

        self._alert_decorator = AlertFrameDecorator(app_settings)
        self._fps_decorator = FpsFrameDecorator()

        if self._scenario_parking_enabled:
            self._obj_count_decorator = ParkingDecorator(
                parking_count=len(self._app_settings.scenario_parking_coords),
                show_category=self._app_settings.show_categories)
        else:
            self._obj_count_decorator = ObjectCounterDecorator(
                show_category=self._app_settings.show_categories,
                scenario_zone_enabled=app_settings.scenario_zone_enabled,
                scenario_time_in_zone_enabled=app_settings.show_time_people_in_zone,
                scenario_door_enabled=app_settings.scenario_door_enabled,
                scenario_door_entering_enabled=app_settings.scenario_door_entering_enabled,
                scenario_door_entering_label=app_settings.scenario_door_entering_label,
                scenario_door_leaving_enabled=app_settings.scenario_door_leaving_enabled,
                scenario_door_leaving_label=app_settings.scenario_door_leaving_label,
                scenario_raise_hands_enabled=app_settings.scenario_pose_enabled)

        self._video_info_decorator = VideoInfoDecorator(
            model_file_name=self._model_file_name,
            model_library=app_settings.model_library,
            model_precision=app_settings.model_precision,
            cuda_enabled=torch.cuda.is_available(),
            tensor_enabled=app_settings.model_use_tensort,
            model_size=app_settings.model_size,
            tracking_enabled=self._tracking_enabled,
            zone_enabled=self._scenario_zone_enabled,
            door_enabled=self._scenario_door_enabled,
            pose_enabled=self._scenario_pose_enabled,
            model_skip_frames=boolean_array_to_binary_sequences(app_settings.model_skip_frames_enabled,
                                                                app_settings.model_skip_frames_mask)
        )
        self._zone_poly = np.array([])
        self._ratio_width = 1
        self._ratio_height = 1

        self.performance_timer = AppTimer()

        self.frame_counter = 0
        self.list_objects: List[DetectedObject] = []

    def setup_video_parameters(self, video_width: int, video_height: int, source_frame_rate_declared: int):
        super().setup_video_parameters(video_width, video_height, source_frame_rate_declared)
        self._image_source_width = video_width
        self._image_source_height = video_height

        # https://github.com/ultralytics/ultralytics/issues/3955
        # modello deve avere dimensioni multiple di 32
        self._logger.warning(f"model image size is set to {self._model_width}x{self._model_height}")

        self._ratio_width = video_width / self._model_width
        self._ratio_height = video_height / self._model_height

        self._zone_poly = self._app_settings.scenario_zone_coords

        self._alert_decorator.init_image_size(video_width, video_height)
        self._fps_decorator.init_image_size(video_width, video_height)
        self._obj_count_decorator.init_image_size(video_width, video_height)
        self._video_info_decorator.init_image_size(video_width, video_height)

    def _execute_frame_analysis(self, source_frame: ndarray,
                                data_aggregator: DataAggregator) -> (Optional[ndarray], AlarmStatus, AlarmStatus):
        try:
            self.performance_timer.start()

            if self.is_mask_opened():
                # model is reduced respect source video_source
                model_frame: ndarray = cv2.resize(source_frame, (self._model_width, self._model_height))

                if self._tracking_enabled:
                    results = self._model.track(model_frame, verbose=False,
                                                tracker=MODELS_TRACKER_YAML,
                                                imgsz=self._app_settings.model_resolution.value["resolution"],
                                                conf=self._app_settings.model_confidence,
                                                iou=self._app_settings.model_iou,
                                                classes=self._categories_list,
                                                persist=True)
                else:
                    results = self._model(model_frame, verbose=False,
                                          imgsz=self._app_settings.model_resolution.value["resolution"],
                                          conf=self._app_settings.model_confidence,
                                          iou=self._app_settings.model_iou,
                                          classes=self._categories_list)

                list_res_detect = []
                list_time_detect = []
                for r in results:
                    list_res_detect.append(r.cpu())
                    list_time_detect.append(r.speed)

                self._logger.debug(f"\t\tperformance model {self.performance_timer.elapsed_time:.2f} ms")
                self.performance_timer.start()

                #
                # Process operations
                #
                self.list_objects = self.extract_detected_objects(list_res_detect,
                                                                  self._raise_your_hand_processor.is_enabled)

            self.frame_counter = (self.frame_counter + 1) % self._model_skip_frames_mask_size

            # count objects detected in this frame or in an old one
            data_aggregator.measure_objects_counter(self.list_objects)

            zone_alarm_status = AlarmStatus.NORMAL

            # Process scenario zone
            if self._zone_processor.is_enabled:
                (avg_time_in_zone, max_time_in_zone,
                 min_time_in_zone, objects_in_zone, zone_alarm_status) = self.process_scenario_in_zone()
                data_aggregator.measure_items_in_zone(objects_in_zone,
                                                      min_time_in_zone, max_time_in_zone,
                                                      avg_time_in_zone)

            # Process scenario parking
            if self._parking_processor.is_enabled:
                (avg_time_in_zone, max_time_in_zone,
                 min_time_in_zone, objects_in_zone, zone_alarm_status) = self.process_scenario_parking()
                data_aggregator.measure_items_in_zone(objects_in_zone,
                                                      min_time_in_zone, max_time_in_zone,
                                                      avg_time_in_zone)
            # Process scenario door
            total_people_entering = 0
            total_people_leaving = 0
            if self._door_processor.is_enabled:
                (entering_by_category, leaving_by_category,
                 total_people_entering, total_people_leaving) = self.process_scenario_door()
                data_aggregator.measure_people_near_door(entering_by_category, leaving_by_category)

            # Process scenario raise hands
            if self._raise_your_hand_processor.is_enabled:
                people_raised_hand, raised_hand_alarm_status = self.process_scenario_raised_hand()
                data_aggregator.measure_people_with_raised_hands(people_raised_hand)
            else:
                raised_hand_alarm_status = AlarmStatus.NORMAL

            self._logger.debug(f"\t\tperformance process {self.performance_timer.elapsed_time / 1000:.2f} ms")
            self.performance_timer.start()

            #
            # Drawing operations
            #

            # draw scenari
            self._draw_scenario_parking(source_frame)
            self._draw_scenario_in_zone(source_frame)
            self._draw_scenario_door(source_frame)

            # draw on input frame, starting from model frame dimensions
            self._detected_objects_processor.draw(source_frame, self.list_objects)

            # draw alarm icon
            self._draw_alert_icons(raised_hand_alarm_status, source_frame, zone_alarm_status)

            # draw boxes
            self._show_count_box(data_aggregator, raised_hand_alarm_status, source_frame, total_people_entering,
                                 total_people_leaving, zone_alarm_status)
            self._show_video_info_box(source_frame)
            self._show_fps_box(data_aggregator, source_frame)
            self._show_time_box(source_frame)

            self._logger.debug(f"\t\tperformance drawing {self.performance_timer.elapsed_time / 1000:.2f} ms")
            self.performance_timer.start()

            return source_frame, zone_alarm_status, raised_hand_alarm_status

        except Exception as e:
            self._logger.error("Unexpected error %s " % e)
            traceback.print_exc()
            return None, None

    def is_mask_opened(self) -> bool:
        return self.frame_counter < self._model_skip_frames_mask_size and self._model_skip_frames_mask[
            self.frame_counter]

    def _show_fps_box(self, data_aggregator: DataAggregator, source_frame: ndarray):
        if self._app_settings.show_fps_enabled:
            self._fps_decorator.draw(source_frame,
                                     data_aggregator.time_frame_source_declared,
                                     data_aggregator.time_frame_acquisition_average,
                                     data_aggregator.time_frame_processing_average,
                                     self._app_settings.video_output_fps)

    def _show_video_info_box(self, source_frame: ndarray):
        if self._app_settings.show_video_info_enabled:
            self._video_info_decorator.draw(source_frame,
                                            self._image_source_width, self._image_source_height,
                                            self._model_width, self._model_height)

    def _show_count_box(self, data_aggregator, raised_hand_alarm_status, source_frame: ndarray, total_people_entering,
                        total_people_leaving, zone_alarm_status):
        if self._app_settings.show_count_enabled:
            self._obj_count_decorator.draw(frame=source_frame, detected_objects=self.list_objects,
                                           in_zone_people=data_aggregator.people_in_zone_counter.last_value,
                                           in_zone_avg_time=data_aggregator.avg_time_in_zone,
                                           people_entering=total_people_entering,
                                           people_leaving=total_people_leaving,
                                           people_raised_hand=data_aggregator.people_with_raised_hands.last_value,
                                           zone_status=zone_alarm_status,
                                           raised_hand_alarm_status=raised_hand_alarm_status)

    def _draw_alert_icons(self, raised_hand_alarm_status, source_frame, zone_alarm_status):
        if self._app_settings.show_alert_icon:
            self._alert_decorator.draw(source_frame, zone_alarm_status, raised_hand_alarm_status)

    def _draw_scenario_door(self, source_frame: ndarray):
        if self._door_processor.is_enabled:
            self._door_processor.draw(source_frame)

    def _draw_scenario_in_zone(self, source_frame: ndarray):
        if self._zone_processor.is_enabled:
            self._zone_processor.draw(source_frame)

    def _draw_scenario_parking(self, source_frame: ndarray):
        if self._parking_processor.is_enabled:
            self._parking_processor.draw(source_frame)

    def process_scenario_raised_hand(self):
        people_raised_hand = 0
        raised_hand_alarm_status = AlarmStatus.NORMAL
        if self._raise_your_hand_processor.is_enabled:
            raised_hand_alarm_status, people_raised_hand = self._raise_your_hand_processor.detected_raised_hands(
                self.list_objects)
        return people_raised_hand, raised_hand_alarm_status

    def process_scenario_in_zone(self):
        objects_in_zone, min_time_in_zone, max_time_in_zone, avg_time_in_zone = 0, 0, 0, 0
        zone_alarm_status = AlarmStatus.NORMAL
        if self._zone_processor.is_enabled:
            (zone_alarm_status, objects_in_zone,
             min_time_in_zone, max_time_in_zone,
             avg_time_in_zone) = self._zone_processor.detected_objects_in_zone(self.list_objects)
        return avg_time_in_zone, max_time_in_zone, min_time_in_zone, objects_in_zone, zone_alarm_status

    def process_scenario_parking(self):
        objects_in_zone, min_time_in_zone, max_time_in_zone, avg_time_in_zone = 0, 0, 0, 0
        zone_alarm_status = AlarmStatus.NORMAL
        if self._parking_processor.is_enabled:
            (zone_alarm_status, objects_in_zone,
             min_time_in_zone, max_time_in_zone,
             avg_time_in_zone) = self._parking_processor.detected_objects_parked(self.list_objects)
        return avg_time_in_zone, max_time_in_zone, min_time_in_zone, objects_in_zone, zone_alarm_status

    def process_scenario_door(self):
        entering_by_category = {}
        leaving_by_category = {}
        total_entering = 0
        total_leaving = 0
        if self._door_processor.is_enabled:
            (entering_by_category, 
             leaving_by_category,
             total_entering,
             total_leaving) = (self._door_processor
                              .detected_objects_near_door_zone(self.list_objects))
        return entering_by_category, leaving_by_category, total_entering, total_leaving

    def extract_detected_objects(self, list_res_detect, pose_enabled: bool) -> List[DetectedObject]:
        list_objects: List[DetectedObject] = []
        for element in list_res_detect:
            boxes = element.boxes.numpy()
            index = 0
            for box in boxes:
                if box.id is None:
                    object_id = int(box.cls)
                else:
                    object_id = int(box.id)
                cls = int(box.cls)

                bbox_xy = box.xyxy[0].astype(np.int32).tolist()
                bbox_wh = box.xywh[0].astype(np.int32).tolist()

                # Set image_source dimension from model_size to source frame size
                bbox_xy[0] *= self._ratio_width
                bbox_xy[2] *= self._ratio_width
                bbox_wh[0] *= self._ratio_width
                bbox_wh[2] *= self._ratio_width

                bbox_xy[1] *= self._ratio_height
                bbox_xy[3] *= self._ratio_height
                bbox_wh[1] *= self._ratio_height
                bbox_wh[3] *= self._ratio_height

                if pose_enabled:
                    current_keypoints = element.keypoints[index]
                else:
                    current_keypoints = None
                index += 1
                detected_object = DetectedObject(object_id=object_id,
                                                 label_num=cls,
                                                 label_str=self._categories[cls].label,
                                                 color=list(self._categories[cls].color),
                                                 conf=round(float(box.conf), 2),
                                                 bbox_xy=bbox_xy,
                                                 bbox_wh=bbox_wh,
                                                 keypoints=current_keypoints)

                list_objects.append(detected_object)
        return list_objects


def boolean_array_to_binary_sequences(mask_enabled: bool, boolean_array: List[bool]) -> str:
    if not mask_enabled:
        return "False"
    # Converti la lista di booleani in una stringa di '0' e '1'
    binary_string = ''.join('1' if bit else '0' for bit in boolean_array)

    # Converti la stringa binaria in un numero intero
    decimal_number = int(binary_string, 2)

    # Converte il numero intero in esadecimale e rimuove il prefisso '0x'
    hex_representation = hex(decimal_number)[2:]

    return hex_representation
