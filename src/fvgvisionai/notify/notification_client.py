import json
import logging
import threading
import traceback
from azure.iot.device import IoTHubModuleClient, Message
from azure.iot.device.exceptions import NoConnectionError
from azure.iot.device.iothub.abstract_clients import AbstractIoTHubClient
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Optional

from fvgvisionai.common.app_alarm import AlarmStatus
from fvgvisionai.common.app_timer import AppTimer
from fvgvisionai.common.atomic_boolean import AtomicBoolean
from fvgvisionai.processor.data_aggregator import DataAggregator
from fvgvisionai.processor.categories import categories_dict

DATE_FORMAT = "%Y-%m-%d %H:%M:%S.%f"

DEFAULT_AGGREGATION_TIME_S = 10.0


def _get_now_as_string():
    return datetime.today().strftime(DATE_FORMAT)[:-3]


def _get_datetime_as_string(date: datetime):
    return date.strftime(DATE_FORMAT)[:-3]


class NotificationClient:
    def __init__(self, enabled: bool, azure_connection_string: str, device_id: str,
                 camera_id: str, model_id: str, measures_aggregation_time_ms: int):
        self._enabled = enabled
        self._lock = threading.Lock()
        self._processor_timer = AppTimer()
        self._logger = logging.getLogger(__name__)
        self._azure_connection_string = azure_connection_string
        self._client: Optional[AbstractIoTHubClient] = None
        self._device_id = device_id
        self._camera_id = camera_id
        self._model_id = model_id
        self._ready = AtomicBoolean()
        self._message_counter = 0
        self._ready.set(True)
        self._measures_aggregation_time_ms = measures_aggregation_time_ms
        self._client = IoTHubModuleClient.create_from_connection_string(self._azure_connection_string)
        self._notification_executor = ThreadPoolExecutor(thread_name_prefix="notification")

        self._global_process_timer = AppTimer()

    def start(self):
        self._global_process_timer.start()

    @property
    def is_enabled(self):
        return self._enabled

    @property
    def is_ready(self):
        return self._ready.get()

    def handle_alarms(self, raised_hand_alarm_status: AlarmStatus, zone_alarm_status: AlarmStatus):
        if zone_alarm_status == AlarmStatus.ALARM_WITH_NOTIFICATION:
            self._notification_executor.submit(self.execute_alert, "people in zone", 1)
        if raised_hand_alarm_status == AlarmStatus.ALARM_WITH_NOTIFICATION:
            self._notification_executor.submit(self.execute_alert, "raised hand", 2)

    def handle_notification(self, data_aggregation_datetime_start: datetime, data_aggregator: DataAggregator) -> bool:
        # simple notification
        global_elapsed_time_ms = self._global_process_timer.elapsed_time
        if global_elapsed_time_ms >= self._measures_aggregation_time_ms:
            self._logger.debug(f"\t\tglobal elapsed time: {global_elapsed_time_ms:.2f} ms")
            data = data_aggregator.copy()
            data_start = data_aggregation_datetime_start
            data_end = datetime.now()
            self._notification_executor.submit(self._execute_notification,
                                               data_start,
                                               data_end,
                                               data)
            self._global_process_timer.start()
            return True
        return False

    def execute_alert(self, alert_type: str, alert_id: int):
        if self._enabled:
            timer = AppTimer()
            timer.start()
            with self._lock:
                try:
                    self._ready.set(False)

                    message = self._build_alert_payload(alert_id, alert_type)
                    self._send_message(message)

                    elapsed_time = timer.stop()
                    self._logger.warning(
                        f"Send alert #{self._message_counter} of type '{alert_type}' "
                        f"in {elapsed_time:.0f} ms")

                except GeneratorExit:
                    # This is raised when the client disconnects
                    logging.error('Client disconnected')
                except Exception as e:
                    # Gestisci l'eccezione qui
                    logging.error(f"Si è verificata un'eccezione: {e}")
                    traceback.print_exc()
                finally:
                    self._ready.set(True)

    def _execute_notification(self, time_interval_start: datetime, time_interval_end: datetime, data: DataAggregator):
        if self._enabled:
            timer = AppTimer()
            timer.start()
            with self._lock:
                try:
                    self._ready.set(False)
                    self._message_counter = (self._message_counter + 1) % 1_000_000

                    message = self._build_message_payload(self._message_counter,
                                                          time_interval_start, time_interval_end,
                                                          data)

                    self._send_message(message)

                    elapsed_time = timer.stop()
                    self._logger.info(
                        f"Send #{self._message_counter} data for {data.aggregation_size} "
                        f"frames in {elapsed_time:.0f} ms from: {_get_datetime_as_string(time_interval_start)} "
                        f"to {_get_datetime_as_string(time_interval_end)} ")
                    self._logger.debug(f"message payload: {message}")

                except GeneratorExit:
                    # This is raised when the client disconnects
                    logging.error('Client disconnected')
                except Exception as e:
                    # Gestisci l'eccezione qui
                    logging.error(f"Si è verificata un'eccezione: {e}")
                    traceback.print_exc()
                finally:
                    self._ready.set(True)

    def _connect(self):
        self._client.connect()

    def _send_message(self, frame_info: dict):
        frame_info_str = json.dumps(obj=frame_info, indent=4)
        msg = Message(frame_info_str)
        msg.content_encoding = "utf-8"
        msg.content_type = "application/json"
        if self._client is not None:
            if not self._client.connected:
                self._connect()
            try:
                self._client.send_message(msg)
            except NoConnectionError as e:
                logging.warning(f"Connessione persa: {e}")
                traceback.print_exc()
            except Exception as e:
                # Gestisci l'eccezione qui
                logging.error(f"Si è verificata un'eccezione: {e}")
                traceback.print_exc()

    def _build_message_payload(self, message_counter: int,
                               time_interval_start: datetime,
                               time_interval_end: datetime,
                               data: DataAggregator) -> dict:
        agg_frame_info_dict = {
            "message_type": "MESSAGE",
            "interval_id": message_counter,
            "interval_start": _get_datetime_as_string(time_interval_start),
            "interval_end": _get_datetime_as_string(time_interval_end),

            "avg_people_down": 0,

            "max_people": data.people_counter.max,
            "min_people": data.people_counter.min,
            "avg_people": data.people_counter.average,

            "avg_bikes": data.bikes_counter.average,

            "max_cars": data.cars_counter.max,
            "min_cars": data.cars_counter.min,
            "avg_cars": data.cars_counter.average,

            "max_people_in_zone": data.people_in_zone_counter.max,
            "min_people_in_zone": data.people_in_zone_counter.min,
            "avg_people_in_zone": data.people_in_zone_counter.average,

            "max_time_in_zone": data.max_time_in_zone,
            "min_time_in_zone": data.max_time_in_zone,
            "avg_time_in_zone": data.avg_time_in_zone,

            "device_id": self._device_id,
            "camera_id": self._camera_id,
            "model_id": self._model_id
        }
        
        # Add per-category door counters
        total_entrances = 0
        total_exits = 0
        
        for label_num, count in data.door_entered_by_category.items():
            # Get category label string, fallback to class number if not found
            if label_num in categories_dict:
                label_str = categories_dict[label_num].label
            else:
                label_str = f'class_{label_num}'
            agg_frame_info_dict[f"sum_entrances_{label_str}"] = count
            total_entrances += count
        
        for label_num, count in data.door_exited_by_category.items():
            # Get category label string, fallback to class number if not found
            if label_num in categories_dict:
                label_str = categories_dict[label_num].label
            else:
                label_str = f'class_{label_num}'
            agg_frame_info_dict[f"sum_exits_{label_str}"] = count
            total_exits += count
        
        # Add aggregate totals for backward compatibility
        agg_frame_info_dict["sum_entrances"] = total_entrances
        agg_frame_info_dict["sum_exits"] = total_exits
        
        return agg_frame_info_dict

    # people in zone,man down
    def _build_alert_payload(self, alert_id: int, alert_type: str) -> dict:
        alert_info_dict = {"message_type": "ALERT",
                           "alert_id": alert_id,
                           "device_id": self._device_id,
                           "time": datetime.today().strftime(DATE_FORMAT)[:-3],
                           "alert_type": alert_type}
        return alert_info_dict
