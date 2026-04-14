import logging
import numpy as np
from configparser import ConfigParser
from typing import List, Dict, Optional

from fvgvisionai.common.utils import is_tensorrt_installed
from fvgvisionai.config.app_settings_utils import from_props, \
    to_bool, from_env, extract_dimensions, get_array_as_string, to_image_type, \
    to_door_poly, to_model_size, to_zone_poly, ModelSize, ImageType, to_model_library, \
    ModelLibrary, to_model_precision, \
    ModelPrecision, to_model_resolution, ModelResolution, to_binary_array, to_model_categories, to_video_source_mode, \
    VideoSourceMode, to_parking_list, to_str, to_model_filename, ModelId, to_model_id
from fvgvisionai.config.colored_formatter import ColoredFormatter
from fvgvisionai.config.constants import NOTIFICATION_AZURE_CONNECTION_STRING, \
    LOGGING_LEVEL, SECTION_SETTINGS, VIDEO_OUTPUT_RTSP, VIDEO_OUTPUT_RTSP_GPU_NVIDIA, VIDEO_OUTPUT_RTSP_RISOLUZIONE, VIDEO_OUTPUT_RTSP_URL, VIDEO_SOURCE, \
    SECTION_VIDEO_SOURCE, VIDEO_SOURCE_FORCED_FPS_ENABLED, VIDEO_SOURCE_FORCED_FPS_VALUE, \
    VIDEO_SOURCE_FORCED_RESOLUTION_ENABLED, VIDEO_SOURCE_FORCED_RESOLUTION_VALUE, MODEL_LIBRARY, SECTION_MODEL, \
    MODEL_ID, MODEL_CONFIDENCE, MODEL_USE_TENSORT, MODEL_IOU, MODEL_IMAGE_SIZE, MODEL_SIZE, MODEL_DO_TRACKING, \
    SCENARIO_DOOR, SECTION_SPACE_ANALYSIS, SCENARIO_IN_ZONE, SCENARIO_RAISED_HAND, SCENARIO_IN_ZONE_COORDS, \
    SCENARIO_DOOR_COORDS, VIDEO_OUTPUT_FPS, SECTION_VIDEO_OUTPUT, VIDEO_OUTPUT_STREAM, VIDEO_OUTPUT_IMAGE, \
    VIDEO_OUTPUT_IMAGE_QUALITY, VIDEO_OUTPUT_IMAGE_TYPE, VIDEO_OUTPUT_IMAGE_PORT, VIDEO_OUTPUT_IMAGE_PASSWORD, \
    NOTIFICATION_ENABLED, SECTION_NOTIFICATION, NOTIFICATION_DEVICE_ID, NOTIFICATION_DATA_AGGREGATION_TIME_MS, \
    DISPLAY_COUNT_ENABLED, NOTIFICATION_CAMERA_ID, SECTION_DISPLAY, \
    DISPLAY_FPS_ENABLED, DISPLAY_TIME_IN_ZONE_ENABLED, DISPLAY_VIDEO_INFO_ENABLED, DISPLAY_ALERT_ICON_ENABLED, \
    VIDEO_OUTPUT_STREAM_PATH, MODEL_SKIP_FRAMES, DISPLAY_TIME, MODEL_PRECISION, MODEL_SKIP_FRAMES_MASK, \
    SECTION_BENCHMARK, BENCHMARK_RESULTS_FILE_NAME, BENCHMARK_DATA_AGGREGATION_TIME_MS, \
    BENCHMARK_WARMUP_TIMS_MS, BENCHMARK_DURATION_TIME_MS, DISPLAY_COUNT_CATEGORIES, MODEL_CATEGORIES, \
    SCENARIO_IN_ZONE_CATEGORIES, SCENARIO_DOOR_CATEGORIES, VIDEO_OUTPUT_STREAM_HLS_BANDWIDTH, \
    VIDEO_OUTPUT_STREAM_HLS_TIME, \
    VIDEO_OUTPUT_STREAM_HLS_GOP, SCENARIO_IN_ZONE_COLD_DOWN_TIME_S, SCENARIO_IN_ZONE_TIME_LIMIT_S, \
    SCENARIO_IN_ZONE_DANGER_LIMIT, ALERT_IN_ZONE_ICON, SECTION_ALERT, VIDEO_SOURCE_IMAGE, VIDEO_SOURCE_MODE, \
    SCENARIO_PARKING, SCENARIO_PARKING_COORDS, SCENARIO_PARKING_COLD_DOWN_TIME_S, SCENARIO_PARKING_TIME_LIMIT_S, \
    SCENARIO_PARKING_CATEGORIES, SCENARIO_PARKING_DANGER_LIMIT, SCENARIO_DOOR_ENTERING_ENABLED, \
    SCENARIO_DOOR_LEAVING_ENABLED, SCENARIO_DOOR_LEAVING_LABEL, SCENARIO_DOOR_ENTERING_LABEL, MODEL_FILENAME
from fvgvisionai.config.file_property_loader import load_properties_from_file
from fvgvisionai.processor.categories import ModelCategory


class AppSettings:

    def __init__(self, cli_args: Dict[str, str] = None, file_name: str = None, config: ConfigParser = None,
                 benchmark_mode: bool = False):
        load_from_environment = file_name is None
        # Crea un handler per i log
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)
        # create console handler with a higher log level
        ch = logging.StreamHandler()

        if not load_from_environment:
            properties = load_properties_from_file(file_name)
            # override .ini file with args
            properties.update(cli_args)

            # section SECTION_SETTINGS
            log_level = from_props(properties, config, LOGGING_LEVEL, SECTION_SETTINGS)

            # section SECTION_VIDEO_SOURCE
            self._video_source = from_props(properties, config, VIDEO_SOURCE, SECTION_VIDEO_SOURCE)
            self._video_source_image = from_props(properties, config, VIDEO_SOURCE_IMAGE, SECTION_VIDEO_SOURCE)
            self._video_source_mode = to_video_source_mode(
                from_props(properties, config, VIDEO_SOURCE_MODE, SECTION_VIDEO_SOURCE))
            self._video_source_forced_fps_enabled = to_bool(from_props(properties, config,
                                                                       VIDEO_SOURCE_FORCED_FPS_ENABLED,
                                                                       SECTION_VIDEO_SOURCE))
            self._video_source_forced_fps = int(from_props(properties, config,
                                                           VIDEO_SOURCE_FORCED_FPS_VALUE, SECTION_VIDEO_SOURCE))
            self._video_source_forced_resolution_enabled = to_bool(from_props(properties, config,
                                                                              VIDEO_SOURCE_FORCED_RESOLUTION_ENABLED,
                                                                              SECTION_VIDEO_SOURCE))
            self._video_source_forced_width, self._video_source_forced_height = extract_dimensions(
                from_props(properties, config, VIDEO_SOURCE_FORCED_RESOLUTION_VALUE, SECTION_VIDEO_SOURCE))

            # section MODEL
            self._model_library = to_model_library(from_props(properties, config, MODEL_LIBRARY, SECTION_MODEL))
            self._model_id = to_model_id(from_props(properties, config, MODEL_ID, SECTION_MODEL))
            self._model_filename = to_model_filename(from_props(properties, config, MODEL_FILENAME, SECTION_MODEL))
            self._model_skip_frames_enabled = to_bool(from_props(properties, config, MODEL_SKIP_FRAMES, SECTION_MODEL))
            self._model_skip_frames_mask = to_binary_array(
                from_props(properties, config, MODEL_SKIP_FRAMES_MASK, SECTION_MODEL))
            self._model_confidence = float(from_props(properties, config, MODEL_CONFIDENCE, SECTION_MODEL))
            self._model_use_tensort = to_bool(from_props(properties, config, MODEL_USE_TENSORT, SECTION_MODEL))
            self._model_iou = float(from_props(properties, config, MODEL_IOU, SECTION_MODEL))
            self._model_resolution = to_model_resolution(
                from_props(properties, config, MODEL_IMAGE_SIZE, SECTION_MODEL))
            self._model_size = to_model_size(from_props(properties, config, MODEL_SIZE, SECTION_MODEL))
            self._model_precision = to_model_precision(from_props(properties, config, MODEL_PRECISION, SECTION_MODEL))
            self._model_tracking_enabled = to_bool(from_props(properties, config, MODEL_DO_TRACKING, SECTION_MODEL))
            self._model_categories = to_model_categories(
                from_props(properties, config, MODEL_CATEGORIES, SECTION_MODEL))

            # section SECTION_SPACE_ANALYSIS
            # scenario door
            self._scenario_door_enabled = to_bool(
                from_props(properties, config, SCENARIO_DOOR, SECTION_SPACE_ANALYSIS))
            self._scenario_door_categories = to_model_categories(
                from_props(properties, config, SCENARIO_DOOR_CATEGORIES, SECTION_SPACE_ANALYSIS))
            self._scenario_door_line, self._scenario_door_leaving_poly, self._scenario_door_enter_poly = (
                to_door_poly(from_props(properties, config, SCENARIO_DOOR_COORDS, SECTION_SPACE_ANALYSIS)))
            self._scenario_door_entering_enabled = to_bool(
                from_props(properties, config, SCENARIO_DOOR_ENTERING_ENABLED, SECTION_SPACE_ANALYSIS))
            self._scenario_door_entering_label = to_str(from_props(properties, config, SCENARIO_DOOR_ENTERING_LABEL,
                                                                   SECTION_SPACE_ANALYSIS))
            self._scenario_door_leaving_enabled = to_bool(
                from_props(properties, config, SCENARIO_DOOR_LEAVING_ENABLED, SECTION_SPACE_ANALYSIS))
            self._scenario_door_leaving_label = to_str(from_props(properties, config, SCENARIO_DOOR_LEAVING_LABEL,
                                                                  SECTION_SPACE_ANALYSIS))
            # scenario pose
            self._scenario_pose_enabled = to_bool(
                from_props(properties, config, SCENARIO_RAISED_HAND, SECTION_SPACE_ANALYSIS))
            # scenario in zone
            self._scenario_zone_enabled = to_bool(
                from_props(properties, config, SCENARIO_IN_ZONE, SECTION_SPACE_ANALYSIS))
            self._scenario_zone_coords = to_zone_poly(
                from_props(properties, config, SCENARIO_IN_ZONE_COORDS, SECTION_SPACE_ANALYSIS))
            self._scenario_zone_cold_down_time = int(
                from_props(properties, config, SCENARIO_IN_ZONE_COLD_DOWN_TIME_S, SECTION_SPACE_ANALYSIS))
            self._scenario_zone_time_limit = int(
                from_props(properties, config, SCENARIO_IN_ZONE_TIME_LIMIT_S, SECTION_SPACE_ANALYSIS))
            self._scenario_zone_categories = to_model_categories(
                from_props(properties, config, SCENARIO_IN_ZONE_CATEGORIES, SECTION_SPACE_ANALYSIS))
            self._scenario_zone_danger_limit = int(
                from_props(properties, config, SCENARIO_IN_ZONE_DANGER_LIMIT, SECTION_SPACE_ANALYSIS))

            # scenario parking
            self._scenario_parking_enabled = to_bool(
                from_props(properties, config, SCENARIO_PARKING, SECTION_SPACE_ANALYSIS))
            self._scenario_parking_coords = to_parking_list(
                from_props(properties, config, SCENARIO_PARKING_COORDS, SECTION_SPACE_ANALYSIS))
            self._scenario_parking_cold_down_time = int(
                from_props(properties, config, SCENARIO_PARKING_COLD_DOWN_TIME_S, SECTION_SPACE_ANALYSIS))
            self._scenario_parking_time_limit = int(
                from_props(properties, config, SCENARIO_PARKING_TIME_LIMIT_S, SECTION_SPACE_ANALYSIS))
            self._scenario_parking_categories = to_model_categories(
                from_props(properties, config, SCENARIO_PARKING_CATEGORIES, SECTION_SPACE_ANALYSIS))
            self._scenario_parking_danger_limit = int(
                from_props(properties, config, SCENARIO_PARKING_DANGER_LIMIT, SECTION_SPACE_ANALYSIS))

            # section SECTION_ALERT
            self._alert_in_zone_icon = from_props(properties, config, ALERT_IN_ZONE_ICON, SECTION_ALERT)

            # section SECTION_VIDEO_OUTPUT
            self._video_output_fps = int(from_props(properties, config, VIDEO_OUTPUT_FPS, SECTION_VIDEO_OUTPUT))
            self._video_output_stream = to_bool(
                from_props(properties, config, VIDEO_OUTPUT_STREAM, SECTION_VIDEO_OUTPUT))
            self._video_output_stream_path = from_props(properties, config, VIDEO_OUTPUT_STREAM_PATH,
                                                        SECTION_VIDEO_OUTPUT)
            self._video_output_stream_bandwidth = int(
                from_props(properties, config, VIDEO_OUTPUT_STREAM_HLS_BANDWIDTH, SECTION_VIDEO_OUTPUT))
            self._video_output_stream_hls_time = int(
                from_props(properties, config, VIDEO_OUTPUT_STREAM_HLS_TIME, SECTION_VIDEO_OUTPUT))
            self._video_output_stream_hls_gop = int(
                from_props(properties, config, VIDEO_OUTPUT_STREAM_HLS_GOP, SECTION_VIDEO_OUTPUT))

            # section SECTION_VIDEO_OUTPUT
            self._video_output_rtsp = to_bool(from_props(properties, config, VIDEO_OUTPUT_RTSP, SECTION_VIDEO_OUTPUT))
            self._video_output_rtsp_url = from_props(properties, config, VIDEO_OUTPUT_RTSP_URL, SECTION_VIDEO_OUTPUT)
            self._video_output_rtsp_gpu_nvidia = to_bool(from_props(properties, config, VIDEO_OUTPUT_RTSP_GPU_NVIDIA, SECTION_VIDEO_OUTPUT))
            self._video_output_rtsp_risoluzione = from_props(properties, config, VIDEO_OUTPUT_RTSP_RISOLUZIONE, SECTION_VIDEO_OUTPUT)

            # section SECTION_VIDEO_OUTPUT
            self._video_output_image = to_bool(from_props(properties, config, VIDEO_OUTPUT_IMAGE, SECTION_VIDEO_OUTPUT))
            self._video_output_image_quality = int(
                from_props(properties, config, VIDEO_OUTPUT_IMAGE_QUALITY, SECTION_VIDEO_OUTPUT))
            self._video_output_image_type = to_image_type(
                from_props(properties, config, VIDEO_OUTPUT_IMAGE_TYPE, SECTION_VIDEO_OUTPUT))
            self._video_output_image_port = int(
                from_props(properties, config, VIDEO_OUTPUT_IMAGE_PORT, SECTION_VIDEO_OUTPUT))
            self._video_output_image_password = from_props(properties, config, VIDEO_OUTPUT_IMAGE_PASSWORD,
                                                           SECTION_VIDEO_OUTPUT)

            # section NOTIFICATION
            self._enabled_notification = to_bool(from_props(properties, config,
                                                            NOTIFICATION_ENABLED, SECTION_NOTIFICATION))
            self._azure_connection_string = from_props(properties, config, NOTIFICATION_AZURE_CONNECTION_STRING,
                                                       SECTION_NOTIFICATION)
            self._notification_device_id = from_props(properties, config, NOTIFICATION_DEVICE_ID, SECTION_NOTIFICATION)
            self._notification_camera_id = from_props(properties, config, NOTIFICATION_CAMERA_ID, SECTION_NOTIFICATION)
            self._notification_aggregation_time_ms = int(
                from_props(properties, config, NOTIFICATION_DATA_AGGREGATION_TIME_MS, SECTION_NOTIFICATION))

            # section DISPLAY
            self._show_count = to_bool(from_props(properties, config, DISPLAY_COUNT_ENABLED, SECTION_DISPLAY))
            self._show_categories = to_model_categories(
                from_props(properties, config, DISPLAY_COUNT_CATEGORIES, SECTION_DISPLAY))

            self._show_fps = to_bool(from_props(properties, config, DISPLAY_FPS_ENABLED, SECTION_DISPLAY))
            self._show_time_in_zone = to_bool(
                from_props(properties, config, DISPLAY_TIME_IN_ZONE_ENABLED, SECTION_DISPLAY))
            self._show_time = to_bool(from_props(properties, config, DISPLAY_TIME, SECTION_DISPLAY))

            self._show_video_info = to_bool(from_props(properties, config, DISPLAY_VIDEO_INFO_ENABLED, SECTION_DISPLAY))
            self._show_alert_icon = to_bool(from_props(properties, config, DISPLAY_ALERT_ICON_ENABLED, SECTION_DISPLAY))

            # section BENCHMARK
            self._benchmark_duration_time_ms = int(
                from_props(properties, config, BENCHMARK_DURATION_TIME_MS, SECTION_BENCHMARK))
            self._benchmark_result_file_name = from_props(properties, config, BENCHMARK_RESULTS_FILE_NAME,
                                                          SECTION_BENCHMARK)
            self._benchmark_aggregation_time_ms = int(
                from_props(properties, config, BENCHMARK_DATA_AGGREGATION_TIME_MS, SECTION_BENCHMARK))
            self._benchmark_warmup_time_ms = int(
                from_props(properties, config, BENCHMARK_WARMUP_TIMS_MS, SECTION_BENCHMARK))
        else:
            # section SECTION_SETTINGS
            log_level = from_env(config, LOGGING_LEVEL, SECTION_SETTINGS, cli_args)

            # section SECTION_VIDEO_SOURCE
            self._video_source = from_env(config, VIDEO_SOURCE, SECTION_VIDEO_SOURCE, cli_args)
            self._video_source_image = from_env(config, VIDEO_SOURCE_IMAGE, SECTION_VIDEO_SOURCE, cli_args)
            self._video_source_mode = to_video_source_mode(
                from_env(config, VIDEO_SOURCE_MODE, SECTION_VIDEO_SOURCE, cli_args))
            self._video_source_forced_fps_enabled = to_bool(
                from_env(config, VIDEO_SOURCE_FORCED_FPS_ENABLED, SECTION_VIDEO_SOURCE, cli_args))
            self._video_source_forced_fps = int(
                from_env(config, VIDEO_SOURCE_FORCED_FPS_VALUE, SECTION_VIDEO_SOURCE, cli_args))
            self._video_source_forced_resolution_enabled = to_bool(from_env(config,
                                                                            VIDEO_SOURCE_FORCED_RESOLUTION_ENABLED,
                                                                            SECTION_VIDEO_SOURCE, cli_args))
            self._video_source_forced_width, self._video_source_forced_height = extract_dimensions(
                from_env(config, VIDEO_SOURCE_FORCED_RESOLUTION_VALUE, SECTION_VIDEO_SOURCE, cli_args))

            # section MODEL
            self._model_id = to_model_id(from_env(config, MODEL_ID, SECTION_MODEL, cli_args))
            self._model_library = to_model_library(from_env(config, MODEL_LIBRARY, SECTION_MODEL, cli_args))
            self._model_filename = to_model_filename(from_env(config, MODEL_FILENAME, SECTION_MODEL, cli_args))
            self._model_skip_frames_enabled = to_bool(from_env(config, MODEL_SKIP_FRAMES, SECTION_MODEL, cli_args))
            self._model_skip_frames_mask = to_binary_array(
                from_env(config, MODEL_SKIP_FRAMES_MASK, SECTION_MODEL, cli_args))
            self._model_confidence = float(from_env(config, MODEL_CONFIDENCE, SECTION_MODEL, cli_args))
            self._model_iou = float(from_env(config, MODEL_IOU, SECTION_MODEL, cli_args))
            self._model_resolution = to_model_resolution(from_env(config, MODEL_IMAGE_SIZE, SECTION_MODEL, cli_args))
            self._model_size = to_model_size(from_env(config, MODEL_SIZE, SECTION_MODEL, cli_args))
            self._model_use_tensort = to_bool(from_env(config, MODEL_USE_TENSORT, SECTION_MODEL, cli_args))
            self._model_precision = to_model_precision(from_env(config, MODEL_PRECISION, SECTION_MODEL, cli_args))
            self._model_tracking_enabled = to_bool(from_env(config, MODEL_DO_TRACKING, SECTION_MODEL, cli_args))
            self._model_categories = to_model_categories(from_env(config, MODEL_CATEGORIES, SECTION_MODEL, cli_args))

            # section SECTION_SPACE_ANALYSIS
            # scenario zone
            self._scenario_zone_enabled = to_bool(from_env(config, SCENARIO_IN_ZONE, SECTION_SPACE_ANALYSIS, cli_args))
            self._scenario_zone_coords = to_zone_poly(
                from_env(config, SCENARIO_IN_ZONE_COORDS, SECTION_SPACE_ANALYSIS, cli_args))
            self._scenario_zone_categories = to_model_categories(
                from_env(config, SCENARIO_IN_ZONE_CATEGORIES, SECTION_SPACE_ANALYSIS, cli_args))
            self._scenario_zone_cold_down_time = int(
                from_env(config, SCENARIO_IN_ZONE_COLD_DOWN_TIME_S, SECTION_SPACE_ANALYSIS, cli_args))
            self._scenario_zone_time_limit = int(
                from_env(config, SCENARIO_IN_ZONE_TIME_LIMIT_S, SECTION_SPACE_ANALYSIS, cli_args))
            self._scenario_zone_danger_limit = int(
                from_env(config, SCENARIO_IN_ZONE_DANGER_LIMIT, SECTION_SPACE_ANALYSIS, cli_args))
            # scenario door
            self._scenario_door_line, self._scenario_door_leaving_poly, self._scenario_door_enter_poly = (
                to_door_poly(from_env(config, SCENARIO_DOOR_COORDS, SECTION_SPACE_ANALYSIS, cli_args)))
            self._scenario_door_enabled = to_bool(from_env(config, SCENARIO_DOOR, SECTION_SPACE_ANALYSIS, cli_args))
            self._scenario_door_categories = to_model_categories(
                from_env(config, SCENARIO_DOOR_CATEGORIES, SECTION_SPACE_ANALYSIS, cli_args))
            self._scenario_door_entering_enabled = to_bool(
                from_env(config, SCENARIO_DOOR_ENTERING_ENABLED, SECTION_SPACE_ANALYSIS, cli_args))
            self._scenario_door_entering_label = to_str(from_env(config, SCENARIO_DOOR_ENTERING_LABEL,
                                                                 SECTION_SPACE_ANALYSIS, cli_args))
            self._scenario_door_leaving_enabled = to_bool(
                from_env(config, SCENARIO_DOOR_LEAVING_ENABLED, SECTION_SPACE_ANALYSIS, cli_args))
            self._scenario_door_leaving_label = to_str(from_env(config, SCENARIO_DOOR_LEAVING_LABEL,
                                                                SECTION_SPACE_ANALYSIS, cli_args))
            # scenario pose
            self._scenario_pose_enabled = to_bool(
                from_env(config, SCENARIO_RAISED_HAND, SECTION_SPACE_ANALYSIS, cli_args))
            # scenario parking
            self._scenario_parking_enabled = to_bool(
                from_env(config, SCENARIO_PARKING, SECTION_SPACE_ANALYSIS, cli_args))
            self._scenario_parking_coords = to_parking_list(
                from_env(config, SCENARIO_PARKING_COORDS, SECTION_SPACE_ANALYSIS, cli_args))
            self._scenario_parking_cold_down_time = int(
                from_env(config, SCENARIO_PARKING_COLD_DOWN_TIME_S, SECTION_SPACE_ANALYSIS, cli_args))
            self._scenario_parking_time_limit = int(
                from_env(config, SCENARIO_PARKING_TIME_LIMIT_S, SECTION_SPACE_ANALYSIS, cli_args))
            self._scenario_parking_categories = to_model_categories(
                from_env(config, SCENARIO_PARKING_CATEGORIES, SECTION_SPACE_ANALYSIS, cli_args))
            self._scenario_parking_danger_limit = int(
                from_env(config, SCENARIO_PARKING_DANGER_LIMIT, SECTION_SPACE_ANALYSIS, cli_args))

            # section SECTION_VIDEO_OUTPUT
            self._video_output_image_port = int(
                from_env(config, VIDEO_OUTPUT_IMAGE_PORT, SECTION_VIDEO_OUTPUT, cli_args))
            self._video_output_image_password = from_env(config, VIDEO_OUTPUT_IMAGE_PASSWORD, SECTION_VIDEO_OUTPUT,
                                                         cli_args)
            self._video_output_stream = to_bool(from_env(config, VIDEO_OUTPUT_STREAM, SECTION_VIDEO_OUTPUT, cli_args))
            self._video_output_stream_path = from_env(config, VIDEO_OUTPUT_STREAM_PATH, SECTION_VIDEO_OUTPUT, cli_args)
            self._video_output_stream_bandwidth = int(
                from_env(config, VIDEO_OUTPUT_STREAM_HLS_BANDWIDTH, SECTION_VIDEO_OUTPUT, cli_args))
            self._video_output_stream_hls_time = int(
                from_env(config, VIDEO_OUTPUT_STREAM_HLS_TIME, SECTION_VIDEO_OUTPUT, cli_args))
            self._video_output_stream_hls_gop = int(
                from_env(config, VIDEO_OUTPUT_STREAM_HLS_GOP, SECTION_VIDEO_OUTPUT, cli_args))

            # section SECTION_VIDEO_OUTPUT_RTSP
            self._video_output_rtsp = to_bool(from_env(config, VIDEO_OUTPUT_RTSP, SECTION_VIDEO_OUTPUT, cli_args))
            self._video_output_rtsp_url = from_env(config, VIDEO_OUTPUT_RTSP_URL, SECTION_VIDEO_OUTPUT, cli_args)
            self._video_output_rtsp_gpu_nvidia = to_bool(
                from_env(config, VIDEO_OUTPUT_RTSP_GPU_NVIDIA, SECTION_VIDEO_OUTPUT, cli_args))
            self._video_output_rtsp_risoluzione = from_env(config, VIDEO_OUTPUT_RTSP_RISOLUZIONE,
                                                             SECTION_VIDEO_OUTPUT, cli_args)

            self._video_output_fps = int(from_env(config, VIDEO_OUTPUT_FPS, SECTION_VIDEO_OUTPUT, cli_args))
            self._video_output_image = to_bool(from_env(config, VIDEO_OUTPUT_IMAGE, SECTION_VIDEO_OUTPUT, cli_args))
            self._video_output_image_quality = int(
                from_env(config, VIDEO_OUTPUT_IMAGE_QUALITY, SECTION_VIDEO_OUTPUT, cli_args))
            self._video_output_image_type = to_image_type(
                from_env(config, VIDEO_OUTPUT_IMAGE_TYPE, SECTION_VIDEO_OUTPUT, cli_args))

            # section SECTION_ALERT
            self._alert_in_zone_icon = from_env(config, ALERT_IN_ZONE_ICON, SECTION_ALERT, cli_args)

            # section NOTIFICATION
            self._enabled_notification = to_bool(from_env(config, NOTIFICATION_ENABLED, SECTION_NOTIFICATION, cli_args))
            self._azure_connection_string = from_env(config, NOTIFICATION_AZURE_CONNECTION_STRING, SECTION_NOTIFICATION,
                                                     cli_args)
            self._notification_device_id = from_env(config, NOTIFICATION_DEVICE_ID, SECTION_NOTIFICATION, cli_args)
            self._notification_camera_id = from_env(config, NOTIFICATION_CAMERA_ID, SECTION_NOTIFICATION, cli_args)
            self._notification_aggregation_time_ms = int(
                from_env(config, NOTIFICATION_DATA_AGGREGATION_TIME_MS, SECTION_NOTIFICATION, cli_args))

            # section DISPLAY
            self._show_count = to_bool(from_env(config, DISPLAY_COUNT_ENABLED, SECTION_DISPLAY, cli_args))
            self._show_categories = to_model_categories(
                from_env(config, DISPLAY_COUNT_CATEGORIES, SECTION_DISPLAY, cli_args))
            self._show_fps = to_bool(from_env(config, DISPLAY_FPS_ENABLED, SECTION_DISPLAY, cli_args))
            self._show_time_in_zone = to_bool(from_env(config, DISPLAY_TIME_IN_ZONE_ENABLED, SECTION_DISPLAY, cli_args))
            self._show_video_info = to_bool(from_env(config, DISPLAY_VIDEO_INFO_ENABLED, SECTION_DISPLAY, cli_args))
            self._show_alert_icon = to_bool(from_env(config, DISPLAY_ALERT_ICON_ENABLED, SECTION_DISPLAY, cli_args))
            self._show_time = to_bool(from_env(config, DISPLAY_TIME, SECTION_DISPLAY, cli_args))

            # section BENCHMARK
            self._benchmark_duration_time_ms = int(
                from_env(config, BENCHMARK_DURATION_TIME_MS, SECTION_BENCHMARK, cli_args))
            self._benchmark_result_file_name = from_env(config, BENCHMARK_RESULTS_FILE_NAME, SECTION_BENCHMARK,
                                                        cli_args)
            self._benchmark_aggregation_time_ms = int(
                from_env(config, BENCHMARK_DATA_AGGREGATION_TIME_MS, SECTION_BENCHMARK, cli_args))
            self._benchmark_warmup_time_ms = int(
                from_env(config, BENCHMARK_WARMUP_TIMS_MS, SECTION_BENCHMARK, cli_args))

        ch.setLevel(logging.getLevelName(log_level))
        ch.setFormatter(ColoredFormatter())
        logger.addHandler(ch)

        self._logger = logging.getLogger(__name__)

        if self._model_use_tensort and not is_tensorrt_installed():
            self._model_use_tensort = False
            self._logger.warning("Tensorrt is not present in the system")

        # Se non esiste maschera, ne creiamo una che fa lavorare sempre il modello
        if not self._model_skip_frames_enabled:
            self._model_skip_frames_mask = [True]

        # Se abbiamo scenario in zone e da visualizzare show time in zone, il tracking deve essere abilitato
        if self._scenario_zone_enabled and self._show_time_in_zone and not self._model_tracking_enabled:
            self._logger.warning(
                "Scenario IN_ZONE and  DISPLAY_TIME_PEOPLE_IN_ZONE True requires MODEL_DO_TRACKING to be true!")
            self._model_tracking_enabled = True

        # Se abbiamo scenario in pose, verrà abilitato il pose e verranno rilevati solo persone
        if self._scenario_pose_enabled:
            self._logger.warning(
                "Model POSE is enabled, so only PERSON can be detected!")

        # Se siamo in modalita' benchmark, le notifiche devono essere per forza disabilitate
        if benchmark_mode and self._enabled_notification:
            self._logger.warning(
                "In benchmark mode, the notification is disabled!")
            self._enabled_notification = False

        self._benchmark_mode = benchmark_mode

    def show_properties(self):
        # Elenca le proprietà (compresi i metodi)
        elenco_proprieta = [attr for attr in dir(self) if
                            not callable(getattr(self, attr)) and not attr.startswith("_")]

        for prop in elenco_proprieta:
            value = getattr(self, prop)
            if isinstance(value, np.ndarray):
                value_str = get_array_as_string(value)
            elif isinstance(value, List):
                value_str = get_array_as_string(value)
            else:
                value_str = str(value)
            self._logger.info(f"{prop}: {value_str}")

    @property
    def video_output_image_port(self) -> int:
        return self._video_output_image_port

    @property
    def video_output_image_password(self) -> str:
        return self._video_output_image_password

    @property
    def model_use_tensort(self) -> bool:
        return self._model_use_tensort

    @property
    def model_skip_frames_enabled(self) -> bool:
        return self._model_skip_frames_enabled

    @property
    def model_skip_frames_mask(self) -> List[bool]:
        return self._model_skip_frames_mask

    @property
    def azure_connection_string(self) -> str:
        return self._azure_connection_string

    @property
    def enabled_notification(self) -> bool:
        return self._enabled_notification

    @property
    def model_tracking_enabled(self) -> bool:
        return self._model_tracking_enabled

    @property
    def video_source(self) -> str:
        return self._video_source

    @property
    def alert_in_zone_icon(self) -> str:
        return self._alert_in_zone_icon

    @property
    def notification_device_id(self) -> str:
        return self._notification_device_id

    @property
    def notification_camera_id(self) -> str:
        return self._notification_camera_id

    @property
    def notification_aggregation_time_ms(self) -> int:
        return self._notification_aggregation_time_ms

    @property
    def model_library(self) -> ModelLibrary:
        return self._model_library

    @property
    def model_id(self) -> ModelId:
        return self._model_id

    @property
    def model_filename(self) -> Optional[str]:
        return self._model_filename

    @property
    def model_confidence(self) -> float:
        return self._model_confidence

    @property
    def model_iou(self) -> float:
        return self._model_iou

    @property
    def model_resolution(self) -> ModelResolution:
        return self._model_resolution

    @property
    def model_width(self) -> int:
        return self._model_resolution.value["resolution"][1]

    @property
    def model_height(self) -> int:
        return self._model_resolution.value["resolution"][0]

    @property
    def model_categories(self) -> List[ModelCategory]:
        return self._model_categories

    @property
    def model_size(self) -> ModelSize:
        return self._model_size

    @property
    def model_precision(self) -> ModelPrecision:
        return self._model_precision

    @property
    def scenario_zone_coords(self) -> np.array:
        return self._scenario_zone_coords

    @property
    def scenario_zone_categories(self) -> List[ModelCategory]:
        return self._scenario_zone_categories

    @property
    def scenario_door_enabled(self) -> bool:
        return self._scenario_door_enabled

    @property
    def scenario_door_categories(self) -> List[ModelCategory]:
        return self._scenario_door_categories

    @property
    def scenario_door_line(self) -> np.array:
        return self._scenario_door_line

    @property
    def scenario_door_leaving_poly(self) -> np.array:
        return self._scenario_door_leaving_poly

    @property
    def scenario_door_enter_poly(self) -> np.array:
        return self._scenario_door_enter_poly

    @property
    def scenario_door_entering_enabled(self) -> bool:
        return self._scenario_door_entering_enabled

    @property
    def scenario_door_leaving_enabled(self) -> bool:
        return self._scenario_door_leaving_enabled

    @property
    def scenario_door_entering_label(self) -> str:
        return self._scenario_door_entering_label

    @property
    def scenario_door_leaving_label(self) -> str:
        return self._scenario_door_leaving_label

    @property
    def scenario_zone_cold_down_time(self) -> int:
        return self._scenario_zone_cold_down_time

    @property
    def scenario_zone_time_limit(self) -> int:
        return self._scenario_zone_time_limit

    @property
    def scenario_zone_danger_limit(self) -> int:
        return self._scenario_zone_danger_limit

    @property
    def scenario_zone_enabled(self) -> bool:
        return self._scenario_zone_enabled

    @property
    def scenario_parking_coords(self) -> List[np.array]:
        return self._scenario_parking_coords

    @property
    def scenario_parking_categories(self) -> List[ModelCategory]:
        return self._scenario_parking_categories

    @property
    def scenario_parking_cold_down_time(self) -> int:
        return self._scenario_parking_cold_down_time

    @property
    def scenario_parking_time_limit(self) -> int:
        return self._scenario_parking_time_limit

    @property
    def scenario_parking_danger_limit(self) -> int:
        return self._scenario_parking_danger_limit

    @property
    def scenario_parking_enabled(self) -> bool:
        return self._scenario_parking_enabled

    @property
    def video_source_image(self) -> str:
        return self._video_source_image

    @property
    def video_source_mode(self) -> VideoSourceMode:
        return self._video_source_mode

    @property
    def video_source_forced_fps_enabled(self) -> bool:
        return self._video_source_forced_fps_enabled

    @property
    def video_source_forced_fps(self) -> int:
        return self._video_source_forced_fps

    @property
    def video_source_forced_resolution_enabled(self) -> bool:
        return self._video_source_forced_resolution_enabled

    @property
    def video_source_forced_width(self) -> int:
        return self._video_source_forced_width

    @property
    def video_source_forced_height(self) -> int:
        return self._video_source_forced_height

    @property
    def video_output_stream_path(self) -> str:
        return self._video_output_stream_path

    @property
    def video_output_stream_bandwidth(self) -> int:
        return self._video_output_stream_bandwidth

    @property
    def video_output_stream_hls_time(self) -> int:
        return self._video_output_stream_hls_time

    @property
    def video_output_stream_hls_gop(self) -> int:
        return self._video_output_stream_hls_gop

    @property
    def video_output_stream(self) -> bool:
        return self._video_output_stream
    
    @property
    def video_output_rtsp(self) -> bool:
        return self._video_output_rtsp

    @property
    def video_output_rtsp_url(self) -> str:
        return self._video_output_rtsp_url

    @property
    def video_output_rtsp_gpu_nvidia(self) -> bool:
        return self._video_output_rtsp_gpu_nvidia

    @property
    def video_output_rtsp_risoluzione(self) -> str:
        return self._video_output_rtsp_risoluzione

    @property
    def video_output_fps(self) -> int:
        return self._video_output_fps

    @property
    def video_output_image(self) -> bool:
        return self._video_output_image

    @property
    def video_output_image_quality(self) -> int:
        return self._video_output_image_quality

    @property
    def video_output_image_type(self) -> ImageType:
        return self._video_output_image_type

    @property
    def scenario_pose_enabled(self) -> bool:
        return self._scenario_pose_enabled

    @property
    def show_count_enabled(self) -> bool:
        return self._show_count

    @property
    def show_categories(self) -> List[ModelCategory]:
        return self._show_categories

    @property
    def show_time_people_in_zone(self) -> bool:
        return self._show_time_in_zone

    @property
    def show_time(self) -> bool:
        return self._show_time

    @property
    def show_fps_enabled(self) -> bool:
        return self._show_fps

    @property
    def show_video_info_enabled(self) -> bool:
        return self._show_video_info

    @property
    def show_alert_icon(self) -> bool:
        return self._show_alert_icon

    @property
    def benchmark_enabled(self) -> bool:
        return self._benchmark_mode

    @property
    def benchmark_duration_time_ms(self) -> int:
        return self._benchmark_duration_time_ms

    @property
    def benchmark_results_file_name(self) -> str:
        return self._benchmark_result_file_name

    @property
    def benchmark_aggregation_time_ms(self) -> int:
        return self._benchmark_aggregation_time_ms

    @property
    def benchmark_warmup_time_ms(self) -> int:
        return self._benchmark_warmup_time_ms


def load_settings_from_file(file_name: str, cli_args: Dict[str, str], config: ConfigParser,
                            benchmark_mode=False) -> AppSettings:
    return AppSettings(file_name=file_name, cli_args=cli_args, config=config, benchmark_mode=benchmark_mode)


def load_settings_from_env(cli_args: Dict[str, str], config: ConfigParser, demo_mode=False) -> AppSettings:
    return AppSettings(cli_args=cli_args, config=config, benchmark_mode=demo_mode)
