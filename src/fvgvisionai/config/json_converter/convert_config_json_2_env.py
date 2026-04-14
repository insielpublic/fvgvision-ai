import json
import os
import string
import re

from fvgvisionai.config.json_converter.node_templates import NodeTemplateEnum


def read_json_file(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)


def process_node(node) -> (any, int, int):
    processed_data = {}
    node_type = node['data']['type']
    node_payload = node['data']['payload']
    input_width = 0
    input_height = 0

    print(node_type)
    print(node_payload)
    if node_type == NodeTemplateEnum.INPUT_STREAM.value:
        process_input_stream(node_payload, processed_data)
        input_width = node['data']['payload']['preview']['original_width']
        input_height = node['data']['payload']['preview']['original_height']
    elif node_type == NodeTemplateEnum.INPUT_RESIZER.value:
        process_input_resizer(node_payload, processed_data)
    elif node_type == NodeTemplateEnum.INPUT_STREAM_FPS_LIMITER.value:
        process_input_stream_fps_limiter(node_payload, processed_data)
    elif node_type == NodeTemplateEnum.DISPLAY.value:
        process_display(node_payload, processed_data)
    elif node_type == NodeTemplateEnum.ULTRALYTICS_PROCESSING.value:
        process_ultralytics_processing(node_payload, processed_data)
    elif node_type == NodeTemplateEnum.ULTRALYTICS_CUSTOM_MODEL_PROCESSING.value:
        process_custom_ultralytics_processing(node_payload, processed_data)
    elif node_type == NodeTemplateEnum.SKIP_FRAMES.value:
        process_skip_frames(node_payload, processed_data)
    elif node_type == NodeTemplateEnum.OUTPUT_STREAM_FPS_LIMITER.value:
        process_output_skip_frames(node_payload, processed_data)
    elif node_type == NodeTemplateEnum.OUTPUT_STREAM.value:
        process_output_stream(node_payload, processed_data)
    elif node_type == NodeTemplateEnum.OUTPUT_RTSP.value:
        process_output_rtsp_stream(node_payload, processed_data)
    elif node_type == NodeTemplateEnum.IOT_HUB_NOTIFICATION.value:
        process_iot_hub_notification(node_payload, processed_data)
    elif node_type == NodeTemplateEnum.SCENARIO_IN_ZONE.value:
        process_scenario_in_zone(node_payload, processed_data)
    elif node_type == NodeTemplateEnum.OUTPUT_IMAGE.value:
        process_output_image(node_payload, processed_data)
    elif node_type == NodeTemplateEnum.SCENARIO_DOOR.value:
        process_scenario_door(node_payload, processed_data)
    elif node_type == NodeTemplateEnum.SCENARIO_PARKING.value:
        process_scenario_parking(node_payload, processed_data)
    elif node_type == NodeTemplateEnum.SCENARIO_RAISED_HANDS.value:
        process_scenario_raised_hand(node_payload, processed_data)
    if 'coordinates' in node['data']['payload']:
        processed_data['coordinates'] = convert_2_string(node['data']['payload']['coordinates'])

    return processed_data, input_width, input_height


def process_display(node_payload, processed_data):
    processed_data['DISPLAY_COUNT_ENABLED'] = convert_2_bool(node_payload['count_enabled'])
    processed_data['DISPLAY_COUNT_CATEGORIES'] = convert_2_string(node_payload['count_categories'])
    processed_data['DISPLAY_TIME_IN_ZONE_ENABLED'] = convert_2_bool(node_payload['time_in_zone_enabled'])
    processed_data['DISPLAY_FPS_ENABLED'] = convert_2_bool(node_payload['fps_enabled'])
    processed_data['DISPLAY_TIME'] = convert_2_bool(node_payload['time_enabled'])
    processed_data['DISPLAY_VIDEO_INFO_ENABLED'] = convert_2_bool(node_payload['video_info_enabled'])
    processed_data['DISPLAY_ALERT_ICON_ENABLED'] = convert_2_bool(node_payload['alert_icon_enabled'])


def process_input_stream_fps_limiter(node_payload, processed_data):
    processed_data['VIDEO_SOURCE_FORCED_FPS_ENABLED'] = convert_2_bool(node_payload['enabled'])
    processed_data['VIDEO_SOURCE_FORCED_FPS_VALUE'] = convert_2_string(node_payload['fps'])


def process_input_resizer(node_payload, processed_data):
    processed_data['VIDEO_SOURCE_FORCED_RESOLUTION_ENABLED'] = convert_2_bool(node_payload['enabled'])
    processed_data['VIDEO_SOURCE_FORCED_RESOLUTION_VALUE'] = convert_2_number_sequence(node_payload['resolution'])


def process_input_stream(node_payload, processed_data):
    processed_data['VIDEO_SOURCE'] = convert_2_string(node_payload['source'])
    processed_data['VIDEO_SOURCE_MODE'] = convert_2_string(node_payload['type'])


def process_ultralytics_processing(node_payload, processed_data):
    processed_data['MODEL_ID'] = convert_2_string(node_payload['model_id'])
    processed_data['MODEL_SIZE'] = convert_2_string(node_payload['model_size'])
    processed_data['MODEL_PRECISION'] = convert_2_string(node_payload['precision']).lower()
    processed_data['MODEL_IMAGE_SIZE'] = convert_2_string(node_payload['image_size'])
    processed_data['MODEL_USE_TENSORT'] = convert_2_bool(node_payload['use_tensorrt'])
    processed_data['MODEL_CONFIDENCE'] = convert_2_number_sequence(node_payload['confidence'])
    processed_data['MODEL_IOU'] = convert_2_number_sequence(node_payload['iou'])

def process_custom_ultralytics_processing(node_payload, processed_data):
    processed_data['MODEL_FILENAME'] = convert_2_string(node_payload['model_filename'])
    processed_data['MODEL_IMAGE_SIZE'] = convert_2_string(node_payload['image_size'])
    processed_data['MODEL_CONFIDENCE'] = convert_2_number_sequence(node_payload['confidence'])
    processed_data['MODEL_IOU'] = convert_2_number_sequence(node_payload['iou'])


def process_skip_frames(node_payload, processed_data):
    processed_data['MODEL_SKIP_FRAMES'] = convert_2_string(node_payload['enabled'])
    processed_data['MODEL_SKIP_FRAMES_MASK'] = convert_2_string(node_payload['mask'])


def process_output_skip_frames(node_payload, processed_data):
    processed_data['VIDEO_OUTPUT_FPS'] = convert_2_string(node_payload['fps'])

def process_output_stream(node_payload, processed_data):
    processed_data['VIDEO_OUTPUT_STREAM'] = convert_2_bool(node_payload['enabled'])
    processed_data['VIDEO_OUTPUT_STREAM_HLS_BANDWIDTH'] = convert_2_number_sequence(node_payload['hls_bandwidth'])
    processed_data['VIDEO_OUTPUT_STREAM_HLS_TIME'] = convert_2_number_sequence(node_payload['hls_time'])
    processed_data['VIDEO_OUTPUT_STREAM_HLS_GOP'] = convert_2_number_sequence(node_payload['hls_gop'])

def process_output_rtsp_stream(node_payload, processed_data):
    processed_data['VIDEO_OUTPUT_RTSP'] = convert_2_bool(node_payload['enabled'])
    processed_data['VIDEO_OUTPUT_RTSP_URL'] = convert_2_string(node_payload['rtsp_url'])
    processed_data['VIDEO_OUTPUT_RTSP_GPU_NVIDIA'] = convert_2_bool(node_payload['rtsp_gpu_nvidia'])
    processed_data['VIDEO_OUTPUT_RTSP_RISOLUZIONE'] = convert_2_string(node_payload['rtsp_resolution'])

def process_iot_hub_notification(node_payload, processed_data):
    processed_data['NOTIFICATION_ENABLED'] = convert_2_bool(node_payload['enabled'])
    processed_data['NOTIFICATION_AZURE_CONNECTION_STRING'] = convert_2_string(node_payload['connectionString'])
    processed_data['NOTIFICATION_DEVICE_ID'] = convert_2_string(node_payload['deviceId'])
    processed_data['NOTIFICATION_CAMERA_ID'] = convert_2_string(node_payload['cameraId'])
    processed_data['NOTIFICATION_DATA_AGGREGATION_TIME_MS'] = convert_2_number_sequence(
        node_payload['dataAggregationTimeMs'])


def process_scenario_in_zone(node_payload, processed_data):
    processed_data['SCENARIO_IN_ZONE'] = convert_2_bool(node_payload['enabled'])
    processed_data['SCENARIO_IN_ZONE_COORDS'] = convert_2_string(node_payload['coords']).lower()
    processed_data['SCENARIO_IN_ZONE_COLD_DOWN_TIME_S'] = convert_2_number_sequence(node_payload['cold_down_time_s'])
    processed_data['SCENARIO_IN_ZONE_TIME_LIMIT_S'] = convert_2_number_sequence(node_payload['time_limit_s'])
    processed_data['SCENARIO_IN_ZONE_DANGER_LIMIT'] = convert_2_number_sequence(node_payload['danger_limit'])
    processed_data['SCENARIO_IN_ZONE_CATEGORIES'] = convert_2_string(node_payload['categories'])


def process_output_image(node_payload, processed_data):
    processed_data['VIDEO_OUTPUT_IMAGE'] = convert_2_bool(node_payload['enabled'])
    processed_data['VIDEO_OUTPUT_IMAGE_TYPE'] = convert_2_string(node_payload['image_type'])
    processed_data['VIDEO_OUTPUT_IMAGE_QUALITY'] = convert_2_number_sequence(node_payload['image_quality'])
    processed_data['VIDEO_OUTPUT_IMAGE_PORT'] = convert_2_number_sequence(node_payload['image_port'])
    processed_data['VIDEO_OUTPUT_IMAGE_PASSWORD'] = convert_2_string(node_payload['image_password'])


def process_scenario_door(node_payload, processed_data):
    processed_data['SCENARIO_DOOR'] = convert_2_bool(node_payload['enabled'])
    processed_data['SCENARIO_DOOR_CATEGORIES'] = convert_2_string(node_payload['categories'])
    processed_data['SCENARIO_DOOR_COORDS'] = convert_2_number_sequence(node_payload['coords'])
    processed_data['SCENARIO_DOOR_ENTERING_ENABLED'] = convert_2_bool(node_payload['entering_enabled'])
    processed_data['SCENARIO_DOOR_ENTERING_LABEL'] = convert_2_string(node_payload['entering_label'])
    processed_data['SCENARIO_DOOR_LEAVING_ENABLED'] = convert_2_bool(node_payload['leaving_enabled'])
    processed_data['SCENARIO_DOOR_LEAVING_LABEL'] = convert_2_string(node_payload['leaving_label'])


def process_scenario_parking(node_payload, processed_data):
    processed_data['SCENARIO_PARKING'] = convert_2_bool(node_payload['enabled'])
    processed_data['SCENARIO_PARKING_CATEGORIES'] = convert_2_string(node_payload['categories'])
    processed_data['SCENARIO_PARKING_COORDS'] = convert_2_number_sequence(node_payload['coords'])
    processed_data['SCENARIO_PARKING_COLD_DOWN_TIME_S'] = convert_2_number_sequence(node_payload['cold_down_time_s'])
    processed_data['SCENARIO_PARKING_TIME_LIMIT_S'] = convert_2_number_sequence(node_payload['time_limit_s'])
    processed_data['SCENARIO_PARKING_DANGER_LIMIT'] = convert_2_number_sequence(node_payload['danger_limit'])


def process_scenario_raised_hand(node_payload, processed_data):
    processed_data['SCENARIO_RAISED_HAND'] = convert_2_bool(node_payload['enabled'])


def convert_coordinates(coordinates):
    # Assuming coordinates is a dictionary with 'x' and 'y' in percentage
    screen_width = 1920  # Example screen width
    screen_height = 1080  # Example screen height
    x_pixels = int(coordinates['x'] * screen_width / 100)
    y_pixels = int(coordinates['y'] * screen_height / 100)
    return {'x': x_pixels, 'y': y_pixels}


def write_env_file(data, file_path):
    with open(file_path, 'w') as file:
        for key, value in data.items():
            file.write(f"{key}={value}\n")


def adapt_coords(processed_data: string, image_width: int, image_height: int) -> string:
    """
    Converte una stringa di punti espressi in percentuale in coordinate espresse in pixel.

    Args:
        processed_data (str): Stringa contenente punti in percentuale, suddivisi da | e ;
        image_width (int): Larghezza dell'immagine in pixel.
        image_height (int): Altezza dell'immagine in pixel.

    Returns:
        str: Stringa contenente i punti convertiti in pixel, mantenendo lo stesso stile di quella in ingresso.
    """
    # Suddividi i gruppi di punti separati da ;
    if len(processed_data) == 0:
        return ""
    groups = processed_data.split(';')
    result_groups = []

    for group in groups:
        points = group.split('|')
        pixel_coords = []
        for point in points:
            x_percent, y_percent = map(lambda p: float(p.strip('%')), point.split(','))
            x_pixel = int((x_percent * image_width / 100))
            y_pixel = int((y_percent * image_height / 100))
            pixel_coords.append(f"{x_pixel},{y_pixel}")
        result_groups.append('|'.join(pixel_coords))

    return '|'.join(result_groups)


def convert_config_json_2_env(json_input_file_name: string, env_output_file_name: string):
    if not os.path.isfile(json_input_file_name):
        print(f"File {json_input_file_name} not found.")
        return

    config_data = read_json_file(json_input_file_name)
    processed_data = {}
    image_width, image_height = 0, 0

    for node in config_data['nodes']:
        node_data, current_width, current_height = process_node(node)
        if current_width > image_width:
            image_width = current_width
        if current_height > image_height:
            image_height = current_height
        processed_data.update(node_data)

    if image_width == 0 or image_height == 0:
        image_width = 800
        image_height = 600

    if 'SCENARIO_PARKING_COORDS' in processed_data:
        processed_data['SCENARIO_PARKING_COORDS'] = adapt_coords(processed_data['SCENARIO_PARKING_COORDS'], image_width,
                                                                 image_height)
    if 'SCENARIO_DOOR_COORDS' in processed_data:
        processed_data['SCENARIO_DOOR_COORDS'] = adapt_coords(processed_data['SCENARIO_DOOR_COORDS'], image_width,
                                                              image_height)
    if 'SCENARIO_IN_ZONE_COORDS' in processed_data:
        processed_data['SCENARIO_IN_ZONE_COORDS'] = adapt_coords(processed_data['SCENARIO_IN_ZONE_COORDS'], image_width,
                                                                 image_height)

    write_env_file(processed_data, env_output_file_name)
    print(f"Configuration has been written to {env_output_file_name}")


def convert_2_bool(s: string) -> string:
    return str(s).lower()


def convert_2_string(value: any) -> str:
    return re.sub(r"\s+", "", str(value))


def convert_2_number_sequence(value: any) -> str:
    return re.sub(r"x", ",", convert_2_string(value))
