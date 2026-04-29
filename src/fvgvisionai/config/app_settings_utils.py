import math
import os
from configparser import ConfigParser
from enum import Enum
from typing import Dict, Optional, List

import cv2
import numpy as np

from fvgvisionai.processor.categories import ModelCategory, all_category_list


class VideoSourceMode(Enum):
    CUDA = {"name": "cuda"}
    STREAM = {"name": "stream"}
    FFMPEG = {"name": "ffmpeg"}
    IMAGE = {"name": "image"}


class ModelLibrary(Enum):
    ULTRALYTICS = {"name": "ultralytics"}
    PASSTHROUGH = {"name": "passthrough"}


class ModelId(Enum):
    V8 = {"name": "yolo8", "prefix": "yolov8"}
    V11 = {"name": "yolo11", "prefix": "yolo11"}
    V26 = {"name": "yolo26", "prefix": "yolo26"}

class ModelSize(Enum):
    NANO = {"name": "nano", "suffix": "n"}
    SMALL = {"name": "small", "suffix": "s"}
    MEDIUM = {"name": "medium", "suffix": "m"}
    LARGE = {"name": "large", "suffix": "l"}
    XLARGE = {"name": "xlarge", "suffix": "x"}


class ModelResolution(Enum):
    RESOLUTION_1024x1024 = {"name": "1024x1024", "suffix": "1024x1024", "resolution": (1024, 1024)}
    RESOLUTION_768x768 = {"name": "768x768", "suffix": "768x768", "resolution": (768, 768)}
    RESOLUTION_640x640 = {"name": "640x640", "suffix": "640x640", "resolution": (640, 640)}
    RESOLUTION_512x512 = {"name": "512x512", "suffix": "512x512", "resolution": (512, 512)}
    RESOLUTION_480x480 = {"name": "480x480", "suffix": "480x480", "resolution": (480, 480)}


class ModelPrecision(Enum):
    FLOAT_32 = {"name": "float32", "suffix": "f32"}
    FLOAT_16 = {"name": "float16", "suffix": "f16"}
    INT_8 = {"name": "int8", "suffix": "i8"}


class ImageType(Enum):
    JPEG = {"name": "jpeg", "extension": ".jpg",
            "mime-type": "image_source/jpeg",
            "quality_param": int(cv2.IMWRITE_JPEG_QUALITY)}
    WEBP = {"name": "webp", "extension": ".webp",
            "mime-type": "image_source/webp",
            "quality_param": int(cv2.IMWRITE_WEBP_QUALITY)}


class InvalidConfigurationException(Exception):
    def __init__(self, message="Invalid configuration"):
        self.message = message
        super().__init__(self.message)


def extract_dimensions(value: str) -> (int, int):
    temp_size = value.split(",")
    return int(temp_size[0]), int(temp_size[1])


def to_model_filename(value: str) -> str:
    file_name = value.strip()

    if file_name.lower() == "none" or file_name == "":
        file_name = None

    return file_name


def from_props(properties: Dict[str, any], config: Optional[ConfigParser], key: str, section: str) -> str:
    section = section.lower()
    if key.upper() in properties:
        return properties.get(key.upper()).strip()
    elif key.lower() in properties:
        return properties.get(key.lower()).strip()
    elif config.has_option(section, key.upper()):
        return config.get(section, key.upper()).strip()
    elif config.has_option(section, key.lower()):
        return config.get(section, key.lower()).strip()
    else:
        raise InvalidConfigurationException(f"Param {key} is not defined!")


def from_env(config: Optional[ConfigParser], key: str, section, cli_args: Dict[str, str]) -> str:
    """
    Recupera dall'environment i parametri. L'ordine di priorita': args, env, default_config.ini

    :param config:
    :param key:
    :param section:
    :param cli_args:
    :return:
    """
    section = section.lower()
    if key.upper() in cli_args:
        return cli_args.get(key.upper()).strip()
    elif key.lower() in cli_args:
        return cli_args.get(key.lower()).strip()
    elif key.upper() in os.environ:
        return os.environ.get(key.upper()).strip()
    elif key.lower() in os.environ:
        return os.environ.get(key.lower()).strip()
    elif config.has_option(section, key.upper()):
        return config.get(section, key.upper()).strip()
    elif config.has_option(section, key.lower()):
        return config.get(section, key.lower()).strip()
    else:
        raise InvalidConfigurationException(f"Param {key} is not defined!")


def to_bool(value: str) -> bool:
    return value.upper() == "TRUE"


def to_str(value: str) -> str:
    return value.replace(" ", "").replace("_", " ")


def to_binary_array(spaced_string: str) -> List[bool]:
    """
    Converte una stringa contenente spazi in una lista di booleani,
    rappresentando la sua forma binaria senza spazi.

    Parameters:
    - spaced_string (str): La stringa di input contenente spazi.

    Returns:
    - List[bool]: Una lista di booleani rappresentante la forma binaria della stringa senza spazi, invertita.
    """

    # Rimuovi gli spazi dalla stringa
    binary_string = ''.join(spaced_string.split())

    # Riempi la lista di booleani
    binary_array = [bit == '1' for bit in binary_string]

    # Inverti l'ordine della lista e restituisci il risultato
    return binary_array[::-1]


def to_zone_poly(zone_coords: str) -> np.array:
    polygon = [[int(y) for y in x.split(',')] for x in zone_coords.split('|')]

    # zone to count people in
    zone_poly = np.array([polygon[0],  # left upper corner
                          polygon[1],  # right upper corner
                          polygon[2],  # right lower corner
                          polygon[3]], np.int32)  # left lower corner
    zone_poly = zone_poly.reshape((-1, 1, 2))

    return zone_poly


def to_parking_list(parking_coords: str) -> List[np.array]:
    # Rimuovi gli spazi bianchi dalla variabile
    parking_coords = parking_coords.replace(" ", "")
    # Rimuovi gli spazi bianchi attorno alle parentesi quadre e dividi le definizioni dei box
    box_definitions_str = parking_coords.strip().split('][')

    # Inizializza un array per memorizzare le coordinate dei box
    box_definitions = []

    # Itera attraverso le definizioni dei box e decodifica le coordinate
    for box_str in box_definitions_str:
        # Rimuovi le parentesi quadre e dividi le coordinate del box
        coordinates = box_str.strip('[]').split('|')
        # Converti le coordinate in numeri interi
        polygon = [[int(coord) for coord in coord_pair.split(',')] for coord_pair in coordinates]
        # Aggiungi le coordinate decodificate all'array di definizioni
        # zone to count people in
        parking_poly = np.array([polygon[0],  # left upper corner
                                 polygon[1],  # right upper corner
                                 polygon[2],  # right lower corner
                                 polygon[3]], np.int32)  # left lower corner
        parking_poly = parking_poly.reshape((-1, 1, 2))

        box_definitions.append(parking_poly)

    # Stampare l'array di definizioni
    return box_definitions


def to_model_categories(input_string: str) -> List[ModelCategory]:
    if input_string.upper().strip() == "ALL":
        return all_category_list

    category_names = input_string.strip().split('|')
    result_categories: List[ModelCategory] = []

    for name in category_names:
        found = False
        for category in all_category_list:
            if name.lower().strip() == category.label.lower():
                result_categories.append(category)
                found = True
                break
        if not found:
            raise ValueError(f"Non e' definita alcuna categoria '{name}'.")

    return result_categories


def to_door_poly(door_coords: str) -> (np.array, np.array, np.array):
    door = [[int(y) for y in x.split(',')] for x in door_coords.split('|')]

    a_x, a_y = door[0]
    b_x, b_y = door[1]
    length = cv2.norm(np.array(door[0]), np.array(door[1])) / 2

    # get mvt vector
    v_x = b_x - a_x
    v_y = b_y - a_y
    # normalize vector
    mag = math.sqrt(v_x * v_x + v_y * v_y)
    v_x = v_x / mag
    v_y = v_y / mag
    # swapping x and y and inverting one of them
    temp = v_x
    swapped_v_x = 0 - v_y
    swapped_v_y = temp
    c_x = int(b_x + swapped_v_x * length)
    c_y = int(b_y + swapped_v_y * length)
    d_x = int(b_x - swapped_v_x * length)
    d_y = int(b_y - swapped_v_y * length)
    e_x = int(a_x + swapped_v_x * length)
    e_y = int(a_y + swapped_v_y * length)
    f_x = int(a_x - swapped_v_x * length)
    f_y = int(a_y - swapped_v_y * length)

    door_line = np.array([(a_x, a_y),
                          (b_x, b_y)])

    door_poly = np.array([(a_x, a_y),
                          (b_x, b_y),
                          (c_x, c_y),
                          (e_x, e_y)], np.int32).reshape((-1, 1, 2))

    door2_poly = np.array([(f_x, f_y),
                           (d_x, d_y),
                           (b_x, b_y),
                           (a_x, a_y)], np.int32).reshape((-1, 1, 2))

    return door_line, door_poly, door2_poly


def to_video_source_mode(input_string: str) -> VideoSourceMode:
    for enum_value in VideoSourceMode:
        if enum_value.value["name"] == input_string.strip().lower():
            return enum_value
    raise ValueError(f"String '{input_string}' non corrisponde a nessun valore enumerativo.")


def to_model_size(input_string: str) -> ModelSize:
    for enum_value in ModelSize:
        if enum_value.value["name"] == input_string.strip().lower():
            return enum_value
    raise ValueError(f"String '{input_string}' non corrisponde a nessun valore enumerativo.")


def to_model_resolution(input_string: str) -> ModelResolution:
    for enum_value in ModelResolution:
        if enum_value.value["name"] == input_string.strip().lower():
            return enum_value
    raise ValueError(f"String '{input_string}' non corrisponde a nessun valore enumerativo.")


def to_model_precision(input_string: str) -> ModelPrecision:
    for enum_value in ModelPrecision:
        if enum_value.value["name"] == input_string.strip().lower():
            return enum_value
    raise ValueError(f"String '{input_string}' non corrisponde a nessun valore enumerativo.")

def to_model_id(input_string: str) -> ModelId:
    for enum_value in ModelId:
        if enum_value.value["name"] == input_string.strip().lower():
            return enum_value
    raise ValueError(f"String '{input_string}' non corrisponde a nessun valore enumerativo.")

def to_model_library(input_string: str) -> ModelLibrary:
    for enum_value in ModelLibrary:
        if enum_value.value["name"] == input_string.strip().lower():
            return enum_value
    raise ValueError(f"String '{input_string}' non corrisponde a nessun valore enumerativo.")


def to_image_type(input_string: str) -> ImageType:
    for enum_value in ImageType:
        if enum_value.value["name"] == input_string.strip().lower():
            return enum_value
    raise ValueError(f"String '{input_string}' non corrisponde a nessun valore enumerativo.")


def get_array_as_string(array_property) -> str:
    return (', '.join(map(str, array_property))
            .replace("[[[", "[(")
            .replace("]]]", ")]")
            .replace("\n", "")
            .replace("[[", "(")
            .replace("]]", ")")
            )
