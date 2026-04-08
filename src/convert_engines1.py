import os
import concurrent.futures
from ultralytics import YOLO

base_dir = "./assets/models/"

models_info = [
    #    'yolov8x',
    #    'yolov8x-pose',
    #"'yolov8n',
    'yolov8n-pose',
    #'yolo11n',
    #'yolo11n-pose',
    #    'yolov8s',
    #    'yolov8s-pose',
    #    'yolov8m',
    #    'yolov8m-pose',
    #    'yolov8l',
    #    'yolov8l-pose',
]

size_info = [
    #    ('-1024x1024', 1024, 1024),
    #    ('-768x768', 768, 768),
    ('-640x640', 640, 640),
    #    ('-512x512', 512, 512),
    #    ('-480x480', 480, 480)
]


def process_model(model_file_name, mode_size_str, model_size_h, model_size_w, half=False, int8=False):
    output_file_path = f"{base_dir}{model_file_name}{mode_size_str}-{'f16' if half else 'i8' if int8 else 'f32'}.engine"

    # Verifica se il file di destinazione esiste
    if not os.path.exists(output_file_path):
        print(f"CREO {output_file_path}")
        model = YOLO(base_dir + model_file_name + ".pt")
        model.export(format='engine', half=half, int8=int8, imgsz=[model_size_h, model_size_w])
        os.rename(f"{base_dir}{model_file_name}.engine", output_file_path)
    else:
        print(f"Il file {output_file_path} esiste già")


def process_size(section_size_info_entry, section_models_info):
    mode_size_str, model_size_h, model_size_w = section_size_info_entry
    # with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
    for model_file_name in section_models_info:
        process_model(model_file_name, mode_size_str, model_size_h, model_size_w)
        process_model(model_file_name, mode_size_str, model_size_h, model_size_w, half=True)
        process_model(model_file_name, mode_size_str, model_size_h, model_size_w, int8=True)


# Esegui il processo per ogni dimensione
for size_info_entry in size_info:
    process_size(size_info_entry, models_info)
