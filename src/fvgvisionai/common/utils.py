import time

import pkg_resources


def is_tensorrt_installed() -> bool:
    try:
        pkg_resources.get_distribution('tensorrt')
        return True
    except pkg_resources.DistributionNotFound:
        return False

def get_tensorrt_version() -> str:
    if is_tensorrt_installed():
        return pkg_resources.get_distribution('tensorrt').version
    else:
        return ""

def wait_frame_duration(minimum_elapsed_time: int, time_to_acquire_frame: int):
    if minimum_elapsed_time > time_to_acquire_frame:
        time.sleep((minimum_elapsed_time - time_to_acquire_frame) / 1000.0)


def compute_elapsed_time_ms(video_fps) -> int:
    elapsed_time = 1000.0 / video_fps
    return round(elapsed_time)
