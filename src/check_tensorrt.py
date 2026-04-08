# Utilizza la funzione
from fvgvisionai.common.utils import is_tensorrt_installed, get_tensorrt_version


if is_tensorrt_installed():
    print(f"TensorRT {get_tensorrt_version()} è installato!")
else:
    print("TensorRT non è installato.")
