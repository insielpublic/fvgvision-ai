import base64
import cv2
from flask import Response, abort, jsonify

DEFAULT_IMAGE_PATH = "/app/assets/images/no_image_available800x600.jpg"
CONFIG_WEB_JSON_FILENAME = '/app/config-web.json'
ENV_WEB_FILENAME = '/app/.env-main-web'
RUN_MAIN_WEB = "/app/run_main_web.sh"


def generate_image_preview(url: str, width: int, height: int) -> tuple[Response, int]:
    """
    Generates a preview image from a video stream URL.

    Args:
        url (str): The URL of the video stream.
        width (int): The desired width of the preview image.
        height (int): The desired height of the preview image.

    Returns:
        Response: A JSON response containing the base64 encoded image and its dimensions.
        int: The HTTP status code.
    """
    original_height = height
    original_width = width
    try:
        # Open the video stream with OpenCV
        cap = cv2.VideoCapture(url)
        if not cap.isOpened():
            raise Exception("Unable to open video stream")

        # Capture the first frame
        ret, frame = cap.read()
        if not ret:
            raise Exception("Unable to capture frame from video stream")

        # Get the original dimensions of the image
        original_height, original_width = frame.shape[:2]

        # Resize the image if width and height are provided
        if width and height:
            frame = cv2.resize(frame, (width, height))
        else:
            width, height = original_width, original_height

        # Convert the frame to base64
        _, buffer = cv2.imencode('.jpg', frame)
        img_bytes = buffer.tobytes()
        img_base64 = base64.b64encode(img_bytes).decode('utf-8')

        # Close the video stream
        cap.release()
    except Exception:
        # In case of error, load the fallback image from disk
        img_base64, width, height = _get_default_image_base64()

    # Return the base64 image along with the dimensions
    return jsonify({"image": img_base64,
                    "width": width, "height": height,
                    "original_width": original_width, "original_height": original_height}), 200


def _get_default_image_base64() -> tuple[str, int, int]:
    """
    Loads the default image from disk and returns it in base64 format along with its dimensions.

    Returns:
        str: The base64 encoded image.
        int: The width of the image.
        int: The height of the image.

    Raises:
        FileNotFoundError: If the default image is not found.
    """
    try:
        # Read the image with OpenCV
        image = cv2.imread(DEFAULT_IMAGE_PATH)
        if image is None:
            raise FileNotFoundError("Default image not found")

        # Get the dimensions of the image
        height, width = image.shape[:2]

        # Convert the image to base64
        _, buffer = cv2.imencode('.jpg', image)
        img_bytes = buffer.tobytes()
        img_base64 = base64.b64encode(img_bytes).decode('utf-8')

        return img_base64, width, height
    except FileNotFoundError:
        abort(500, description="Default image not found")
