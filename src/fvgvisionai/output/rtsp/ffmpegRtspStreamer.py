import subprocess
import threading
import time
import logging
from typing import Optional

import numpy as np

from fvgvisionai.common.triple_buffer import TripleBuffer


class FFMpegRtspStreamer:
    """
    Legge i frame dal TripleBuffer e li trasmette via RTSP usando FFmpeg.
    Compatibile con MediaMTX (RTSP push) o FFmpeg listen mode (test).
    """

    def __init__(
        self,
        buffer: TripleBuffer,
        image_size: str,
        fps: int,
        rtsp_url: str,
        use_nvenc: bool = False,
    ):
        self.buffer = buffer
        self.image_size = image_size
        self.fps = fps
        self.rtsp_url = rtsp_url
        self.use_nvenc = use_nvenc

        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._process: Optional[subprocess.Popen] = None

        self.logger = logging.getLogger(self.__class__.__name__)

    # ---------- public API ----------

    def start(self):
        if self._thread and self._thread.is_alive():
            return

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="FFMpegRtspStreamer",
            daemon=True,
        )
        self._thread.start()
        self.logger.info("RTSP streamer started")

    def stop(self):
        self._stop_event.set()
        self._cleanup_ffmpeg()
        if self._thread:
            self._thread.join(timeout=5)
        self.logger.info("RTSP streamer stopped")

    # ---------- internals ----------

    def _run(self):
        backoff = 1.0
        last_frame_id = None
        self.logger.info(f'FFMPEG start writing')
        while not self._stop_event.is_set():
            try:
                self._start_ffmpeg()
                backoff = 1.0
                frame_interval = 1.0 / self.fps
                while not self._stop_event.is_set():
                    frame_id, frame = self.buffer.get_ready_frame()

                    if frame is None or frame_id == last_frame_id:
                        time.sleep(0.002)
                        continue

                    last_frame_id = frame_id
                    self._write_frame(frame)
                    time.sleep(frame_interval)

            except Exception as e:
                self.logger.error(f"RTSP streaming error: {e}")
                self._cleanup_ffmpeg()
                self.logger.warning(f"Restarting FFmpeg in {backoff:.1f}s")
                time.sleep(backoff)
                backoff = min(backoff * 2, 10.0)

    def _start_ffmpeg(self):
        cmd = self._build_ffmpeg_cmd()
        self.logger.info(f"Starting FFmpeg: {' '.join(cmd)}")

        self._process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize = 0
        )

    def _write_frame(self, frame: np.ndarray):

        if not self._process or not self._process.stdin:
            raise RuntimeError("FFmpeg process not running")

        try:
            self._process.stdin.write(frame.tobytes())
        except BrokenPipeError:
            rc = self._process.poll()
            self.logger.error("FFmpeg stdin broken, returncode=%s", rc)

            if self._process.stderr:
                err = self._process.stderr.read().decode(errors="ignore")
                if err.strip():
                    self.logger.error("FFmpeg stderr:\n%s", err)

            raise RuntimeError("FFmpeg stdin broken")

    def _cleanup_ffmpeg(self):
        if self._process:
            try:
                self._process.kill()
            except Exception:
                pass
            self._process = None

    # ---------- ffmpeg command ----------

    def _build_ffmpeg_cmd(self):
        """
        Costruisce il comando FFmpeg per pubblicare frame raw BGR24
        verso RTSP (MediaMTX in produzione oppure listen mode in test).
        """

        if self.use_nvenc:
            video_encoder_args = [
                "-c:v", "h264_nvenc",
                "-pix_fmt", "yuv420p",
                "-preset", "llhq",
                "-rc", "cbr",
                "-b:v", "2000k",
                "-maxrate", "2000k",
                "-bufsize", "4000k",
                "-bf", "0",
                "-g", str(self.fps),
            ]
        else:
            video_encoder_args = [
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-profile:v", "baseline",
                "-preset", "ultrafast",
                "-tune", "zerolatency",
                "-bf", "0",
                "-g", str(self.fps),                
                "-keyint_min", str(self.fps),
                "-sc_threshold", "0",
                "-forced-idr", "1",
            ]

        cmd = [
            "ffmpeg",
            "-loglevel", "warning",
            "-fflags", "nobuffer",
            "-flags", "low_delay",
            "-use_wallclock_as_timestamps", "1",

            # Input raw video da stdin
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", self.image_size,
            "-r", str(self.fps),
            "-i", "-",
            "-rtsp_transport", "tcp",
        ]

        cmd += video_encoder_args

        # Output RTSP
        cmd += [
            "-f", "rtsp",
            "-muxdelay", "0",
            self.rtsp_url,   # es: rtsp://mediamtx:8554/fvgvision-ai
        ]

        return cmd
