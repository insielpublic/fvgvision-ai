import argparse
import atexit
import configparser
import signal
import sys
import threading
import time
from configparser import ConfigParser
from types import FrameType
from typing import Dict

from fvgvisionai.benchmark.benchmark_monitor import BenchmarkMonitor
from fvgvisionai.common.config_executor import ConfigExecutor
from fvgvisionai.common.pid_file import remove_pid_file, is_another_instance_running, write_pid_file
from fvgvisionai.common.triple_buffer import TripleBuffer
from fvgvisionai.common.video_observable import VideoObservable
from fvgvisionai.config.app_settings import load_settings_from_file, load_settings_from_env
from fvgvisionai.input.frame_reader import run_reader
from fvgvisionai.notify.notification_client import NotificationClient
from fvgvisionai.output.hls.hls_streamer import run_hls_streamer_thread
from fvgvisionai.output.rtsp.ffmpegRtspStreamer import FFMpegRtspStreamer
from fvgvisionai.webserver.web_server import run_web_server

DEFAULT_CONFIG_INI = 'default_config.ini'

exit_signal: threading.Event = threading.Event()


def handle_exit(_: int, __: FrameType):
    print(f"Received interrupt signal (CTRL+C). Shutdown is running and pid file will be removed")
    exit_signal.set()
    remove_pid_file()


def main() -> int:
    global exit_signal

    # Crea un oggetto ConfigParser
    config: ConfigParser = configparser.ConfigParser()

    # Leggi il file di configurazione
    config.read(DEFAULT_CONFIG_INI)

    # Leggi il valore associato alla chiave "chiave" nella sezione "Sezione"
    app_version = config.get('application', 'version')
    app_name = config.get('application', 'name')

    parser = argparse.ArgumentParser(description="Pongo configuration")
    parser.add_argument('--file', help="Use a property file for configuration")
    parser.add_argument('--env', action='store_true', help="Use environment variables for configuration")
    parser.add_argument('--benchmark', action='store_true', help="Launch app in benchmark mode")
    parser.add_argument('--version', action='store_true', help="Show program version")
    parser.add_argument('--arg', action='append', nargs='*', metavar=('key', 'value'),
                        help='Specify a parameter via CLI')

    args = parser.parse_args()

    if args.version:
        print(f"{app_name} - v.{app_version}")
        return 0

    config_executor = ConfigExecutor()
    config_executor.apply_logging_config(config)

    # Se sono specificati parametri, li aggiunge al file di configurazione
    cli_args = build_args(args)

    if args.benchmark:
        app_settings = load_settings_from_file(".env-benchmark", cli_args, config, benchmark_mode=True)
    elif args.file:
        app_settings = load_settings_from_file(args.file, cli_args, config)
    elif args.env:
        app_settings = load_settings_from_env(cli_args, config)
    else:
        print("Argument --file or --env must be used to configure program.")
        return -1

    if is_another_instance_running(app_name):
        exit_signal.set()
        return -1

    # Scrive il PID al lancio dell'applicazione
    write_pid_file()

    # Rimuove il file PID alla terminazione dell'applicazione
    atexit.register(remove_pid_file)

    app_settings.show_properties()

    # Collega il gestore dei segnali al segnale SIGINT (CTRL+C)
    signal.signal(signal.SIGINT, handle_exit)
    signal.signal(signal.SIGTERM, handle_exit)

    video_observable = VideoObservable()
    image_buffer = TripleBuffer()

    hls_streamer_thread = None
    if app_settings.video_output_stream:
        hls_streamer_thread = run_hls_streamer_thread(video_observable, image_buffer, app_settings, exit_signal)

    if app_settings.video_output_rtsp:
        rtsp_streamer_thread = FFMpegRtspStreamer(
            buffer=image_buffer,
            image_size=app_settings.video_output_rtsp_risoluzione,
            fps=app_settings.video_output_fps,
            rtsp_url=app_settings.video_output_rtsp_url,
            use_nvenc=app_settings.video_output_rtsp_gpu_nvidia,   # se hai GPU NVIDIA usa nvenc
        )
        rtsp_streamer_thread.start()


    if app_settings.video_output_image:
        web_server_thread = threading.Thread(target=run_web_server,
                                             args=(image_buffer, app_settings, exit_signal,),
                                             name='WebServerThread')
        web_server_thread.daemon = True
        web_server_thread.start()

    if app_settings.enabled_notification:
        notification_client = NotificationClient(enabled=True,
                                                 azure_connection_string=app_settings.azure_connection_string,
                                                 device_id=app_settings.notification_device_id,
                                                 camera_id=app_settings.notification_camera_id,
                                                 model_id=app_settings.model_id.value,
                                                 measures_aggregation_time_ms=app_settings.notification_aggregation_time_ms)
        notification_client.start()
    else:
        notification_client = None

    benchmark_monitor = None
    if app_settings.benchmark_enabled:
        benchmark_monitor = BenchmarkMonitor(enabled=True, app_settings=app_settings,
                                             measures_aggregation_time_ms=app_settings.benchmark_aggregation_time_ms,
                                             exit_signal=exit_signal)
        benchmark_monitor.start()

    computer_vision_thread = threading.Thread(target=run_reader,
                                              args=(video_observable, image_buffer,
                                                    notification_client, benchmark_monitor,
                                                    app_settings, exit_signal,),
                                              name='InputThread')
    computer_vision_thread.daemon = True
    computer_vision_thread.start()
    computer_vision_thread.join()

    if app_settings.benchmark_enabled:
        benchmark_monitor.close()
        exit_signal.set()

    if hls_streamer_thread is not None:
        hls_streamer_thread.join()

    time.sleep(2)

    return 0


def build_args(args):
    cli_args: Dict[str, str] = {}
    if args.arg:
        for arg in args.arg:
            key, value = arg[0].split('=')
            cli_args[key.upper()] = value
    return cli_args


if __name__ == "__main__":
    sys.exit(main())
