<img src="docs/fvg%20vision%20ai-trasparent.png" width="384px"/>

# FVG Vision AI - Edge Stream and Analysis

FVG Vision AI is an advanced real-time video analysis platform based on AI and edge computing technologies. Utilizing [**Ultralytics**](https://github.com/ultralytics/ultralytics)
and other open-source libraries, it detects and tracks objects such as vehicles, people, and more,
providing real-time insights for applications like surveillance and smart city management. With a focus on low-latency
processing and edge deployment, FVG Vision IA enables event-driven notifications and seamless integration with IoT hubs
and intelligent systems.

<img src="https://github.com/user-attachments/assets/71cbe0cb-072c-4102-abb7-0348f6d34a2a" width="600px"/>

## Key Features:

- Real-time object detection and tracking
- Edge computing optimization for low-latency processing
- Seamless integration with IoT Hub platform
- Built on Ultralytics and other open-source libraries

## Quick start

To quickly run the application inside a Docker container, follow the steps below for your platform.

### On Jetson AGX/Orin platform, WITH cuda acceleration

- Open a bash shell in the `docker/jetson-agx-jp6` directory.
- Run `setup.sh` to configure the appropriate user permissions and CUDA paths.
- Execute `./docker_start_main.sh` to start the containerized application.

### On PC platform, WITH cuda acceleration

- Open a bash shell in the `docker/amd64-cuda` directory.
- Run `setup.sh` to configure the appropriate user permissions and CUDA paths.
- Execute `./docker_start_main.sh` to launch the container.

### On PC platform without cuda acceleration

To run the application without CUDA support, use the following Docker command:

```bash
docker run -d                                                         \
        --shm-size=5gb                                                \
        -p 5000:5000 -p 80:8080                                       \
  	    --mount type=tmpfs,destination=/mnt/hls                       \
	    --name fvgvision-ai xcesco/fvgvision-ai:0.2.0-amd64-main
```

Alternatively, you can open a bash shell in the `docker/amd64` directory and execute `./docker_start_main.sh`.

### Verifying the Application

Once started, the application will run with a default configuration. To verify the setup:

- Open a browser and navigate to http://localhost.
- If the setup is successful, you should see a live video.






