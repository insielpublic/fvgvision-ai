FROM nvcr.io/nvidia/l4t-jetpack:r36.3.0 AS common

# Aggiunta della direttiva LABEL per fornire metadati
LABEL maintainer="Francesco Benincasa <xcesco@gmail.com>" \
      version="0.1.0" \
      description="FVG Vision AI on docker"


# Esposizione delle porte 5000 e 8080 (video in oupput), 8081 web controller
EXPOSE 5000/tcp
EXPOSE 8080/tcp

EXPOSE 8081/tcp

# Base layer
FROM common AS development-base

# Installazione dei package **essenziali**
RUN apt-get update && apt-get install -y \
    git nano sudo python3-pip \
    wget python3-venv

ARG USER_ID=1000
ARG GROUP_ID=1000

# Mount HLS
RUN mkdir /mnt/hls
RUN chmod a+rw /mnt/hls

# Imposta le variabili d'ambiente nel Dockerfile
ENV USER_ID=$USER_ID
ENV GROUP_ID=$GROUP_ID

RUN groupadd -g $GROUP_ID developer

# Creazione di un utente develper con permessi sudo e video
RUN useradd -rm -d /home/developer -s /bin/bash -g ${GROUP_ID} -G sudo,video -u ${USER_ID} developer
RUN echo 'developer:developer' | chpasswd

# Install ffmpeg
RUN apt-get update && apt-get install -y ffmpeg

WORKDIR /home/developer

# Timezone
ENV TZ=Europe/Rome
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN mkdir /home/developer/git

WORKDIR /home/developer/git
RUN git clone https://github.com/jetsonhacks/jetsonUtilities

RUN pip3 install --upgrade pip
RUN pip3 install jetson-stats

# Nginx
RUN apt-get update && apt install -y nginx
COPY ./build/assets/nginx/editor /usr/share/nginx/html/editor
COPY ./build/assets/nginx/players /usr/share/nginx/html/output-stream
COPY ./build/assets/nginx/config/nginx.conf /etc/nginx/nginx.conf

# Start Nginx
RUN service nginx start

##############################################################
#
#            P y t h o n    D e p e n d e n c i e s
#
##############################################################

# Numpy
RUN pip3 install numpy==1.26.4
RUN pip3 install blinker==1.8.2


# Torch, Torchvision, Torchaudio
# https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048
RUN wget https://nvidia.box.com/shared/static/mp164asf3sceb570wvjsrezk1p4ftj8t.whl -O torch-2.3.0-cp310-cp310-linux_aarch64.whl
RUN apt-get update && apt-get install -y libopenblas-base libopenmpi-dev libomp-dev
RUN pip3 install 'Cython<3'
RUN pip3 install torch-2.3.0-cp310-cp310-linux_aarch64.whl

RUN wget https://nvidia.box.com/shared/static/9agsjfee0my4sxckdpuk9x9gt8agvjje.whl -O  torchaudio-2.3.0+952ea74-cp310-cp310-linux_aarch64.whl
RUN pip3 install torchaudio-2.3.0+952ea74-cp310-cp310-linux_aarch64.whl

RUN wget https://nvidia.box.com/shared/static/xpr06qe6ql3l6rj22cu3c45tz1wzi36p.whl -O torchvision-0.18.0a0+6043bc2-cp310-cp310-linux_aarch64.whl
RUN pip3 install torchvision-0.18.0a0+6043bc2-cp310-cp310-linux_aarch64.whl

# OpenCV
RUN apt-get update && apt-get install -y             \
            software-properties-common apt-utils     \
            gcc curl wget ffmpeg libsm6 libxext6     \
            software-properties-common apt-utils gcc \
            libsm6 libxext6 python3-dev

WORKDIR /home/developer/git
RUN git clone --branch 4.8.0 https://github.com/opencv/opencv
RUN git clone --branch 4.8.0 https://github.com/opencv/opencv_contrib
RUN mkdir opencv_build
WORKDIR /home/developer/git/opencv_build

RUN apt-get update && apt-get install -y \
    build-essential cmake                \
    libavcodec-dev                       \
    libavformat-dev                      \
    libavutil-dev                        \
    libeigen3-dev                        \
    libglew-dev                          \
    libgstreamer-plugins-base1.0-dev     \
    libgstreamer1.0-dev                  \
    libgtk2.0-dev                        \
    libgtk-3-dev                         \
    libjpeg-dev                          \
    libpng-dev                           \
    libpostproc-dev                      \
    libswscale-dev                       \
    libtbb-dev                           \
    libtiff5-dev                         \
    libv4l-dev                           \
    libxvidcore-dev                      \
    libx264-dev                          \
    qtbase5-dev qt5-qmake                \
    zlib1g-dev                           \
    pkg-config

RUN cmake /home/developer/git/opencv 	  \
   -D CMAKE_BUILD_TYPE=Release		  \
   -D CMAKE_INSTALL_PREFIX=/usr/local	  \
   -D BUILD_EXAMPLES=OFF        	  \
   -D BUILD_opencv_python2=OFF  	  \
   -D BUILD_opencv_python3=ON   	  \
   -D PYTHON3_EXECUTABLE=/usr/bin/python3 \
   -D PYTHON3_PACKAGES_PATH=/usr/local/lib/python3.10/dist-packages                             \
   -D PYTHON3_NUMPY_INCLUDE_DIRS=/usr/local/lib/python3.10/dist-packages/numpy/core/include 	\
   -D OPENCV_PYTHON3_INSTALL_PATH=/usr/local/lib/python3.10/dist-packages 			\
   -D OPENCV_EXTRA_MODULES_PATH=/home/developer/git/opencv_contrib/modules 			\
   -D CUDA_ARCH_BIN=8.7                                 \
   -D CUDA_FAST_MATH=ON                                 \
   -D CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda 		\
   -D CMAKE_LIBRARY_PATH=/usr/local/cuda/lib64/stubs	\
   -D OPENCV_DNN_CUDA=OFF       \
   -D OPENCV_ENABLE_NONFREE=ON	\
   -D WITH_CUDA=ON              \
   -D WITH_CUDNN=OFF            \
   -D WITH_FFMPEG=ON            \
   -D WITH_GSTREAMER=ON         \
   -D WITH_OPENGL=ON            \
   -D BUILD_PERF_TESTS=OFF      \
   -D BUILD_TESTS=OFF
RUN make -j$(nproc)
RUN make install /usr/local

WORKDIR /home/developer/git/opencv_build/python_loader
RUN python3 setup.py bdist_wheel

# Installa il pacchetto wheel
RUN pip install dist/*.whl
WORKDIR /home/developer/

# Installa le altre dipendenze
RUN pip3 install                        \
        matplotlib                      \
        pillow                          \
        pyyaml                          \
        requests                        \
        scipy                           \
        tqdm

RUN pip3 install                        \
        psutil                          \
        py-cpuinfo                      \
        thop                            \
        pandas                          \
        seaborn configparser pytz       \
        azure.iot.device openpyxl       \
        onnx onnxsim    \
        flask flask_wtf Flask-CORS flask-swagger-ui waitress

RUN pip install --no-deps ultralytics

RUN apt-get update && apt-get install -y \
    gstreamer1.0-python3-plugin-loader   \
    python3-gst-1.0  

# Other dependencies
RUN pip3 install psutil==6.0.0 openpyxl==3.1.5

##############################################################

WORKDIR /home/developer/git
#RUN rm -r opencv*
RUN rm -r *.whl
RUN chown -R developer:developer /home/developer/git
RUN chmod -R a+rw /home/developer/git

# Fase di sviluppo
FROM development-base AS development

# Esposizione della porta SSH per il debug
EXPOSE 22/tcp
# Esposizione di jupiter notebook
EXPOSE 8888/tcp

# Installazione del server SSH
RUN apt-get update && apt-get install -y openssh-server
RUN mkdir /run/sshd
RUN chown root:root /run/sshd
RUN chmod 755 /run/sshd

# Copia le installazioni di OpenCV dalla fase di build
#COPY --from=builder /usr/local /usr/local
#COPY --from=builder /usr/local/lib/python3.10/dist-packages /usr/local/lib/python3.10/dist-packages

# Copia dello script di avvio personalizzato per lo sviluppos
COPY ./build/assets/entrypoints/start_dev.sh /docker-entrypoint.sh
# Impostazione dei permessi di esecuzione per lo script di avvio
RUN chmod +x /docker-entrypoint.sh

WORKDIR /home/developer/

# Comando di avvio per la fase di sviluppo
CMD ["/docker-entrypoint.sh"]

# Fase di produzione
FROM development-base AS production

# Copia dello script di avvio personalizzato per lo sviluppo
COPY ./build/assets/entrypoints/start_prod.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh

# Copy the app directory into the container
COPY build/app /app

WORKDIR /home/developer/

# Comando di avvio per la fase di produzione
CMD ["/docker-entrypoint.sh"]
