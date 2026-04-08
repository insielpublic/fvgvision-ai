# https://hub.docker.com/_/python/tags?name=3.10
FROM python:3.10.15-slim-bullseye AS common

# Aggiunta della direttiva LABEL per fornire metadati
LABEL maintainer="Francesco Benincasa <xcesco@gmail.com>" \
      version="0.3.0" \
      description="FVG Vision AI on docker"

# Esposizione delle porte 5000 e 8080 (video in oupput)
EXPOSE 5000/tcp
EXPOSE 8080/tcp

# Base layer
FROM common AS development-base

# Installazione dei package **essenziali**
RUN apt-get update && apt-get install -y  \
    git nano sudo python3-pip             \
    wget python3-venv

# Argomenti per definire ID utente e ID gruppo
ARG USER_ID=1000
ARG GROUP_ID=1000

# Creazione di una directory per HLS e impostazione dei permessi
# Crea una cartella per il ramdisk e montalo
RUN mkdir /mnt/hls
RUN chmod a+rw /mnt/hls

# Imposta le variabili d'ambiente nel Dockerfile
ENV USER_ID=$USER_ID
ENV GROUP_ID=$GROUP_ID

# Creazione di un gruppo 'developer'
RUN groupadd -g $GROUP_ID developer

# Creazione di un utente develper con permessi sudo e video
# https://jira.insiel.it/browse/FV-133
RUN useradd -rm -d /home/developer -s /bin/bash -g ${GROUP_ID} -G sudo,video -u ${USER_ID} developer
RUN echo 'developer:developer' | chpasswd

WORKDIR /home/developer

# Timezone
ENV TZ=Europe/Rome
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Aggiungi il repository universe per avere accesso agli alternative
RUN apt-get update && apt-get install -y software-properties-common apt-utils \
                        gcc git nano curl wget ffmpeg libsm6 libxext6 \
                        python3-dev

# Nginx
RUN apt-get update && apt-get install -y nginx
COPY ./build/assets/nginx/editor /usr/share/nginx/html/editor
COPY ./build/assets/nginx/players /usr/share/nginx/html/output-stream
COPY ./build/assets/nginx/config/nginx.conf /etc/nginx/nginx.conf

##############################################################
#
#            P y t h o n    D e p e n d e n c i e s
#
##############################################################

# Crea cartella git e sistema relativi permessi
RUN mkdir /home/developer/git
RUN chown -R developer:developer /home/developer
RUN chmod a+rw /home/developer/git

RUN chown -R developer:developer /home/developer

RUN pip3 install torch
RUN pip3 install torchvision

RUN pip3 install azure.iot.device openpyxl
RUN pip3 install flask flask_wtf Flask-CORS flask-swagger-ui configparser  cython pytz waitress
RUN pip3 install ultralytics
#RUN pip3 uninstall opencv-python -y

# Installazione dipendenze python
RUN pip3 install numpy==1.26.4

# Other dependencies
RUN pip3 install psutil==6.0.0 openpyxl==3.1.5

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
