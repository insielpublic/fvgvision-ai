#!/bin/bash

# Nome del container
CONTAINER_NAME="fvgvision-ai-main"

# Controlla se il container è in esecuzione e fermalo
if [ "$(docker ps -q -f name=$CONTAINER_NAME)" ]; then
    echo "Arresto del container $CONTAINER_NAME in corso..."
    docker stop $CONTAINER_NAME
fi

# Rimuovi il container
if [ "$(docker ps -a -q -f name=$CONTAINER_NAME)" ]; then
    echo "Rimozione del container $CONTAINER_NAME in corso..."
    docker rm $CONTAINER_NAME
    echo "Il container $CONTAINER_NAME è stato rimosso."
else
    echo "Nessun container con il nome $CONTAINER_NAME trovato."
fi

