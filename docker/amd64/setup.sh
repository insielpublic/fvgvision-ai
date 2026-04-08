#!/bin/bash

# Ottieni l'UID e il GID correnti
USER_ID=$(id -u)
GROUP_ID=$(id -g)

# Chiedi all'utente il percorso dell'applicazione
read -p "Inserisci il percorso dell'applicazione (default: ../../src): " APP_PATH
APP_PATH=${APP_PATH:-../../src} # Se non viene inserito nulla, usa ../../src
APP_PATH=$(realpath "$APP_PATH")

# Verifica l'esistenza della directory runtime e crea se non esiste
RUNTIME_DIR=./runtime
if [ ! -d "$RUNTIME_DIR" ]; then
    echo "La cartella runtime non esiste. Creazione della cartella..."
    mkdir -p "$RUNTIME_DIR"
fi

# Verifica l'esistenza della directory app e crea se non esiste
if [ ! -d "$APP_PATH" ]; then
    echo "La cartella app non esiste. Creazione della cartella in $APP_PATH..."
    mkdir -p "$APP_PATH"
fi

# Scrivi l'UID, il GID, il percorso di CUDA e APP_PATH nel file .env
echo "USER_ID=$USER_ID" > .env
echo "GROUP_ID=$GROUP_ID" >> .env
echo "APP_PATH=$APP_PATH" >> .env

echo "Le variabili USER_ID=$USER_ID, GROUP_ID=$GROUP_ID e APP_PATH=$APP_PATH sono state scritte nel file .env."
