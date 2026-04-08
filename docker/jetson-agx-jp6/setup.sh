#!/bin/bash

# Ottieni l'UID e il GID correnti
USER_ID=$(id -u)
GROUP_ID=$(id -g)

# Verifica se CUDA è nel PATH
CUDA_PATH=$(command -v nvcc)
if [ -z "$CUDA_PATH" ]; then
    # CUDA non è nel PATH, verificare /usr/local/cuda
    if [ -L "/usr/local/cuda" ]; then
        CUDA_PATH=$(readlink -f /usr/local/cuda)
    elif [ -d "/usr/local/cuda" ]; then
        CUDA_PATH="/usr/local/cuda"
    else
        echo "CUDA non è installato o nvcc non è nel PATH e /usr/local/cuda non esiste."
        exit 1
    fi
else
    CUDA_PATH=$(dirname $(dirname "$CUDA_PATH"))
fi

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
echo "CUDA_PATH=$CUDA_PATH" >> .env
echo "APP_PATH=$APP_PATH" >> .env

echo "Le variabili USER_ID=$USER_ID, GROUP_ID=$GROUP_ID, CUDA_PATH=$CUDA_PATH e APP_PATH=$APP_PATH sono state scritte nel file .env."

# Aggiungi le righe al file .bashrc nella cartella runtime
BASHRC_FILE=./runtime/.bashrc

if [ ! -f "$BASHRC_FILE" ]; then
    touch "$BASHRC_FILE"
fi

if ! grep -qF "export PATH=$CUDA_PATH/bin:\$PATH" "$BASHRC_FILE"; then
    echo "export PATH=$CUDA_PATH/bin:\$PATH" >> "$BASHRC_FILE"
fi

if ! grep -qF "export LD_LIBRARY_PATH=$CUDA_PATH/lib64:\$LD_LIBRARY_PATH" "$BASHRC_FILE"; then
    echo "export LD_LIBRARY_PATH=$CUDA_PATH/lib64:\$LD_LIBRARY_PATH" >> "$BASHRC_FILE"
fi

echo "Le righe per il percorso di CUDA sono state aggiunte a $BASHRC_FILE nella stessa cartella."

