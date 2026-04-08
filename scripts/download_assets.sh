#!/bin/bash

# Funzione per leggere il file INI e ottenere un valore basato sulla sezione e sulla chiave
function get_ini_value() {
    local file=$1
    local section=$2
    local key=$3
    awk -F= -v section="[$section]" -v key="$key" '
    $0 == section {found=1}
    found && $1 ~ key {gsub(/^ +| +$/, "", $2); print $2; exit}
    ' "$file"
}

# Funzione per scaricare un file da un URL e salvarlo in un percorso locale con messaggi di avanzamento
function download_file() {
    local url=$1
    local local_path=$2
    local platform_name=$3

    # Estrai il nome del file dall'URL
    local file_name=$(basename "$url")

    # Costruisci il percorso completo dove salvare il file
    local full_local_path="$local_path/$file_name"

    # Crea la directory in modo ricorsivo se non esiste
    mkdir -p "$(dirname "$full_local_path")"

    # Scarica il file senza barra di avanzamento
    echo "Downloading $file_name to $full_local_path..."
    if curl -s -L -o "$full_local_path" "$url"; then
        echo "Downloaded $file_name successfully."
    else
        echo "Failed to download $file_name."
    fi

    # Crea il file {platform_name}_assets.txt nella directory del file scaricato
    local assets_file_path="$local_path/${platform_name}_assets.txt"
    touch "$assets_file_path"
    #echo "Created assets file: $assets_file_path"
}

# Funzione per verificare se la piattaforma ha già gli asset scaricati
function check_assets_exists() {
    local platform_name=$1
    local config_file=$2

    local i=1
    while true; do
        local file_path
        file_path=$(get_ini_value "$config_file" "$platform_name" "file${i}_path")
        if [[ -z $file_path ]]; then
            break
        fi

        local assets_file_path="$file_path/${platform_name}_assets.txt"

        if [[ -f $assets_file_path ]]; then
            echo "Skipping download for $platform_name. Assets file already exists: $assets_file_path"
            return 0
        fi

        ((i++))
    done

    return 1
}

# Funzione principale
function main() {
    local config_file="./scripts/download_assets.ini"

    # Leggi le sezioni dal file INI (lista delle piattaforme disponibili)
    mapfile -t platforms < <(awk -F'[][]' '/\[.*\]/ {print $2}' "$config_file")
    if [[ ${#platforms[@]} -eq 0 ]]; then
        echo "No platforms found in the configuration file."
        exit 1
    fi

    # Se la piattaforma non è stata specificata da CLI, mostra le sezioni disponibili
    if [[ -z $1 ]]; then
        echo "Available platforms (sections in the INI file):"
        for platform in "${platforms[@]}"; do
            echo "- $platform"
        done
        exit 0
    fi

    # Verifica che la piattaforma specificata da CLI esista nel file di configurazione
    selected_platform="$1"
    if [[ ! " ${platforms[*]} " =~ " ${selected_platform} " ]]; then
        echo "Platform '$selected_platform' not found in the configuration file."
        exit 1
    fi

    echo "Selected platform: $selected_platform"

    # Controlla se il file assets esiste, se sì, salta il download
    if check_assets_exists "$selected_platform" "$config_file"; then
        exit 0
    fi

    # Procedi con il download dei file per la piattaforma selezionata
    local i=1
    while true; do
        local file_url file_path
        file_url=$(get_ini_value "$config_file" "$selected_platform" "file${i}_url")
        file_path=$(get_ini_value "$config_file" "$selected_platform" "file${i}_path")

        if [[ -z $file_url || -z $file_path ]]; then
            break
        fi

        download_file "$file_url" "$file_path" "$selected_platform"
        ((i++))
    done
}

# Avvia il programma passando l'argomento della piattaforma se fornito
main "$1"
