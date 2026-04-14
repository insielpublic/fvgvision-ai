#!/bin/bash
source .build-env
source .env

DOCKER_BASE_DIR=.
DOCKER_IMAGE_NAME=$BUILD_IMAGE_NAME
DOCKER_IMAGE_VERSION=${BUILD_IMAGE_VERSION}-${BUILD_IMAGE_PLATFORM}

echo "Effettuo build $DOCKER_IMAGE_NAME:$DOCKER_IMAGE_VERSION"

DEPLOY=false
DEPLOY_AGX=false
APP_OVERRIDE=false
BUILD_TYPE=""
APP_DIR="$APP_PATH"
ASSETS_DIR="../assets"
DOCKER_FILENAME=Dockerfile


usage() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  -h, --help               Display this help message"
    echo "  --deploy                 Push the image to Docker Hub"
    echo "  --deploy-agx             Push su Docker Hub e pull automatico sull'AGX"
    echo "  --dev                    Build the development image"
    echo "  --dev-base               Build the base development image"
    echo "  --prod                   Build the production image, with a clear build"
    echo "  --prod-fast              Build the production image, starting with intermediate image"
    echo "  --app=<path>             Specify the application directory path (required for production)"
    exit 1
}

while getopts ":h-:" opt; do
    case $opt in
        -)
            case "${OPTARG}" in
                help)
                    usage
                    ;;
                deploy)
                    DEPLOY=true
                    ;;
                deploy-agx)
                    DEPLOY=true
                    DEPLOY_AGX=true
                    ;;
                dev)
                    BUILD_TYPE="development"
                    ;;
                dev-base)
                    BUILD_TYPE="development-base"
                    ;;
                prod)
                    BUILD_TYPE="production"
                    ;;
                prod-fast)
                    BUILD_TYPE="production"
                    DOCKER_FILENAME="Dockerfile.fast"
                    ;;
                app=*)
                    APP_DIR="${OPTARG#*=}"
                    APP_OVERRIDE=true
                    ;;
                *)
                    echo "Invalid option: --$OPTARG"
                    usage
                    ;;
            esac
            ;;
        h)
            usage
            ;;
        *)
            echo "Invalid option: -$OPTARG"
            usage
            ;;
    esac
done

shift $((OPTIND -1))

if [[ -z "$BUILD_TYPE" ]]; then
    echo "Devi specificare almeno uno dei seguenti parametri: --dev, --prod o --dev-base."
    exit 1
fi

if [[ "$BUILD_TYPE" == "development" && "$APP_OVERRIDE" = true ]]; then
    echo "L'opzione --app non può essere utilizzata con --dev."
    exit 1
fi

if [[ "$BUILD_TYPE" == "production" && -z "$APP_DIR" ]]; then
    echo "[$APP_DIR] Deve essere specificata la cartella dell'applicazione quando si utilizza --prod."
    exit 1
fi

# Preparazione contesto di build
BUILD_DIR="$DOCKER_BASE_DIR/build"

if [ -d "$BUILD_DIR" ]; then
    echo "La cartella build esiste già. Verrà rimossa..."
    rm -rf "$BUILD_DIR"
fi
mkdir -p "$BUILD_DIR"

ABSOLUTE_BUILD_DIR=$(realpath "$BUILD_DIR")
echo "Il percorso assoluto della cartella di build è: $ABSOLUTE_BUILD_DIR"

echo "Copio la cartella assets in $BUILD_DIR/assets"
cp -r "$ASSETS_DIR" "$BUILD_DIR/assets"

if [[ "$BUILD_TYPE" == "production" ]]; then
    if [ -d "$APP_DIR" ]; then
        APP_SIZE=$(du -sh "$APP_DIR" | cut -f1)
        echo "Dimensione della cartella app: $APP_SIZE"
        cp -r "$APP_DIR" "$BUILD_DIR/app"
    else
        echo "La cartella specificata in --app non esiste: $APP_DIR"
        exit 1
    fi
fi

# Suffisso versione
if [[ "$BUILD_TYPE" == "development" ]]; then
    DOCKER_IMAGE_VERSION="${DOCKER_IMAGE_VERSION}${BUILD_IMAGE_DEVELOPMENT_PREFIX}"
elif [[ "$BUILD_TYPE" == "development-base" ]]; then
    DOCKER_IMAGE_VERSION="${DOCKER_IMAGE_VERSION}${BUILD_IMAGE_DEVELOPMENT_BASE_PREFIX}"
elif [[ "$BUILD_TYPE" == "production-fast" ]]; then
    DOCKER_IMAGE_VERSION="${DOCKER_IMAGE_VERSION}${BUILD_IMAGE_PRODUCTION_PREFIX}"
elif [[ "$BUILD_TYPE" == "production" ]]; then
    DOCKER_IMAGE_VERSION="${DOCKER_IMAGE_VERSION}${BUILD_IMAGE_PRODUCTION_PREFIX}"
fi

FULL_IMAGE_TAG="$DOCKER_ACCOUNT_NAME/$DOCKER_IMAGE_NAME:$DOCKER_IMAGE_VERSION"
echo "Building $FULL_IMAGE_TAG (linux/arm64) con builder distribuito..."

# Build con buildx — arm64 nativo sull'AGX
docker buildx build \
    --builder multiarch-builder \
    --platform linux/arm64 \
    --build-arg USER_ID=$USER_ID \
    --build-arg GROUP_ID=$GROUP_ID \
    --target $BUILD_TYPE \
    -f $DOCKER_BASE_DIR/$DOCKER_FILENAME \
    --tag $FULL_IMAGE_TAG \
    --push \
    $DOCKER_BASE_DIR

if [ $? -ne 0 ]; then
    echo "Build fallita."
    exit 1
fi

echo "Build completata: $FULL_IMAGE_TAG"

# --deploy-agx: pull automatico sull'AGX
if [ "$DEPLOY_AGX" = true ]; then
    echo ""
    echo "Eseguo pull dell'immagine sull'AGX ($AGX_USER@$AGX_HOST)..."
    ssh ${AGX_USER}@${AGX_HOST} "docker pull $FULL_IMAGE_TAG"

    if [ $? -ne 0 ]; then
        echo "Pull sull'AGX fallito."
        exit 1
    fi

    echo "Immagine disponibile sull'AGX: $FULL_IMAGE_TAG"
fi

# --deploy senza --deploy-agx
if [ "$DEPLOY" = false ]; then
    echo ""
    echo "Nota: l'immagine è stata pushata su Docker Hub (obbligatorio con buildx)."
    echo "Per deployarla sull'AGX esegui: $0 --deploy-agx --${BUILD_TYPE}"
fi