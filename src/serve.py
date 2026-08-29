from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import Path

import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
from torchvision import transforms

from model import get_model


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "checkpoints"
    / "best_model.pt"
)

CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

NUM_CLASSES = len(CLASS_NAMES)

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ---------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------

# IMPORTANT:
# These values must match dataset.py validation transforms.
transform = transforms.Compose(
    [
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[
                0.4914,
                0.4822,
                0.4465,
            ],
            std=[
                0.2470,
                0.2435,
                0.2616,
            ],
        ),
    ]
)


# ---------------------------------------------------------
# Global model state
# ---------------------------------------------------------

model = None
MODEL_LOADED = False
MODEL_ERROR = None


# ---------------------------------------------------------
# Model loading
# ---------------------------------------------------------

def load_model():
    global model
    global MODEL_LOADED
    global MODEL_ERROR

    try:
        if not CHECKPOINT_PATH.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {CHECKPOINT_PATH}"
            )

        # Build exactly the same architecture used during training.
        model = get_model(
            architecture="resnet18",
            num_classes=NUM_CLASSES,
        )

        checkpoint = torch.load(
            CHECKPOINT_PATH,
            map_location=device,
        )

        # Your train.py saves a dictionary like:
        #
        # {
        #   "epoch": ...,
        #   "model_state_dict": ...,
        #   "optimizer_state_dict": ...,
        #   "metrics": ...,
        #   "config": ...
        # }

        if "model_state_dict" not in checkpoint:
            raise ValueError(
                "Checkpoint does not contain "
                "'model_state_dict'"
            )

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        model.to(device)
        model.eval()

        MODEL_LOADED = True
        MODEL_ERROR = None

        print(
            {
                "event": "model_loaded",
                "checkpoint": str(CHECKPOINT_PATH),
                "device": str(device),
                "epoch": checkpoint.get("epoch"),
                "metrics": checkpoint.get("metrics"),
            }
        )

    except Exception as exc:
        model = None
        MODEL_LOADED = False
        MODEL_ERROR = str(exc)

        print(
            {
                "event": "model_load_failed",
                "error": MODEL_ERROR,
            }
        )


# ---------------------------------------------------------
# FastAPI lifespan
# ---------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()

    yield

    # Optional cleanup.
    global model

    model = None

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


app = FastAPI(
    title="CIFAR-10 Image Classifier",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------

@app.get("/health")
def health():
    if not MODEL_LOADED:
        raise HTTPException(
            status_code=503,
            detail={
                "status": "unhealthy",
                "model_loaded": False,
                "error": MODEL_ERROR,
            },
        )

    return {
        "status": "healthy",
        "model_loaded": True,
        "device": str(device),
        "checkpoint": str(CHECKPOINT_PATH),
    }


# ---------------------------------------------------------
# Prediction endpoint
# ---------------------------------------------------------

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
):
    if not MODEL_LOADED or model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is unavailable",
        )

    # -----------------------------------------------------
    # Validate file type
    # -----------------------------------------------------

    if file.content_type is not None:
        if not file.content_type.startswith("image/"):
            raise HTTPException(
                status_code=400,
                detail="Uploaded file must be an image",
            )

    # -----------------------------------------------------
    # Read image
    # -----------------------------------------------------

    try:
        image_bytes = await file.read()

        if not image_bytes:
            raise HTTPException(
                status_code=400,
                detail="Uploaded image is empty",
            )

        image = Image.open(
            BytesIO(image_bytes)
        ).convert("RGB")

    except UnidentifiedImageError:
        raise HTTPException(
            status_code=400,
            detail="Invalid or unsupported image",
        )

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Unable to read image: {exc}",
        )

    # -----------------------------------------------------
    # Preprocess
    # -----------------------------------------------------

    tensor = transform(image)

    tensor = (
        tensor
        .unsqueeze(0)
        .to(device)
    )

    # -----------------------------------------------------
    # Inference
    # -----------------------------------------------------

    try:
        with torch.inference_mode():
            outputs = model(tensor)

            probabilities = F.softmax(
                outputs,
                dim=1,
            )[0]

            predicted_index = int(
                torch.argmax(probabilities).item()
            )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Inference failed: {exc}",
        )

    # -----------------------------------------------------
    # Response
    # -----------------------------------------------------

    predictions = {
        CLASS_NAMES[i]: round(
            float(probabilities[i].item()),
            6,
        )
        for i in range(NUM_CLASSES)
    }

    return {
        "predicted_class": CLASS_NAMES[
            predicted_index
        ],
        "confidence": round(
            float(
                probabilities[
                    predicted_index
                ].item()
            ),
            6,
        ),
        "predictions": predictions,
    }