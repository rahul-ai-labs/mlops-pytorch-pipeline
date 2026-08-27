from io import BytesIO

import torch
import torch.nn.functional as F

from PIL import Image

from fastapi import FastAPI, File, UploadFile, HTTPException

from torchvision import transforms

from model import build_model


CHECKPOINT_PATH = "checkpoints/resnet18_cifar10.pth"

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
    "truck"
]

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

app = FastAPI()

model = build_model(10)

try:
    model.load_state_dict(
        torch.load(
            CHECKPOINT_PATH,
            map_location=device
        )
    )

    model.to(device)
    model.eval()

    MODEL_LOADED = True

except Exception:
    MODEL_LOADED = False


transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010]
    )
])


@app.get("/health")
def health():
    if not MODEL_LOADED:
        raise HTTPException(
            status_code=500,
            detail="Model not loaded"
        )

    return {"status": "healthy"}


@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):
    if not MODEL_LOADED:
        raise HTTPException(
            status_code=500,
            detail="Model unavailable"
        )

    image_bytes = await file.read()

    image = Image.open(
        BytesIO(image_bytes)
    ).convert("RGB")

    tensor = transform(image)

    tensor = tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)

        probs = F.softmax(
            outputs,
            dim=1
        )[0]

    return {
        "predictions": {
            CLASS_NAMES[i]: float(probs[i])
            for i in range(len(CLASS_NAMES))
        }
    }