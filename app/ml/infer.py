import base64
import io
import os
import torch
import numpy as np
from PIL import Image

MODEL_PATH = os.getenv("MODEL_PATH", "artifacts/models/mnist_scripted.pt")

_model = None

def _load():
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise RuntimeError(f"Model not found at {MODEL_PATH}. Run: python scripts/train_mnist.py")
        _model = torch.jit.load(MODEL_PATH).eval()
    return _model

def _decode_data_url(s: str) -> bytes:
    # supports "data:image/png;base64,...."
    if "," in s and s.strip().startswith("data:"):
        s = s.split(",", 1)[1]
    return base64.b64decode(s)

def predict_digit_from_base64(image_base64: str):
    model = _load()
    raw = _decode_data_url(image_base64)

    img = Image.open(io.BytesIO(raw)).convert("L").resize((28, 28))
    arr = np.array(img).astype("float32") / 255.0
    # MNIST-like: black background white digit; invert if needed
    # arr = 1.0 - arr

    x = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)  # (1,1,28,28)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).squeeze(0).tolist()
        pred = int(torch.argmax(logits, dim=1).item())
    return pred, probs
