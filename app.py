from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import urllib.request
import json
import os
from pathlib import Path
from functools import lru_cache

app = FastAPI()

# ===== Vertex AI / Gemini 設定 =====
PROJECT_ID = "my-project-csl-486600"
REGION = "asia-northeast1"
MODEL = "gemini-2.5-flash"
URL = (
    f"https://{REGION}-aiplatform.googleapis.com/v1/"
    f"projects/{PROJECT_ID}/locations/{REGION}/publishers/google/"
    f"models/{MODEL}:generateContent"
)

# ===== 学習済み Hugging Face モデル設定 =====
MODEL_DIR = os.environ.get("MODEL_DIR", "trained_model")

LABEL_MAP = {
    0: "negative",
    1: "positive",
}


class ChatRequest(BaseModel):
    message: str


def get_access_token() -> str:
    req = urllib.request.Request(
        "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token",
        headers={"Metadata-Flavor": "Google"},
    )
    with urllib.request.urlopen(req, timeout=5) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data["access_token"]


@lru_cache(maxsize=1)
def load_classifier():
    """
    trained_model/ から tokenizer と model を読み込む。
    torch / transformers はここで初めて import する。
    そのため python -c "import app" だけでは torch が不要になる。
    """
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    model_path = Path(MODEL_DIR)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model directory not found: {MODEL_DIR}. "
            "Make sure trained_model/ is generated before Cloud Run deployment "
            "and included in the deployment source."
        )

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_DIR,
        local_files_only=True,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_DIR,
        local_files_only=True,
    )

    model.eval()

    return tokenizer, model, torch


def classify_text(text: str) -> dict:
    """
    学習済みモデルで入力文を positive / negative に分類する。
    """
    try:
        tokenizer, model, torch = load_classifier()
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except ModuleNotFoundError as e:
        raise HTTPException(
            status_code=500,
            detail=(
                f"Required ML library is missing: {e}. "
                "Make sure torch and transformers are installed in the runtime."
            ),
        )

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=128,
    )

    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.softmax(outputs.logits, dim=-1)
        predicted_class = int(torch.argmax(probabilities, dim=-1).item())
        confidence = float(probabilities[0][predicted_class].item())

    return {
        "label": LABEL_MAP.get(predicted_class, f"LABEL_{predicted_class}"),
        "confidence": confidence,
    }


def call_gemini(message: str) -> str:
    token = get_access_token()

    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {
                        "text": message
                    }
                ],
            }
        ]
    }

    req = urllib.request.Request(
        URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=30) as resp:
        result = json.loads(resp.read().decode("utf-8"))

    text = result["candidates"][0]["content"]["parts"][0]["text"]
    return text


@app.get("/")
async def root():
    return {
        "status": "ok",
        "model_dir": MODEL_DIR,
        "model_dir_exists": Path(MODEL_DIR).exists(),
    }


@app.get("/model-status")
async def model_status():
    model_path = Path(MODEL_DIR)

    return {
        "model_dir": MODEL_DIR,
        "exists": model_path.exists(),
        "files": [p.name for p in model_path.iterdir()] if model_path.exists() else [],
    }


@app.post("/classify")
async def classify(body: ChatRequest):
    if not body.message:
        raise HTTPException(status_code=400, detail="message is required")

    classification = classify_text(body.message)

    return {
        "input": body.message,
        "classification": classification,
    }


@app.post("/chat")
async def chat(body: ChatRequest):
    if not body.message:
        raise HTTPException(status_code=400, detail="message is required")

    classification = classify_text(body.message)

    enriched_prompt = f"""
ユーザー入力:
{body.message}

事前分類結果:
- label: {classification["label"]}
- confidence: {classification["confidence"]:.3f}

上記の分類結果は、IMDbデータセットでファインチューニングした
Hugging Face分類モデルによる参考情報です。
分類結果を参考にしつつ、必要以上に断定せず、ユーザー入力に対して自然に回答してください。
"""

    text = call_gemini(enriched_prompt)

    return {
        "input": body.message,
        "classification": classification,
        "response": text,
    }
