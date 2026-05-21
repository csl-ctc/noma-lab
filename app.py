from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import urllib.request
import json
import os
from pathlib import Path
from functools import lru_cache

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


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
# GitHub Actions 内で生成した trained_model/ を Cloud Run に含める前提
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
    lru_cache により、初回だけ読み込み、以降はメモリ上のモデルを再利用する。
    """
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
    return tokenizer, model


def classify_text(text: str) -> dict:
    """
    学習済み DistilBERT 分類モデルで入力テキストを分類する。
    今回のモデルは IMDb データセットで学習した positive / negative 分類モデル。
    """
    try:
        tokenizer, model = load_classifier()
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))

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

    return result["candidates"][0]["content"]["parts"][0]["text"]


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
    """
    学習済みモデル単体の動作確認用エンドポイント。
    """
    if not body.message:
        raise HTTPException(status_code=400, detail="message is required")

    classification = classify_text(body.message)

    return {
        "input": body.message,
        "classification": classification,
    }


@app.post("/chat")
async def chat(body: ChatRequest):
    """
    既存の Gemini チャットに、学習済み分類モデルの結果を組み込む。
    """
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
