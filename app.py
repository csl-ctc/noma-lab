from fastapi import FastAPI
import urllib.request
import json

app = FastAPI()

PROJECT_ID = "my-project-csl-486600"
REGION = "asia-northeast1"
MODEL = "gemini-2.5-flash"
URL = f"https://{REGION}-aiplatform.googleapis.com/v1/projects/{PROJECT_ID}/locations/{REGION}/publishers/google/models/{MODEL}:generateContent"

def get_access_token() -> str:
    req = urllib.request.Request(
        "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token",
        headers={"Metadata-Flavor": "Google"},
    )
    with urllib.request.urlopen(req, timeout=5) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data["access_token"]

@app.post("/chat")
async def chat(body: dict):
    token = get_access_token()
    payload = {
        "contents": [{"role": "user", "parts": [{"text": body["message"]}]}]
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
    return {"response": text}
