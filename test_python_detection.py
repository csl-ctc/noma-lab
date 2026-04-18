# test_python_detection.py

from fastapi import FastAPI
import json

app = FastAPI()

DUMMY_API_KEY = "test-api-key-123456"

@app.post("/test")
async def test_endpoint(body: dict):
    response = {
        "message": "This is a test response",
        "input": body.get("message", ""),
        "api_key": DUMMY_API_KEY
    }
    return response

def helper_function():
    data = {"status": "ok"}
    return json.dumps(data)

if __name__ == "__main__":
    print("Test script executed")
