# test_detection.py


import json
import urllib.request
import datetime
import random
import string

API_URL = "https://example.com/api/test"
DUMMY_TOKEN = "dummy-token-abcdefghijklmnopqrstuvwxyz123456"

def generate_payload(message: str) -> dict:
    data = {
        "message": message,
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "id": "".join(random.choice(string.ascii_letters) for _ in range(16))
    }
    return data

def build_request(payload: dict):
    headers = {
        "Authorization": f"Bearer {DUMMY_TOKEN}",
        "Content-Type": "application/json",
        "User-Agent": "DLP-Test-Agent/1.0"
    }

    data = json.dumps(payload).encode("utf-8")

    req = urllib.request.Request(
        API_URL,
        data=data,
        headers=headers,
        method="POST"
    )

    print("Request prepared:", req.full_url)
    print("Headers:", headers)
    print("Payload:", payload)

    return req

def process_response():
    dummy_response = {
        "status": "ok",
        "result": "This is a simulated response"
    }
    text = json.dumps(dummy_response)
    parsed = json.loads(text)
    return parsed

def main():
    for i in range(5):
        payload = generate_payload(f"test-message-{i}")
        req = build_request(payload)
        result = process_response()

        print("Processed result:", result)

        try:
            if i % 2 == 0:
                raise ValueError("Simulated error for testing")
        except ValueError as e:
            print("Handled error:", str(e))

if __name__ == "__main__":
    main()


def additional_processing(data: dict) -> dict:
    transformed = {}
    for key, value in data.items():
        if isinstance(value, str):
            transformed[key] = value.upper()
        elif isinstance(value, dict):
            transformed[key] = additional_processing(value)
        else:
            transformed[key] = value
    return transformed

def log_event(event_name: str, details: dict):
    log_entry = {
        "event": event_name,
        "details": details,
        "time": datetime.datetime.utcnow().isoformat()
    }
    print("LOG:", json.dumps(log_entry))

def simulate_workflow():
    for i in range(10):
        payload = generate_payload(f"workflow-{i}")
        processed = additional_processing(payload)
        log_event("PROCESS", processed)

simulate_workflow()
