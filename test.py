import requests

api_key = "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"

headers = {
    "Authorization": f"Bearer {api_key}",
    "HTTP-Referer": "http://localhost:8501",  # or your actual local URL
    "Content-Type": "application/json"
}

body = {
    "model": "mistralai/mistral-7b-instruct",
    "messages": [{"role": "user", "content": "Say hello"}]
}

response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=body)

print("Status:", response.status_code)
print("Response:", response.text)
