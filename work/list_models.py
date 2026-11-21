import os

import requests

API_KEY = os.environ["GEMINI_API_KEY"]  # never hard-code keys (the old one is in git history: revoke it)

url = "https://generativelanguage.googleapis.com/v1/models"

response = requests.get(url, headers={"x-goog-api-key": API_KEY}, timeout=30)

if response.status_code == 200:
    models = response.json()
    print("✅ Available models:\n")
    for model in models.get('models', []):
        name = model.get('name', '')
        if 'gemini' in name.lower():
            print(f"  📌 {name}")
else:
    print(f"❌ Error: {response.status_code}")
    print(response.text)