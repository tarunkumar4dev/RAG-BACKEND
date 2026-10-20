import os

import requests

# Key comes from the environment, never from source (this repo is public).
API_KEY = os.environ["GEMINI_API_KEY"]

url = "https://generativelanguage.googleapis.com/v1/models"

response = requests.get(url, params={"key": API_KEY}, timeout=30)

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
