import requests

API_KEY = "YOUR_GEMINI_API_KEY_HERE"  # Put your actual API key

url = f"https://generativelanguage.googleapis.com/v1/models?key={"AIzaSyDK21ykZZfXcpnZfhNj4aXQ60PPcy5-efw "}"

response = requests.get(url)

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