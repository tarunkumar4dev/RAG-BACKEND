import os
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get API key
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    print("❌ GEMINI_API_KEY not found in .env file")
    exit(1)

print(f"✅ API Key found: {api_key[:10]}...")

# Configure Gemini
genai.configure(api_key=api_key)

# Test models
models_to_test = [
    "models/gemini-2.0-flash",
    "models/gemini-1.5-flash",
    "models/gemini-pro"
]

for model_name in models_to_test:
    try:
        print(f"\n🔧 Testing model: {model_name}")
        model = genai.GenerativeModel(model_name)
        response = model.generate_content("Say 'Hello World'")
        print(f"✅ Success: {response.text}")
        print(f"✅ Model works: {model_name}")
        break
    except Exception as e:
        print(f"❌ Failed with {model_name}: {str(e)}")