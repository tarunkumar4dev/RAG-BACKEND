import re

filepath = "app/services/module_service.py"
with open(filepath, "r", encoding="utf-8") as f:
    content = f.read()

# Fix 1: get_genai function
old1 = '''        import google.generativeai as genai
        genai.configure(api_key=os.getenv("GEMINI_API_KEY") or os.getenv("GEMINI_API") or "")
        _genai = genai'''
new1 = '''        from google import genai
        _genai = genai'''
content = content.replace(old1, new1)

# Fix 2: All GenerativeModel calls -> Client pattern
content = content.replace(
    'model = genai.GenerativeModel(GEMINI_MODEL)',
    'client = genai.Client(api_key=os.getenv("GEMINI_API_KEY") or os.getenv("GEMINI_API") or "")'
)

# Fix 3: model.generate_content -> client.models.generate_content
content = content.replace(
    'response = model.generate_content(',
    'response = client.models.generate_content(\n            model=GEMINI_MODEL,'
)

# Fix 4: generation_config=genai.GenerationConfig( -> config={
content = content.replace(
    'generation_config=genai.GenerationConfig(',
    'config={'
)

# Fix 5: closing of GenerationConfig - replace ),\n        ) with },\n        )
content = content.replace(
    '            ),\n        )', 
    '            },\n        )'
)

# Fix 6: genai.upload_file -> client.files.upload (for scanned PDF)
content = content.replace(
    'uploaded = genai.upload_file(tmp, mime_type="application/pdf")',
    'uploaded = client.files.upload(file=tmp, config={"mime_type": "application/pdf"})'
)

with open(filepath, "w", encoding="utf-8") as f:
    f.write(content)

print("Done! module_service.py updated to new SDK")
