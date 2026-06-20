from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
embedding = model.encode("Hello world test")
print(f"✅ Local embeddings work! Dimensions: {len(embedding)}")