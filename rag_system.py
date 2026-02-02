"""
NCERT RAG SYSTEM - PRODUCTION READY v4.0
Minimal, Fast, Vector Search Enabled
"""

import os
import json
import time
import hashlib
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path

import psycopg2
from psycopg2.extras import RealDictCursor
import google.generativeai as genai

# For semantic search
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

# ========== CONFIGURATION ==========
class Config:
    # Database
    DB_HOST = "db.dcmnzvjftmdbywrjkust.supabase.co"
    DB_NAME = "postgres"
    DB_USER = "postgres"
    DB_PASSWORD = os.getenv("DATABASE_PASSWORD", "")
    DB_PORT = "5432"
    
    # Gemini
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
    MODEL_NAME = "gemini-2.0-flash"
    
    # Vector Search
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Lightweight, 384 dimensions
    FAISS_INDEX_PATH = "faiss_index/index.faiss"
    METADATA_PATH = "faiss_index/metadata.json"
    
    # Performance
    CHUNK_LIMIT = 5
    SIMILARITY_THRESHOLD = 0.7
    CACHE_TTL = 300  # 5 minutes
    
    @classmethod
    def validate(cls):
        if not cls.DB_PASSWORD:
            raise ValueError("DATABASE_PASSWORD not set")
        if not cls.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY not set")
        return True

# ========== SEMANTIC SEARCH ENGINE ==========
class SemanticSearch:
    """FAISS-based semantic search with metadata."""
    
    def __init__(self):
        self.embedding_model = None
        self.faiss_index = None
        self.metadata = {}
        self._load_index()
    
    def _load_index(self):
        """Load FAISS index and metadata."""
        try:
            # Load embedding model
            print("🔧 Loading embedding model...")
            self.embedding_model = SentenceTransformer(Config.EMBEDDING_MODEL)
            
            # Load FAISS index
            if Path(Config.FAISS_INDEX_PATH).exists():
                print(f"📁 Loading FAISS index from {Config.FAISS_INDEX_PATH}")
                self.faiss_index = faiss.read_index(Config.FAISS_INDEX_PATH)
            else:
                print("⚠️ FAISS index not found, creating new one...")
                self.faiss_index = faiss.IndexFlatL2(384)  # 384-dim embeddings
            
            # Load metadata
            if Path(Config.METADATA_PATH).exists():
                with open(Config.METADATA_PATH, 'r') as f:
                    data = json.load(f)
                    self.metadata = data.get("id_to_chunk", {})
                    print(f"✅ Loaded metadata for {len(self.metadata)} chunks")
            else:
                print("⚠️ Metadata file not found")
            
        except Exception as e:
            print(f"❌ Failed to load semantic search: {e}")
            raise
    
    def search(self, query: str, limit: int = 5) -> List[Dict]:
        """Semantic search using FAISS."""
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query])
            query_embedding = np.array(query_embedding, dtype='float32')
            
            # Search in FAISS
            distances, indices = self.faiss_index.search(query_embedding, limit)
            
            # Prepare results
            results = []
            for idx, distance in zip(indices[0], distances[0]):
                if idx != -1 and str(idx) in self.metadata:
                    chunk_data = self.metadata[str(idx)]
                    
                    # Convert distance to similarity score (0 to 1)
                    similarity = 1 / (1 + distance)
                    
                    if similarity >= Config.SIMILARITY_THRESHOLD:
                        results.append({
                            "id": chunk_data.get("id"),
                            "content": chunk_data.get("content", ""),
                            "chapter": chunk_data.get("chapter", ""),
                            "subject": chunk_data.get("subject", ""),
                            "class_grade": chunk_data.get("class_grade", ""),
                            "similarity": float(similarity)
                        })
            
            return results
            
        except Exception as e:
            print(f"❌ Semantic search error: {e}")
            return []

# ========== DATABASE MANAGER ==========
class DatabaseManager:
    """Simple database connection manager."""
    
    @staticmethod
    def get_connection():
        """Get database connection."""
        return psycopg2.connect(
            host=Config.DB_HOST,
            database=Config.DB_NAME,
            user=Config.DB_USER,
            password=Config.DB_PASSWORD,
            port=Config.DB_PORT,
            sslmode='require'
        )
    
    @staticmethod
    def get_chapter_list(limit: int = 50) -> List[str]:
        """Get list of available chapters."""
        try:
            conn = DatabaseManager.get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT DISTINCT chapter 
                FROM ncert_chunks 
                ORDER BY chapter
                LIMIT %s
            """, [limit])
            
            chapters = [row[0] for row in cursor.fetchall()]
            conn.close()
            
            return chapters
            
        except Exception as e:
            print(f"❌ Failed to get chapters: {e}")
            return []
    
    @staticmethod
    def get_system_stats() -> Dict:
        """Get database statistics."""
        try:
            conn = DatabaseManager.get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_chunks,
                    COUNT(DISTINCT chapter) as chapters,
                    COUNT(DISTINCT subject) as subjects
                FROM ncert_chunks
            """)
            
            row = cursor.fetchone()
            conn.close()
            
            return {
                "total_chunks": row[0] if row else 0,
                "chapters": row[1] if row else 0,
                "subjects": row[2] if row else 0
            }
            
        except Exception as e:
            print(f"❌ Failed to get stats: {e}")
            return {"error": str(e)}

# ========== AI ANSWER GENERATOR ==========
class AnswerGenerator:
    """Generate answers using Gemini AI."""
    
    def __init__(self):
        self.model = None
        self._initialize()
    
    def _initialize(self):
        """Initialize Gemini AI."""
        try:
            genai.configure(api_key=Config.GEMINI_API_KEY)
            self.model = genai.GenerativeModel(Config.MODEL_NAME)
            print(f"✅ Gemini AI initialized: {Config.MODEL_NAME}")
        except Exception as e:
            print(f"❌ Failed to initialize Gemini: {e}")
            raise
    
    def generate(self, question: str, context_chunks: List[Dict]) -> str:
        """Generate answer using context."""
        try:
            # Prepare context
            context_parts = []
            for i, chunk in enumerate(context_chunks, 1):
                context_parts.append(
                    f"[Source {i}]\n"
                    f"Chapter: {chunk.get('chapter', 'Unknown')}\n"
                    f"Subject: {chunk.get('subject', 'Unknown')}\n"
                    f"Class: {chunk.get('class_grade', 'Unknown')}\n"
                    f"Content: {chunk.get('content', '')}\n"
                )
            
            context = "\n---\n".join(context_parts)
            
            # Create prompt
            prompt = f"""You are an expert NCERT textbook assistant.

NCERT CONTEXT:
{context}

QUESTION:
{question}

INSTRUCTIONS:
1. Answer STRICTLY based on the NCERT context above
2. Use simple, student-friendly language
3. Be concise but complete
4. If information is not in context, say: "This specific information is not available in the NCERT textbook."
5. Reference the chapter and subject when relevant

ANSWER:"""
            
            # Generate response
            response = self.model.generate_content(prompt)
            return response.text.strip()
            
        except Exception as e:
            print(f"❌ Answer generation failed: {e}")
            return f"Error generating answer: {str(e)[:100]}"

# ========== MAIN RAG SYSTEM ==========
class NCERTRAGSystem:
    """Production-ready NCERT RAG System with semantic search."""
    
    def __init__(self):
        print("🚀 Initializing NCERT RAG System...")
        
        # Validate config
        Config.validate()
        
        # Initialize components
        self.semantic_search = SemanticSearch()
        self.db_manager = DatabaseManager()
        self.answer_generator = AnswerGenerator()
        
        # Cache for frequent queries
        self.cache = {}
        self.cache_timestamps = {}
        
        print("✅ RAG System initialized successfully!")
    
    def _get_cache_key(self, query: str) -> str:
        """Generate cache key."""
        return hashlib.sha256(query.lower().strip().encode()).hexdigest()[:16]
    
    def query(self, question: str, use_cache: bool = True) -> Dict[str, Any]:
        """
        Process query with semantic search.
        
        Args:
            question: User's question
            use_cache: Whether to use cache
            
        Returns:
            Dictionary with answer and metadata
        """
        start_time = time.time()
        
        # Check cache
        if use_cache:
            cache_key = self._get_cache_key(question)
            if cache_key in self.cache:
                cache_time = self.cache_timestamps.get(cache_key, 0)
                if time.time() - cache_time < Config.CACHE_TTL:
                    result = self.cache[cache_key]
                    result["response_time"] = time.time() - start_time
                    result["cache_hit"] = True
                    return result
        
        try:
            # Step 1: Semantic search
            search_results = self.semantic_search.search(
                question, 
                limit=Config.CHUNK_LIMIT
            )
            
            if not search_results:
                return {
                    "success": False,
                    "answer": "No relevant NCERT content found for your question.",
                    "chunks_used": 0,
                    "chapters": [],
                    "similarity_score": 0,
                    "response_time": time.time() - start_time,
                    "model_used": "none"
                }
            
            # Step 2: Calculate average similarity
            avg_similarity = sum(r["similarity"] for r in search_results) / len(search_results)
            
            # Step 3: Generate answer
            answer = self.answer_generator.generate(question, search_results)
            
            # Step 4: Extract chapters
            chapters = list(set(r["chapter"] for r in search_results if r["chapter"]))
            
            # Prepare result
            result = {
                "success": True,
                "answer": answer,
                "chunks_used": len(search_results),
                "chapters": chapters,
                "similarity_score": avg_similarity,
                "response_time": time.time() - start_time,
                "model_used": Config.MODEL_NAME,
                "cache_hit": False
            }
            
            # Cache result
            if use_cache:
                cache_key = self._get_cache_key(question)
                self.cache[cache_key] = result
                self.cache_timestamps[cache_key] = time.time()
            
            return result
            
        except Exception as e:
            print(f"❌ Query processing failed: {e}")
            return {
                "success": False,
                "answer": f"Error processing query: {str(e)[:100]}",
                "chunks_used": 0,
                "chapters": [],
                "similarity_score": 0,
                "response_time": time.time() - start_time,
                "model_used": "error"
            }
    
    def get_system_info(self) -> Dict:
        """Get system information."""
        db_stats = self.db_manager.get_system_stats()
        
        return {
            "status": "operational",
            "database": db_stats,
            "ai": {
                "model": Config.MODEL_NAME,
                "semantic_search": True,
                "embedding_model": Config.EMBEDDING_MODEL
            },
            "cache": {
                "size": len(self.cache),
                "ttl": Config.CACHE_TTL
            }
        }
    
    def get_chapters(self, limit: int = 50) -> List[str]:
        """Get available chapters."""
        return self.db_manager.get_chapter_list(limit)
    
    def clear_cache(self):
        """Clear query cache."""
        self.cache.clear()
        self.cache_timestamps.clear()
        print("✅ Cache cleared")

# ========== SINGLETON INSTANCE ==========
_rag_instance = None

def get_rag_system():
    """Get or create RAG system instance."""
    global _rag_instance
    if _rag_instance is None:
        _rag_instance = NCERTRAGSystem()
    return _rag_instance

# ========== SIMPLE TEST FUNCTION ==========
def test_system():
    """Test the RAG system."""
    print("\n🧪 Testing RAG System...")
    
    rag = get_rag_system()
    
    test_questions = [
        "What is photosynthesis?",
        "Explain chemical reactions",
        "What is Ohm's law?"
    ]
    
    for question in test_questions:
        print(f"\n🔍 Question: {question}")
        result = rag.query(question, use_cache=False)
        
        print(f"   ✅ Answer: {result['answer'][:100]}...")
        print(f"   📊 Chunks: {result['chunks_used']}, Similarity: {result['similarity_score']:.2f}")
    
    print("\n✅ System test completed!")

# ========== MAIN ENTRY ==========
if __name__ == "__main__":
    # Quick test
    test_system()