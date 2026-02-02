"""
NCERT RAG API SERVER - PRODUCTION READY v4.0
With FAISS Semantic Search & Gemini AI
"""

import os
import json
import time
import hashlib
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path

from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

# ========== CONFIGURATION ==========
load_dotenv()

class Config:
    # Database Configuration
    DB_HOST = os.getenv("DB_HOST", "db.dcmnzvjftmdbywrjkust.supabase.co")
    DB_NAME = os.getenv("DB_NAME", "postgres")
    DB_USER = os.getenv("DB_USER", "postgres")
    DB_PASSWORD = os.getenv("DB_PASSWORD", "")
    DB_PORT = os.getenv("DB_PORT", "5432")
    
    # Gemini API
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
    MODEL_NAME = os.getenv("MODEL_NAME", "gemini-2.0-flash")
    
    # Semantic Search
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    FAISS_INDEX_PATH = "faiss_index/index.faiss"
    METADATA_PATH = "faiss_index/metadata.json"
    
    # Application Settings
    CHUNK_LIMIT = int(os.getenv("CHUNK_LIMIT", "5"))
    SIMILARITY_THRESHOLD = 0.7
    CACHE_TTL = 300  # 5 minutes
    
    @classmethod
    def validate(cls):
        """Validate required environment variables."""
        if not cls.DB_PASSWORD:
            raise ValueError("DATABASE_PASSWORD environment variable is required")
        if not cls.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY environment variable is required")
        return True

# Validate configuration
try:
    Config.validate()
    print("✅ Configuration validated successfully")
except Exception as e:
    print(f"❌ Configuration error: {e}")

# ========== LAZY LOAD COMPONENTS ==========
# Import heavy dependencies only when needed
def lazy_import(module_name, import_name=None):
    """Lazy import for serverless compatibility."""
    import importlib
    if import_name is None:
        import_name = module_name
    return importlib.import_module(module_name).__getattribute__(import_name)

# Global variables for lazy loading
_psycopg2 = None
_genai = None
_faiss = None
_sentence_transformers = None
_numpy = None

def get_psycopg2():
    global _psycopg2
    if _psycopg2 is None:
        _psycopg2 = lazy_import('psycopg2')
    return _psycopg2

def get_genai():
    global _genai
    if _genai is None:
        _genai = lazy_import('google.generativeai', 'genai')
    return _genai

def get_faiss():
    global _faiss
    if _faiss is None:
        _faiss = lazy_import('faiss')
    return _faiss

def get_sentence_transformers():
    global _sentence_transformers
    if _sentence_transformers is None:
        _sentence_transformers = lazy_import('sentence_transformers', 'SentenceTransformer')
    return _sentence_transformers

def get_numpy():
    global _numpy
    if _numpy is None:
        _numpy = lazy_import('numpy', 'np')
    return _numpy

# ========== SEMANTIC SEARCH ENGINE ==========
class SemanticSearch:
    """FAISS-based semantic search."""
    
    def __init__(self):
        self.embedder = None
        self.index = None
        self.metadata = {}
        self._initialized = False
    
    def initialize(self):
        """Lazy initialization."""
        if self._initialized:
            return True
        
        try:
            print("🔧 Loading semantic search engine...")
            
            # Load embedding model
            SentenceTransformer = get_sentence_transformers()
            self.embedder = SentenceTransformer(Config.EMBEDDING_MODEL)
            
            # Load FAISS index
            faiss = get_faiss()
            index_path = Path(Config.FAISS_INDEX_PATH)
            
            if index_path.exists():
                self.index = faiss.read_index(str(index_path))
                print(f"✅ Loaded FAISS index with {self.index.ntotal} vectors")
            else:
                print("⚠️ FAISS index not found, falling back to keyword search")
                self.index = None
            
            # Load metadata
            metadata_path = Path(Config.METADATA_PATH)
            if metadata_path.exists():
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.metadata = data.get("id_to_chunk", {})
                print(f"✅ Loaded metadata for {len(self.metadata)} chunks")
            
            self._initialized = True
            return True
            
        except Exception as e:
            print(f"❌ Semantic search initialization failed: {e}")
            self._initialized = False
            return False
    
    def semantic_search(self, query: str, limit: int = 5) -> List[Dict]:
        """Perform semantic search using FAISS."""
        if not self._initialized and not self.initialize():
            return []
        
        if self.index is None:
            return []
        
        try:
            np = get_numpy()
            
            # Generate query embedding
            query_embedding = self.embedder.encode([query])
            query_embedding = np.array(query_embedding, dtype='float32')
            
            # Search in FAISS
            distances, indices = self.index.search(query_embedding, limit)
            
            # Process results
            results = []
            for idx, distance in zip(indices[0], distances[0]):
                if idx != -1 and str(idx) in self.metadata:
                    chunk_data = self.metadata[str(idx)]
                    
                    # Convert distance to similarity (0 to 1)
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
    """Database operations with connection pooling."""
    
    @staticmethod
    def get_connection():
        """Get database connection."""
        psycopg2 = get_psycopg2()
        return psycopg2.connect(
            host=Config.DB_HOST,
            database=Config.DB_NAME,
            user=Config.DB_USER,
            password=Config.DB_PASSWORD,
            port=Config.DB_PORT,
            sslmode='require',
            connect_timeout=10
        )
    
    @staticmethod
    def keyword_search(query: str, limit: int = 3) -> List[Dict]:
        """Fallback keyword search."""
        try:
            conn = DatabaseManager.get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT chapter, content, id
                FROM ncert_chunks 
                WHERE content ILIKE %s 
                ORDER BY chapter
                LIMIT %s
            """, [f"%{query}%", limit])
            
            results = []
            for row in cursor.fetchall():
                results.append({
                    "chapter": row[0],
                    "content": row[1][:500],
                    "id": row[2],
                    "similarity": 0.5  # Default for keyword search
                })
            
            conn.close()
            return results
            
        except Exception as e:
            print(f"❌ Database search error: {e}")
            return []
    
    @staticmethod
    def get_chapters(limit: int = 50) -> List[str]:
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
    def get_stats() -> Dict:
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
        self._initialized = False
    
    def initialize(self):
        """Lazy initialization of Gemini."""
        if self._initialized:
            return True
        
        try:
            genai = get_genai()
            genai.configure(api_key=Config.GEMINI_API_KEY)
            self.model = genai.GenerativeModel(Config.MODEL_NAME)
            self._initialized = True
            print(f"✅ Gemini AI initialized: {Config.MODEL_NAME}")
            return True
            
        except Exception as e:
            print(f"❌ Gemini initialization failed: {e}")
            return False
    
    def generate(self, question: str, context_chunks: List[Dict]) -> str:
        """Generate answer using context."""
        if not self._initialized and not self.initialize():
            return "AI service temporarily unavailable."
        
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
            # Fallback answer
            if context_chunks:
                return f"Based on NCERT content from {context_chunks[0].get('chapter', 'unknown chapter')}: {context_chunks[0].get('content', '')[:200]}..."
            return "I couldn't generate an answer. Please try again."

# ========== MAIN RAG SYSTEM ==========
class RAGSystem:
    """Production RAG system with semantic search."""
    
    def __init__(self):
        print("🚀 Initializing RAG System...")
        self.semantic_search = SemanticSearch()
        self.db_manager = DatabaseManager()
        self.answer_generator = AnswerGenerator()
        
        # Simple in-memory cache
        self.cache = {}
        self.cache_timestamps = {}
        
        # Statistics
        self.stats = {
            "total_queries": 0,
            "semantic_searches": 0,
            "keyword_searches": 0,
            "cache_hits": 0
        }
        
        print("✅ RAG System initialized")
    
    def _get_cache_key(self, query: str) -> str:
        """Generate cache key."""
        return hashlib.sha256(query.lower().strip().encode()).hexdigest()[:16]
    
    def query(self, question: str, use_cache: bool = True) -> Dict[str, Any]:
        """Process query with semantic search."""
        self.stats["total_queries"] += 1
        
        # Check cache
        if use_cache:
            cache_key = self._get_cache_key(question)
            if cache_key in self.cache:
                cache_time = self.cache_timestamps.get(cache_key, 0)
                if time.time() - cache_time < Config.CACHE_TTL:
                    self.stats["cache_hits"] += 1
                    result = self.cache[cache_key].copy()
                    result["cache_hit"] = True
                    return result
        
        start_time = time.time()
        
        try:
            # Step 1: Try semantic search
            search_results = self.semantic_search.semantic_search(
                question, 
                limit=Config.CHUNK_LIMIT
            )
            
            if search_results:
                self.stats["semantic_searches"] += 1
                search_type = "semantic"
            else:
                # Fallback to keyword search
                search_results = self.db_manager.keyword_search(
                    question, 
                    limit=Config.CHUNK_LIMIT
                )
                self.stats["keyword_searches"] += 1
                search_type = "keyword"
            
            if not search_results:
                return {
                    "success": False,
                    "answer": "No relevant NCERT content found for your question.",
                    "chunks_used": 0,
                    "chapters": [],
                    "similarity_score": 0,
                    "search_type": "none",
                    "cache_hit": False
                }
            
            # Step 2: Calculate average similarity
            avg_similarity = sum(r.get("similarity", 0.5) for r in search_results) / len(search_results)
            
            # Step 3: Generate answer
            answer = self.answer_generator.generate(question, search_results)
            
            # Step 4: Extract chapters
            chapters = list(set(r.get("chapter", "") for r in search_results if r.get("chapter")))
            
            # Prepare result
            result = {
                "success": True,
                "answer": answer,
                "chunks_used": len(search_results),
                "chapters": chapters[:5],  # Limit to 5
                "similarity_score": avg_similarity,
                "search_type": search_type,
                "cache_hit": False
            }
            
            # Cache result
            if use_cache:
                cache_key = self._get_cache_key(question)
                self.cache[cache_key] = result.copy()
                self.cache_timestamps[cache_key] = time.time()
            
            result["response_time"] = time.time() - start_time
            return result
            
        except Exception as e:
            print(f"❌ Query processing failed: {e}")
            return {
                "success": False,
                "answer": f"Error processing query: {str(e)[:100]}",
                "chunks_used": 0,
                "chapters": [],
                "similarity_score": 0,
                "search_type": "error",
                "cache_hit": False,
                "response_time": time.time() - start_time
            }
    
    def get_system_info(self) -> Dict:
        """Get system information."""
        db_stats = self.db_manager.get_stats()
        
        return {
            "status": "operational",
            "database": db_stats,
            "ai": {
                "model": Config.MODEL_NAME,
                "semantic_search": self.semantic_search._initialized,
                "gemini_available": self.answer_generator._initialized
            },
            "cache": {
                "size": len(self.cache),
                "hits": self.stats["cache_hits"],
                "hit_rate": f"{(self.stats['cache_hits'] / self.stats['total_queries'] * 100):.1f}%" if self.stats['total_queries'] > 0 else "0%"
            },
            "statistics": self.stats
        }
    
    def get_chapters(self, limit: int = 50) -> List[str]:
        """Get available chapters."""
        return self.db_manager.get_chapters(limit)
    
    def clear_cache(self):
        """Clear query cache."""
        self.cache.clear()
        self.cache_timestamps.clear()
        print("✅ Cache cleared")

# ========== FLASK APP SETUP ==========
app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False  # Faster responses

# Enable CORS
CORS(app)

# Initialize RAG system (lazy loading will happen on first request)
rag_system = None
STARTUP_TIME = datetime.now()

def get_rag_system():
    """Get or initialize RAG system."""
    global rag_system
    if rag_system is None:
        rag_system = RAGSystem()
    return rag_system

# ========== API ENDPOINTS ==========
@app.route('/', methods=['GET'])
def index():
    """API information endpoint."""
    return jsonify({
        "api": "NCERT RAG API",
        "version": "4.0.0",
        "description": "Semantic search powered NCERT question answering",
        "status": "operational",
        "endpoints": {
            "GET /": "API information",
            "GET /health": "Health check",
            "GET /system": "System info",
            "GET /chapters": "List chapters",
            "POST /query": "Ask questions",
            "POST /cache/clear": "Clear cache"
        },
        "timestamp": datetime.now().isoformat()
    })

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    system = get_rag_system()
    system_info = system.get_system_info()
    
    health = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "uptime": str(datetime.now() - STARTUP_TIME),
        "components": {
            "api": "operational",
            "rag_system": "initialized",
            "database": "connected" if system_info["database"].get("total_chunks", 0) > 0 else "unknown",
            "semantic_search": "available" if system_info["ai"]["semantic_search"] else "unavailable",
            "gemini_ai": "available" if system_info["ai"]["gemini_available"] else "unavailable"
        },
        "cache": system_info["cache"]
    }
    
    return jsonify(health)

@app.route('/system', methods=['GET'])
def system_info():
    """Get system information."""
    system = get_rag_system()
    info = system.get_system_info()
    info["timestamp"] = datetime.now().isoformat()
    info["uptime"] = str(datetime.now() - STARTUP_TIME)
    
    return jsonify(info)

@app.route('/chapters', methods=['GET'])
def list_chapters():
    """List available chapters."""
    try:
        limit = min(int(request.args.get('limit', 50)), 100)
        system = get_rag_system()
        chapters = system.get_chapters(limit)
        
        return jsonify({
            "success": True,
            "count": len(chapters),
            "chapters": chapters,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500

@app.route('/query', methods=['POST', 'OPTIONS'])
def query():
    """Process query."""
    # Handle CORS preflight
    if request.method == 'OPTIONS':
        return '', 200
    
    start_time = time.time()
    
    try:
        # Get JSON data
        data = request.get_json()
        if not data:
            return jsonify({
                "success": False,
                "error": "Invalid request",
                "message": "Request body must be JSON",
                "timestamp": datetime.now().isoformat()
            }), 400
        
        # Get question
        question = data.get('question', '').strip()
        
        if not question:
            return jsonify({
                "success": False,
                "error": "Invalid request",
                "message": "Question is required",
                "timestamp": datetime.now().isoformat()
            }), 400
        
        if len(question) > 500:
            return jsonify({
                "success": False,
                "error": "Invalid request",
                "message": "Question too long (max 500 characters)",
                "timestamp": datetime.now().isoformat()
            }), 400
        
        # Get optional parameters
        use_cache = data.get('use_cache', True)
        
        # Process question
        system = get_rag_system()
        result = system.query(question, use_cache=use_cache)
        
        # Add metadata
        result["question"] = question
        result["response_time"] = time.time() - start_time
        result["timestamp"] = datetime.now().isoformat()
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": "Internal server error",
            "message": str(e)[:100],
            "response_time": time.time() - start_time,
            "timestamp": datetime.now().isoformat()
        }), 500

@app.route('/cache/clear', methods=['POST'])
def clear_cache():
    """Clear cache."""
    try:
        system = get_rag_system()
        system.clear_cache()
        
        return jsonify({
            "success": True,
            "message": "Cache cleared successfully",
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500

# ========== ERROR HANDLERS ==========
@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "error": "Not Found",
        "message": "The requested endpoint does not exist",
        "timestamp": datetime.now().isoformat()
    }), 404

@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({
        "error": "Method Not Allowed",
        "timestamp": datetime.now().isoformat()
    }), 405

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        "error": "Internal Server Error",
        "timestamp": datetime.now().isoformat()
    }), 500

# ========== VERCEL COMPATIBILITY ==========
app.config['DEBUG'] = False
application = app

# ========== LOCAL DEVELOPMENT ==========
if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 NCERT RAG API SERVER v4.0")
    print("="*60)
    print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🌐 API: http://localhost:5000")
    print(f"🔧 Model: {Config.MODEL_NAME}")
    print(f"🔍 Semantic Search: Enabled")
    print("="*60)
    
    # Pre-initialize for faster first request
    get_rag_system()
    
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=False
    )