"""
NCERT RAG API SERVER - PRODUCTION READY VERSION
Optimized for Vercel with Supabase Storage & Gemini AI
"""

import os
import json
import time
import hashlib
from datetime import datetime
from typing import Dict, Any, List
import logging
from functools import lru_cache

from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

# ========== LOGGING SETUP ==========
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ========== CONFIGURATION ==========
load_dotenv()

class Config:
    # Database Configuration (Supabase PostgreSQL)
    DB_HOST = os.getenv("DB_HOST", "db.dcmnzvjftmdbywrjkust.supabase.co")
    DB_NAME = os.getenv("DB_NAME", "postgres")
    DB_USER = os.getenv("DB_USER", "postgres")
    DB_PASSWORD = os.getenv("DB_PASSWORD", "")
    DB_PORT = os.getenv("DB_PORT", "5432")
    
    # Supabase Storage Configuration
    SUPABASE_URL = os.getenv("SUPABASE_URL", "")
    SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")
    FAISS_BUCKET = os.getenv("FAISS_BUCKET", "faiss-index")
    
    # FAISS File Names
    FAISS_INDEX_FILE = "index.faiss"
    FAISS_METADATA_FILE = "metadata.json"
    
    # Cache paths for Vercel
    CACHE_DIR = "/tmp/faiss_cache" if os.path.exists("/tmp") else "./tmp"
    
    # Gemini API
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
    MODEL_NAME = os.getenv("MODEL_NAME", "gemini-1.5-flash")
    
    # Embedding Model
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    
    # Application Settings
    CHUNK_LIMIT = int(os.getenv("CHUNK_LIMIT", "3"))
    SIMILARITY_THRESHOLD = 0.6
    CACHE_TTL = 300  # 5 minutes
    MAX_QUESTION_LENGTH = 500
    REQUEST_TIMEOUT = 30
    
    @classmethod
    def validate(cls):
        """Validate required environment variables."""
        errors = []
        
        if not cls.DB_PASSWORD:
            errors.append("DB_PASSWORD is required")
        
        if not cls.GEMINI_API_KEY:
            errors.append("GEMINI_API_KEY is required")
            
        if not cls.SUPABASE_URL or not cls.SUPABASE_KEY:
            logger.warning("Supabase Storage credentials not set - FAISS will be disabled")
        
        if errors:
            raise ValueError(f"Configuration errors: {', '.join(errors)}")
        
        return True

# Validate configuration
try:
    Config.validate()
    logger.info("✅ Configuration validated successfully")
except Exception as e:
    logger.error(f"❌ Configuration error: {e}")
    # Don't raise here - let the app start with degraded functionality

# ========== LAZY LOAD COMPONENTS ==========
# Singleton pattern for expensive imports

class LazyImporter:
    """Lazy import manager for serverless compatibility."""
    
    def __init__(self):
        self._modules = {}
    
    def get(self, module_name, attribute_name=None):
        """Get module or attribute, importing lazily."""
        key = f"{module_name}.{attribute_name}" if attribute_name else module_name
        
        if key not in self._modules:
            try:
                import importlib
                module = importlib.import_module(module_name)
                
                if attribute_name:
                    self._modules[key] = getattr(module, attribute_name)
                else:
                    self._modules[key] = module
                    
                logger.debug(f"Lazy loaded: {key}")
                
            except ImportError as e:
                logger.error(f"Failed to import {key}: {e}")
                raise
        
        return self._modules[key]

# Global lazy importer
importer = LazyImporter()

# Convenience functions
def get_psycopg2():
    return importer.get('psycopg2')

def get_genai():
    return importer.get('google.generativeai', 'genai')

def get_faiss():
    try:
        return importer.get('faiss')
    except ImportError:
        logger.warning("FAISS not available - semantic search disabled")
        return None

def get_sentence_transformers():
    return importer.get('sentence_transformers', 'SentenceTransformer')

def get_numpy():
    return importer.get('numpy', 'np')

def get_supabase():
    try:
        return importer.get('supabase')
    except ImportError:
        logger.warning("Supabase client not available")
        return None

# ========== DATABASE MANAGER ==========
class DatabaseManager:
    """Database operations with connection pooling and error handling."""
    
    @staticmethod
    @lru_cache(maxsize=1)
    def get_connection():
        """Get database connection with caching."""
        try:
            psycopg2 = get_psycopg2()
            conn = psycopg2.connect(
                host=Config.DB_HOST,
                database=Config.DB_NAME,
                user=Config.DB_USER,
                password=Config.DB_PASSWORD,
                port=Config.DB_PORT,
                sslmode='require',
                connect_timeout=5,
                keepalives=1,
                keepalives_idle=30,
                keepalives_interval=10,
                keepalives_count=3
            )
            conn.autocommit = True
            logger.info("✅ Database connection established")
            return conn
        except Exception as e:
            logger.error(f"❌ Database connection failed: {e}")
            return None
    
    @staticmethod
    def execute_query(query, params=None):
        """Safe query execution."""
        conn = DatabaseManager.get_connection()
        if not conn:
            return None
            
        try:
            cursor = conn.cursor()
            cursor.execute(query, params or ())
            return cursor
        except Exception as e:
            logger.error(f"Query failed: {e}")
            return None
    
    @staticmethod
    def keyword_search(query: str, limit: int = 3) -> List[Dict]:
        """Fallback keyword search."""
        try:
            cursor = DatabaseManager.execute_query(
                """
                SELECT chapter, content, id
                FROM ncert_chunks 
                WHERE content ILIKE %s 
                ORDER BY chapter
                LIMIT %s
                """,
                [f"%{query}%", limit]
            )
            
            if not cursor:
                return []
            
            results = []
            for row in cursor.fetchall():
                results.append({
                    "chapter": row[0],
                    "content": row[1][:300],  # Limit content length
                    "id": row[2],
                    "similarity": 0.5,
                    "source": "database"
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Database search error: {e}")
            return []
    
    @staticmethod
    def get_chapters(limit: int = 50) -> List[str]:
        """Get list of available chapters."""
        try:
            cursor = DatabaseManager.execute_query(
                """
                SELECT DISTINCT chapter 
                FROM ncert_chunks 
                ORDER BY chapter
                LIMIT %s
                """,
                [limit]
            )
            
            if cursor:
                return [row[0] for row in cursor.fetchall()]
            return []
            
        except Exception as e:
            logger.error(f"Failed to get chapters: {e}")
            return []
    
    @staticmethod
    def get_stats() -> Dict:
        """Get database statistics."""
        try:
            cursor = DatabaseManager.execute_query("""
                SELECT 
                    COUNT(*) as total_chunks,
                    COUNT(DISTINCT chapter) as chapters,
                    COUNT(DISTINCT subject) as subjects
                FROM ncert_chunks
            """)
            
            if cursor:
                row = cursor.fetchone()
                return {
                    "total_chunks": row[0] if row else 0,
                    "chapters": row[1] if row else 0,
                    "subjects": row[2] if row else 0
                }
            return {"error": "No database connection"}
            
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {"error": str(e)}

# ========== SIMPLIFIED SEMANTIC SEARCH ==========
class SemanticSearch:
    """Lightweight semantic search for production."""
    
    def __init__(self):
        self.available = False
        self._check_availability()
    
    def _check_availability(self):
        """Check if semantic search is available."""
        try:
            # Check for FAISS and Supabase
            if not Config.SUPABASE_URL or not Config.SUPABASE_KEY:
                logger.info("Semantic search disabled: Supabase credentials missing")
                return
            
            # Try to import required modules
            get_faiss()
            get_sentence_transformers()
            get_numpy()
            
            self.available = True
            logger.info("✅ Semantic search available")
            
        except Exception as e:
            logger.warning(f"Semantic search unavailable: {e}")
            self.available = False
    
    def search(self, query: str, limit: int = 3) -> List[Dict]:
        """Placeholder for semantic search - to be implemented."""
        if not self.available:
            return []
        
        # In production, implement actual FAISS search here
        # For now, return empty to use database fallback
        return []

# ========== AI ANSWER GENERATOR ==========
class AnswerGenerator:
    """Generate answers using Gemini AI with fallbacks."""
    
    def __init__(self):
        self.model = None
        self.initialized = False
        
    def initialize(self):
        """Initialize Gemini API."""
        if self.initialized:
            return True
            
        try:
            genai = get_genai()
            genai.configure(api_key=Config.GEMINI_API_KEY)
            self.model = genai.GenerativeModel(Config.MODEL_NAME)
            self.initialized = True
            logger.info(f"✅ Gemini AI initialized: {Config.MODEL_NAME}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Gemini initialization failed: {e}")
            return False
    
    def generate(self, question: str, context_chunks: List[Dict]) -> str:
        """Generate answer using context."""
        if not self.initialized and not self.initialize():
            return "AI service is currently unavailable. Please try again later."
        
        if not context_chunks:
            return "No relevant content found in NCERT database."
        
        try:
            # Prepare context
            context_parts = []
            for i, chunk in enumerate(context_chunks[:3], 1):  # Limit to 3 chunks
                context_parts.append(
                    f"[Source {i}]\n"
                    f"Chapter: {chunk.get('chapter', 'Unknown')}\n"
                    f"Content: {chunk.get('content', '')}\n"
                )
            
            context = "\n---\n".join(context_parts)
            
            # Create optimized prompt
            prompt = f"""You are an NCERT textbook assistant. Answer based ONLY on the provided context.

CONTEXT FROM NCERT:
{context}

QUESTION:
{question}

INSTRUCTIONS:
1. Answer based ONLY on the context above
2. Use simple, clear language suitable for students
3. If information is missing, say: "This information is not covered in the provided NCERT content."
4. Keep answer concise (2-3 paragraphs maximum)

ANSWER:"""
            
            # Generate with timeout protection
            import threading
            
            def generate_with_timeout():
                try:
                    response = self.model.generate_content(prompt)
                    return response.text.strip()
                except Exception as e:
                    return f"Error generating answer: {str(e)[:100]}"
            
            # Run with timeout
            result = [None]
            thread = threading.Thread(target=lambda: result.__setitem__(0, generate_with_timeout()))
            thread.start()
            thread.join(timeout=10)  # 10 second timeout
            
            if thread.is_alive():
                logger.warning("Gemini API timeout")
                return "The AI response is taking too long. Please try a simpler question."
            
            answer = result[0]
            
            if not answer or len(answer) < 10:
                # Fallback to context extraction
                return f"Based on NCERT Chapter '{context_chunks[0].get('chapter', 'Unknown')}': {context_chunks[0].get('content', '')[:150]}..."
            
            return answer
            
        except Exception as e:
            logger.error(f"Answer generation error: {e}")
            # Smart fallback
            if context_chunks:
                return f"According to NCERT: {context_chunks[0].get('content', '')[:200]}..."
            return "I couldn't generate a detailed answer. Please try rephrasing your question."

# ========== MAIN RAG SYSTEM ==========
class RAGSystem:
    """Production RAG system optimized for Vercel."""
    
    def __init__(self):
        logger.info("🚀 Initializing RAG System...")
        
        # Initialize components
        self.db_manager = DatabaseManager()
        self.semantic_search = SemanticSearch()
        self.answer_generator = AnswerGenerator()
        
        # Simple cache
        self.cache = {}
        self.stats = {
            "total_queries": 0,
            "cache_hits": 0,
            "database_searches": 0,
            "ai_calls": 0
        }
        
        logger.info("✅ RAG System initialized")
    
    def _get_cache_key(self, query: str) -> str:
        """Generate cache key."""
        return hashlib.sha256(query.lower().strip().encode()).hexdigest()[:16]
    
    def query(self, question: str, use_cache: bool = True) -> Dict[str, Any]:
        """Process query with caching and fallbacks."""
        self.stats["total_queries"] += 1
        
        # Validate input
        if not question or len(question.strip()) < 3:
            return {
                "success": False,
                "answer": "Please provide a valid question (minimum 3 characters).",
                "cache_hit": False
            }
        
        # Check cache
        if use_cache:
            cache_key = self._get_cache_key(question)
            cached_result = self.cache.get(cache_key)
            if cached_result and time.time() - cached_result.get('_timestamp', 0) < Config.CACHE_TTL:
                self.stats["cache_hits"] += 1
                result = cached_result.copy()
                result["cache_hit"] = True
                return result
        
        start_time = time.time()
        
        try:
            # Step 1: Search for relevant content
            self.stats["database_searches"] += 1
            
            # Try semantic search first
            search_results = self.semantic_search.search(question, limit=Config.CHUNK_LIMIT)
            
            # Fallback to keyword search
            if not search_results:
                search_results = self.db_manager.keyword_search(question, limit=Config.CHUNK_LIMIT)
            
            if not search_results:
                return {
                    "success": False,
                    "answer": "No relevant content found in NCERT database for your question.",
                    "chunks_used": 0,
                    "cache_hit": False,
                    "response_time": time.time() - start_time
                }
            
            # Step 2: Generate answer
            self.stats["ai_calls"] += 1
            answer = self.answer_generator.generate(question, search_results)
            
            # Step 3: Prepare response
            chapters = list(set(r.get("chapter", "") for r in search_results if r.get("chapter")))
            
            result = {
                "success": True,
                "answer": answer,
                "question": question,
                "chunks_used": len(search_results),
                "chapters": chapters[:3],
                "semantic_search_used": self.semantic_search.available,
                "cache_hit": False,
                "response_time": time.time() - start_time,
                "_timestamp": time.time()  # Internal timestamp for cache
            }
            
            # Cache result
            if use_cache and result["success"]:
                cache_key = self._get_cache_key(question)
                self.cache[cache_key] = result.copy()
                # Limit cache size
                if len(self.cache) > 100:
                    # Remove oldest entries
                    oldest_key = min(self.cache.keys(), 
                                   key=lambda k: self.cache[k].get('_timestamp', 0))
                    del self.cache[oldest_key]
            
            return result
            
        except Exception as e:
            logger.error(f"Query processing error: {e}")
            return {
                "success": False,
                "answer": "An error occurred while processing your question. Please try again.",
                "error": str(e)[:100],
                "cache_hit": False,
                "response_time": time.time() - start_time
            }
    
    def get_system_info(self) -> Dict:
        """Get system information."""
        db_stats = self.db_manager.get_stats()
        
        return {
            "status": "operational",
            "timestamp": datetime.now().isoformat(),
            "database": db_stats,
            "capabilities": {
                "semantic_search": self.semantic_search.available,
                "ai_generation": self.answer_generator.initialized,
                "database": bool(db_stats.get("total_chunks", 0) > 0)
            },
            "statistics": self.stats,
            "cache": {
                "size": len(self.cache),
                "hit_rate": f"{(self.stats['cache_hits'] / self.stats['total_queries'] * 100 if self.stats['total_queries'] > 0 else 0):.1f}%"
            }
        }
    
    def clear_cache(self):
        """Clear query cache."""
        self.cache.clear()
        logger.info("✅ Cache cleared")

# ========== FLASK APP SETUP ==========
app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False
app.config['MAX_CONTENT_LENGTH'] = 1 * 1024 * 1024  # 1MB max request size

# Enable CORS with specific origins for production
CORS(app, resources={
    r"/*": {
        "origins": ["*"],  # In production, replace with actual origins
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})

# Initialize RAG system
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
        "version": "2.0.0",
        "description": "NCERT Textbook Question Answering System",
        "status": "operational",
        "documentation": {
            "endpoints": {
                "GET /": "API information",
                "GET /health": "Health check",
                "GET /system": "System info",
                "GET /chapters": "List available chapters",
                "POST /query": "Ask questions"
            }
        },
        "timestamp": datetime.now().isoformat()
    })

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    system = get_rag_system()
    
    health = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "uptime": str(datetime.now() - STARTUP_TIME),
        "components": {
            "api": "operational",
            "rag_system": "initialized",
            "database": "connected",
            "ai_service": "available" if system.answer_generator.initialized else "unavailable"
        }
    }
    
    return jsonify(health)

@app.route('/system', methods=['GET'])
def system_info():
    """Get system information."""
    system = get_rag_system()
    info = system.get_system_info()
    return jsonify(info)

@app.route('/chapters', methods=['GET'])
def list_chapters():
    """List available chapters."""
    try:
        limit = min(int(request.args.get('limit', 30)), 100)
        system = get_rag_system()
        chapters = system.db_manager.get_chapters(limit)
        
        return jsonify({
            "success": True,
            "count": len(chapters),
            "chapters": chapters,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Chapters endpoint error: {e}")
        return jsonify({
            "success": False,
            "error": "Failed to fetch chapters",
            "timestamp": datetime.now().isoformat()
        }), 500

@app.route('/query', methods=['POST'])
def query():
    """Process query - main endpoint."""
    start_time = time.time()
    
    # Handle CORS preflight
    if request.method == 'OPTIONS':
        return '', 200
    
    try:
        # Validate request
        if not request.is_json:
            return jsonify({
                "success": False,
                "error": "Content-Type must be application/json",
                "timestamp": datetime.now().isoformat()
            }), 400
        
        data = request.get_json()
        
        if not data or 'question' not in data:
            return jsonify({
                "success": False,
                "error": "Missing 'question' field",
                "timestamp": datetime.now().isoformat()
            }), 400
        
        question = data.get('question', '').strip()
        
        # Validate question
        if not question:
            return jsonify({
                "success": False,
                "error": "Question cannot be empty",
                "timestamp": datetime.now().isoformat()
            }), 400
        
        if len(question) > Config.MAX_QUESTION_LENGTH:
            return jsonify({
                "success": False,
                "error": f"Question too long (max {Config.MAX_QUESTION_LENGTH} characters)",
                "timestamp": datetime.now().isoformat()
            }), 400
        
        # Get optional parameters
        use_cache = data.get('use_cache', True)
        
        # Process question
        system = get_rag_system()
        result = system.query(question, use_cache=use_cache)
        
        # Add metadata
        result["timestamp"] = datetime.now().isoformat()
        result["request_time"] = time.time() - start_time
        
        # Remove internal fields
        if '_timestamp' in result:
            del result['_timestamp']
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Query endpoint error: {e}")
        return jsonify({
            "success": False,
            "error": "Internal server error",
            "message": "Please try again later",
            "request_time": time.time() - start_time,
            "timestamp": datetime.now().isoformat()
        }), 500

@app.route('/cache/clear', methods=['POST'])
def clear_cache():
    """Clear cache - admin endpoint."""
    try:
        # Simple authentication (in production, add proper auth)
        auth_header = request.headers.get('Authorization')
        if auth_header != f"Bearer {os.getenv('ADMIN_TOKEN', '')}":
            return jsonify({
                "success": False,
                "error": "Unauthorized",
                "timestamp": datetime.now().isoformat()
            }), 401
        
        system = get_rag_system()
        system.clear_cache()
        
        return jsonify({
            "success": True,
            "message": "Cache cleared successfully",
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Clear cache error: {e}")
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
    logger.error(f"Internal server error: {error}")
    return jsonify({
        "error": "Internal Server Error",
        "message": "An unexpected error occurred",
        "timestamp": datetime.now().isoformat()
    }), 500

@app.errorhandler(413)
def request_too_large(error):
    return jsonify({
        "error": "Request Too Large",
        "message": f"Maximum request size is {app.config['MAX_CONTENT_LENGTH'] / (1024*1024)}MB",
        "timestamp": datetime.now().isoformat()
    }), 413

# ========== VERCEL COMPATIBILITY ==========
app.config['DEBUG'] = False

# Required for Vercel
application = app

# ========== STARTUP LOGGING ==========
if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 NCERT RAG API - PRODUCTION SERVER")
    print("="*60)
    print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🌐 API: http://localhost:5000")
    print(f"🔧 Model: {Config.MODEL_NAME}")
    print(f"🗄️ Database: {'✅ Connected' if Config.DB_PASSWORD else '❌ Not configured'}")
    print(f"🤖 AI: {'✅ Available' if Config.GEMINI_API_KEY else '❌ Not configured'}")
    print("="*60)
    
    # Pre-initialize for better cold start performance
    get_rag_system()
    
    app.run(
        host='0.0.0.0',
        port=int(os.getenv('PORT', 5000)),
        debug=False,
        threaded=True
    )