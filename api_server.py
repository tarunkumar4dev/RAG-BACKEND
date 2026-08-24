"""
NCERT RAG API SERVER - PRODUCTION READY v5.0
Fully corrected with bug fixes for Vercel deployment
"""

import os
import json
import time
import hashlib
import logging
import base64
from datetime import datetime
from typing import Dict, Any, List
from functools import lru_cache

from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from module_service import ModuleService

# ========== LOGGING SETUP ==========
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ========== CONFIGURATION ==========
load_dotenv()

class Config:
    # Database Configuration
    DB_HOST = os.getenv("DB_HOST", "db.dcmnzvjftmdbywrjkust.supabase.co")
    DB_NAME = os.getenv("DB_NAME", "postgres")
    DB_USER = os.getenv("DB_USER", "postgres")
    DB_PASSWORD = os.getenv("DB_PASSWORD", "")
    DB_PORT = os.getenv("DB_PORT", "5432")
    
    # Gemini AI
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
    MODEL_NAME = os.getenv("MODEL_NAME", "gemini-1.5-flash")
    
    # Application Settings
    CHUNK_LIMIT = int(os.getenv("CHUNK_LIMIT", "3"))
    SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.5"))
    CACHE_TTL = int(os.getenv("CACHE_TTL", "300"))  # 5 minutes
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
        
        if errors:
            raise ValueError(f"Configuration errors: {', '.join(errors)}")
        
        logger.info("✅ Configuration validated successfully")
        return True

# Validate configuration
try:
    Config.validate()
    CONFIG_VALID = True
except Exception as e:
    logger.error(f"❌ Configuration error: {e}")
    CONFIG_VALID = False

# ========== LAZY LOAD COMPONENTS ==========
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

importer = LazyImporter()

def get_psycopg2():
    return importer.get('psycopg2')

def get_genai():
    return importer.get('google.generativeai', 'genai')

# ========== DATABASE MANAGER ==========
class DatabaseManager:
    """Database operations with proper column mapping."""
    
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
    def execute_query(query: str, params=None):
        """Safe query execution."""
        conn = DatabaseManager.get_connection()
        if not conn:
            logger.error("No database connection available")
            return None
        
        try:
            cursor = conn.cursor()
            cursor.execute(query, params or ())
            return cursor
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            # Try to re-establish connection
            DatabaseManager.get_connection.cache_clear()
            return None
    
    @staticmethod
    def keyword_search(query: str, limit: int = 3) -> List[Dict]:
        """Keyword search - FIXED COLUMN MAPPING."""
        try:
            logger.info(f"Searching for: '{query}'")
            
            # FIXED: Use correct column names from your table
            cursor = DatabaseManager.execute_query(
                """
                SELECT id, chapter, subject, class_grade, content
                FROM ncert_chunks 
                WHERE content ILIKE %s 
                ORDER BY id
                LIMIT %s
                """,
                [f"%{query}%", limit]
            )
            
            if not cursor:
                logger.warning("No cursor returned from query")
                return []
            
            results = []
            rows = cursor.fetchall()
            logger.info(f"Found {len(rows)} matching rows")
            
            for row in rows:
                results.append({
                    "id": row[0],
                    "chapter": row[1] or "Unknown",
                    "subject": row[2] or "Unknown",
                    "class_grade": row[3] or "Unknown",
                    "content": row[4][:500] if row[4] else "",  # Limit content length
                    "similarity": 0.7,  # Higher confidence for exact matches
                    "source": "database"
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Keyword search error: {e}")
            return []
    
    @staticmethod
    def get_chapters(limit: int = 50) -> List[str]:
        """Get list of available chapters - FIXED."""
        try:
            cursor = DatabaseManager.execute_query(
                """
                SELECT DISTINCT chapter 
                FROM ncert_chunks 
                WHERE chapter IS NOT NULL AND chapter != ''
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
        """Get database statistics - FIXED."""
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
                if row:
                    return {
                        "total_chunks": row[0],
                        "chapters": row[1],
                        "subjects": row[2],
                        "status": "connected"
                    }
            
            return {"status": "no_data", "total_chunks": 0}
            
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {"status": "error", "error": str(e)}

# ========== AI ANSWER GENERATOR ==========
class AnswerGenerator:
    """Generate answers using Gemini AI with proper initialization."""
    
    def __init__(self):
        self.model = None
        self.initialized = False
        self.initialize()  # Initialize immediately
    
    def initialize(self) -> bool:
        """Initialize Gemini API."""
        if self.initialized:
            return True
        
        if not Config.GEMINI_API_KEY:
            logger.error("GEMINI_API_KEY not configured")
            return False
        
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
        if not self.initialized:
            logger.error("Gemini not initialized")
            return "AI service is currently unavailable."
        
        if not context_chunks:
            return "No relevant content found in NCERT database."
        
        try:
            # Prepare context (limit to 3 chunks)
            context_parts = []
            for i, chunk in enumerate(context_chunks[:3], 1):
                context_parts.append(
                    f"[Source {i}]\n"
                    f"Chapter: {chunk.get('chapter', 'Unknown')}\n"
                    f"Subject: {chunk.get('subject', 'Unknown')}\n"
                    f"Class: {chunk.get('class_grade', 'Unknown')}\n"
                    f"Content: {chunk.get('content', '')}\n"
                )
            
            context = "\n---\n".join(context_parts)
            
            # Create optimized prompt
            prompt = f"""You are an expert NCERT textbook assistant. Answer based ONLY on the provided context.

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
            answer = response.text.strip()
            
            if not answer or len(answer) < 10:
                # Fallback
                return f"Based on NCERT content: {context_chunks[0].get('content', '')[:200]}..."
            
            return answer
            
        except Exception as e:
            logger.error(f"Answer generation error: {e}")
            # Smart fallback
            if context_chunks:
                return f"According to NCERT Chapter '{context_chunks[0].get('chapter', 'Unknown')}': {context_chunks[0].get('content', '')[:200]}..."
            return "I couldn't generate a detailed answer. Please try rephrasing your question."

# ========== MAIN RAG SYSTEM ==========
class RAGSystem:
    """Production RAG system with proper initialization."""
    
    def __init__(self):
        logger.info("🚀 Initializing RAG System...")
        
        if not CONFIG_VALID:
            raise ValueError("Configuration validation failed")
        
        # Initialize components
        self.db_manager = DatabaseManager()
        self.answer_generator = AnswerGenerator()
        
        # Initialize immediately
        self._initialize_components()
        
        # Simple cache
        self.cache = {}
        self.stats = {
            "total_queries": 0,
            "cache_hits": 0,
            "database_searches": 0,
            "ai_calls": 0,
            "successful_queries": 0,
            "failed_queries": 0
        }
        
        logger.info("✅ RAG System initialized")
    
    def _initialize_components(self):
        """Initialize all components with proper error handling."""
        # Test database connection
        try:
            conn = self.db_manager.get_connection()
            if conn:
                logger.info("✅ Database component ready")
            else:
                logger.error("❌ Database component failed")
        except Exception as e:
            logger.error(f"Database initialization error: {e}")
        
        # Test AI
        if self.answer_generator.initialized:
            logger.info("✅ AI component ready")
        else:
            logger.error("❌ AI component failed")
    
    def _get_cache_key(self, query: str) -> str:
        """Generate cache key."""
        return hashlib.sha256(query.lower().strip().encode()).hexdigest()[:16]
    
    def query(self, question: str, use_cache: bool = True) -> Dict[str, Any]:
        """Process query with proper error handling."""
        self.stats["total_queries"] += 1
        start_time = time.time()
        
        # Validate input
        question = question.strip()
        if not question or len(question) < 3:
            self.stats["failed_queries"] += 1
            return {
                "success": False,
                "answer": "Please provide a valid question (minimum 3 characters).",
                "chunks_used": 0,
                "cache_hit": False,
                "response_time": time.time() - start_time
            }
        
        # Check cache
        if use_cache:
            cache_key = self._get_cache_key(question)
            cached = self.cache.get(cache_key)
            if cached and time.time() - cached.get('_timestamp', 0) < Config.CACHE_TTL:
                self.stats["cache_hits"] += 1
                result = cached.copy()
                result["cache_hit"] = True
                result["response_time"] = time.time() - start_time
                return result
        
        try:
            # Step 1: Search for relevant content
            self.stats["database_searches"] += 1
            search_results = self.db_manager.keyword_search(question, limit=Config.CHUNK_LIMIT)
            
            logger.info(f"Search returned {len(search_results)} results")
            
            if not search_results:
                self.stats["failed_queries"] += 1
                return {
                    "success": False,
                    "answer": "No relevant NCERT content found for your question.",
                    "chunks_used": 0,
                    "cache_hit": False,
                    "response_time": time.time() - start_time
                }
            
            # Step 2: Generate answer
            self.stats["ai_calls"] += 1
            answer = self.answer_generator.generate(question, search_results)
            
            # Step 3: Calculate similarity
            similarities = [r.get("similarity", 0.5) for r in search_results]
            avg_similarity = sum(similarities) / len(similarities)
            
            # Step 4: Extract chapters
            chapters = list(set(r.get("chapter") for r in search_results if r.get("chapter")))
            
            # Prepare result
            result = {
                "success": True,
                "answer": answer,
                "question": question,
                "chunks_used": len(search_results),
                "chapters": chapters[:3],
                "similarity_score": round(avg_similarity, 3),
                "cache_hit": False,
                "response_time": time.time() - start_time,
                "_timestamp": time.time()  # Internal for cache
            }
            
            # Cache result
            if use_cache:
                cache_key = self._get_cache_key(question)
                self.cache[cache_key] = result.copy()
                # Limit cache size
                if len(self.cache) > 100:
                    # Simple LRU: remove first item
                    first_key = next(iter(self.cache))
                    del self.cache[first_key]
            
            self.stats["successful_queries"] += 1
            return result
            
        except Exception as e:
            logger.error(f"Query processing error: {e}")
            self.stats["failed_queries"] += 1
            return {
                "success": False,
                "answer": "An error occurred while processing your question. Please try again.",
                "chunks_used": 0,
                "cache_hit": False,
                "response_time": time.time() - start_time
            }
    
    def get_system_info(self) -> Dict:
        """Get comprehensive system information."""
        db_stats = self.db_manager.get_stats()
        
        success_rate = 0
        if self.stats["total_queries"] > 0:
            success_rate = (self.stats["successful_queries"] / self.stats["total_queries"]) * 100
        
        return {
            "status": "operational",
            "timestamp": datetime.now().isoformat(),
            "database": db_stats,
            "ai": {
                "initialized": self.answer_generator.initialized,
                "model": Config.MODEL_NAME
            },
            "statistics": {
                "total_queries": self.stats["total_queries"],
                "successful_queries": self.stats["successful_queries"],
                "failed_queries": self.stats["failed_queries"],
                "success_rate": f"{success_rate:.1f}%",
                "database_searches": self.stats["database_searches"],
                "ai_calls": self.stats["ai_calls"],
                "cache_hits": self.stats["cache_hits"]
            },
            "cache": {
                "size": len(self.cache),
                "ttl_seconds": Config.CACHE_TTL
            },
            "configuration": {
                "chunk_limit": Config.CHUNK_LIMIT,
                "similarity_threshold": Config.SIMILARITY_THRESHOLD
            }
        }
    
    def get_chapters(self, limit: int = 50) -> List[str]:
        """Get available chapters."""
        return self.db_manager.get_chapters(limit)
    
    def clear_cache(self):
        """Clear query cache."""
        self.cache.clear()
        logger.info("✅ Cache cleared")

# ========== FLASK APP SETUP ==========
app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False
app.config['MAX_CONTENT_LENGTH'] = 1 * 1024 * 1024  # 1MB

# Enable CORS
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

# Initialize RAG system
rag_system = None
STARTUP_TIME = datetime.now()

def get_rag_system():
    """Get or initialize RAG system with error handling."""
    global rag_system
    if rag_system is None:
        try:
            rag_system = RAGSystem()
            logger.info("✅ RAG System initialized successfully")
        except Exception as e:
            logger.error(f"❌ Failed to initialize RAG system: {e}")
            # Create minimal fallback system
            class FallbackSystem:
                def query(self, question, use_cache=True):
                    return {
                        "success": False,
                        "answer": "System initialization failed. Please check server logs.",
                        "chunks_used": 0,
                        "cache_hit": False,
                        "response_time": 0
                    }
                def get_system_info(self):
                    return {"status": "error", "message": "Initialization failed"}
                def get_chapters(self, limit=50):
                    return []
                def clear_cache(self):
                    pass
            
            rag_system = FallbackSystem()
    
    return rag_system

# ========== HELPER: Extract teacher_id from JWT ==========
def get_teacher_id_from_request():
    """Extract teacher_id from Authorization header JWT or request body/args."""
    # Try from request body
    if request.is_json:
        data = request.get_json(silent=True)
        if data and data.get("teacher_id"):
            return data["teacher_id"].strip()
    
    # Try from query params
    if request.args.get("teacher_id"):
        return request.args.get("teacher_id").strip()
    
    # Try from Authorization header (decode JWT)
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        try:
            token = auth_header.replace("Bearer ", "")
            payload = token.split(".")[1]
            # Add padding for base64
            payload += "=" * (4 - len(payload) % 4)
            decoded = json.loads(base64.b64decode(payload))
            return decoded.get("sub", "")
        except Exception:
            pass
    
    return None

# ========== API ENDPOINTS ==========
@app.route('/', methods=['GET'])
def index():
    """API information endpoint."""
    return jsonify({
        "api": "NCERT RAG API",
        "version": "5.0.0",
        "description": "NCERT Textbook Question Answering System - Bug Fixed Version",
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
    
    # Get actual database status
    db_stats = system.db_manager.get_stats() if hasattr(system, 'db_manager') else {"status": "unknown"}
    
    health = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "uptime": str(datetime.now() - STARTUP_TIME),
        "components": {
            "api": "operational",
            "rag_system": "initialized",
            "database": db_stats.get("status", "unknown"),
            "ai_service": "available" if hasattr(system, 'answer_generator') and system.answer_generator.initialized else "unavailable"
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
        chapters = system.get_chapters(limit)
        
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

# ========== MODULE ENDPOINTS ==========

@app.route('/modules/create', methods=['POST'])
def create_module():
    """Create a new module after frontend uploads PDF to Supabase Storage."""
    try:
        if not request.is_json:
            return jsonify({"success": False, "error": "Content-Type must be application/json"}), 400

        data = request.get_json()

        # Required fields
        storage_path = data.get("storage_path", "").strip()
        original_filename = data.get("original_filename", "").strip()
        subject = data.get("subject", "").strip()
        class_level = data.get("class_level", "").strip() or data.get("class", "").strip()

        if not all([storage_path, original_filename, subject, class_level]):
            return jsonify({
                "success": False,
                "error": "Missing required fields: storage_path, original_filename, subject, class_level"
            }), 400

        # Get teacher_id
        teacher_id = get_teacher_id_from_request()
        if not teacher_id:
            return jsonify({"success": False, "error": "teacher_id required (in body or Authorization header)"}), 400

        # Optional fields
        file_type = data.get("file_type", "pdf")
        file_size_bytes = data.get("file_size_bytes")
        institute_id = data.get("institute_id")

        module_id, error = ModuleService.create_module(
            teacher_id=teacher_id,
            storage_path=storage_path,
            original_filename=original_filename,
            subject=subject,
            class_level=class_level,
            file_type=file_type,
            file_size_bytes=file_size_bytes,
            institute_id=institute_id,
        )

        if error:
            return jsonify({"success": False, "error": error}), 500

        return jsonify({
            "success": True,
            "module_id": module_id,
            "status": "processing",
            "message": "Module created. Call /modules/process to start processing."
        }), 201

    except Exception as e:
        logger.error(f"Create module error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/modules/process', methods=['POST'])
def process_module():
    """Process a module — extract text, summarize, chunk. This is the heavy endpoint."""
    try:
        if not request.is_json:
            return jsonify({"success": False, "error": "Content-Type must be application/json"}), 400

        data = request.get_json()
        module_id = data.get("module_id", "").strip()

        if not module_id:
            return jsonify({"success": False, "error": "module_id required"}), 400

        result = ModuleService.process_module(module_id)

        if result.get("success"):
            return jsonify(result)
        else:
            return jsonify(result), 500

    except Exception as e:
        logger.error(f"Process module error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/modules/list', methods=['GET'])
def list_modules():
    """List teacher's modules."""
    try:
        teacher_id = get_teacher_id_from_request()
        if not teacher_id:
            return jsonify({"success": False, "error": "teacher_id required"}), 400

        subject = request.args.get("subject")
        class_level = request.args.get("class")

        modules = ModuleService.list_modules(teacher_id, subject, class_level)

        return jsonify({
            "success": True,
            "count": len(modules),
            "modules": modules,
        })

    except Exception as e:
        logger.error(f"List modules error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/modules/<module_id>', methods=['GET'])
def get_module(module_id):
    """Get module details with summary."""
    try:
        teacher_id = get_teacher_id_from_request()
        if not teacher_id:
            return jsonify({"success": False, "error": "teacher_id required as query param"}), 400

        module = ModuleService.get_module(module_id, teacher_id)
        if not module:
            return jsonify({"success": False, "error": "Module not found"}), 404

        return jsonify({"success": True, **module})

    except Exception as e:
        logger.error(f"Get module error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/modules/<module_id>', methods=['DELETE'])
def delete_module(module_id):
    """Delete a module."""
    try:
        teacher_id = get_teacher_id_from_request()
        if not teacher_id:
            return jsonify({"success": False, "error": "teacher_id required"}), 400

        success, error = ModuleService.delete_module(module_id, teacher_id)
        if success:
            return jsonify({"success": True, "message": "Module deleted"})
        else:
            return jsonify({"success": False, "error": error}), 404

    except Exception as e:
        logger.error(f"Delete module error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/modules/<module_id>/generate-test', methods=['POST'])
def generate_test_from_module(module_id):
    """Generate test paper from module content."""
    try:
        if not request.is_json:
            return jsonify({"success": False, "error": "Content-Type must be application/json"}), 400

        data = request.get_json()

        teacher_id = get_teacher_id_from_request()
        if not teacher_id:
            return jsonify({"success": False, "error": "teacher_id required"}), 400

        num_questions = data.get("num_questions", 10)
        question_types = data.get("question_types")
        difficulty = data.get("difficulty", "medium")
        topics = data.get("topics")

        test_data, error = ModuleService.generate_test_from_module(
            module_id=module_id,
            teacher_id=teacher_id,
            num_questions=num_questions,
            question_types=question_types,
            difficulty=difficulty,
            topics=topics,
        )

        if error:
            return jsonify({"success": False, "error": error}), 500

        return jsonify({"success": True, "test": test_data})

    except Exception as e:
        logger.error(f"Module test generation error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

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

# ========== VERCEL COMPATIBILITY ==========
app.config['DEBUG'] = False
application = app

# ========== STARTUP ==========
if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 NCERT RAG API - PRODUCTION SERVER v5.0")
    print("="*60)
    print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🌐 API: http://localhost:5000")
    
    # Initialize system
    system = get_rag_system()
    
    # Show status
    if hasattr(system, 'get_system_info'):
        info = system.get_system_info()
        print(f"🗄️ Database: {info.get('database', {}).get('status', 'unknown')}")
        print(f"🤖 AI: {'✅ Available' if info.get('ai', {}).get('initialized') else '❌ Unavailable'}")
        print(f"📊 Total chunks: {info.get('database', {}).get('total_chunks', 0)}")
    
    print("="*60)
    
    app.run(
        host='0.0.0.0',
        port=int(os.getenv('PORT', 5000)),
        debug=False,
        threaded=True
    )