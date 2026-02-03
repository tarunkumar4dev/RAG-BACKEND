"""
NCERT RAG SYSTEM - PRODUCTION READY v5.0
Optimized for Vercel with lazy loading and error handling
"""

import os
import json
import time
import hashlib
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path
from functools import lru_cache

# ========== LOGGING SETUP ==========
logger = logging.getLogger(__name__)

# ========== LAZY IMPORTS ==========
# Lazy import for serverless compatibility and faster cold starts
class LazyImporter:
    """Lazy import manager for heavy dependencies."""
    
    def __init__(self):
        self._modules = {}
        self._available = {}
    
    def get(self, module_name, attribute_name=None):
        """Get module or attribute, importing lazily."""
        key = f"{module_name}.{attribute_name}" if attribute_name else module_name
        
        if key not in self._modules:
            try:
                import importlib
                
                # For specific modules, check availability first
                if module_name == "faiss":
                    # FAISS is optional - not required for basic functionality
                    try:
                        import faiss
                        self._modules[key] = faiss
                        self._available["faiss"] = True
                    except ImportError:
                        logger.warning("FAISS not available - semantic search will be disabled")
                        self._available["faiss"] = False
                        return None
                elif module_name == "google.generativeai":
                    module = importlib.import_module(module_name)
                    if attribute_name:
                        self._modules[key] = getattr(module, attribute_name)
                    else:
                        self._modules[key] = module
                else:
                    module = importlib.import_module(module_name)
                    if attribute_name:
                        self._modules[key] = getattr(module, attribute_name)
                    else:
                        self._modules[key] = module
                
                logger.debug(f"Lazy loaded: {key}")
                
            except ImportError as e:
                logger.error(f"Failed to import {key}: {e}")
                self._available[module_name] = False
                raise
        
        return self._modules[key]
    
    def is_available(self, module_name):
        """Check if module is available."""
        if module_name not in self._available:
            # Try to import to check availability
            try:
                self.get(module_name)
            except:
                pass
        return self._available.get(module_name, False)

# Global lazy importer
importer = LazyImporter()

# Convenience functions
def get_psycopg2():
    try:
        return importer.get('psycopg2')
    except ImportError:
        logger.error("psycopg2 is required but not installed")
        raise

def get_genai():
    try:
        return importer.get('google.generativeai', 'genai')
    except ImportError:
        logger.error("google-generativeai is required but not installed")
        raise

def get_faiss():
    if importer.is_available('faiss'):
        return importer.get('faiss')
    return None

def get_sentence_transformers():
    try:
        return importer.get('sentence_transformers', 'SentenceTransformer')
    except ImportError:
        logger.warning("sentence-transformers not available")
        return None

def get_numpy():
    try:
        return importer.get('numpy', 'np')
    except ImportError:
        logger.warning("numpy not available")
        return None

# ========== CONFIGURATION ==========
class Config:
    """Configuration with environment variable support."""
    
    # Load from environment or use defaults
    DB_HOST = os.getenv("DB_HOST", "db.dcmnzvjftmdbywrjkust.supabase.co")
    DB_NAME = os.getenv("DB_NAME", "postgres")
    DB_USER = os.getenv("DB_USER", "postgres")
    DB_PASSWORD = os.getenv("DB_PASSWORD", "")
    DB_PORT = os.getenv("DB_PORT", "5432")
    
    # Gemini AI
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
    MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
    
    # Semantic Search (optional)
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "./faiss_index/index.faiss")
    METADATA_PATH = os.getenv("METADATA_PATH", "./faiss_index/metadata.json")
    
    # Performance settings
    CHUNK_LIMIT = int(os.getenv("CHUNK_LIMIT", "3"))
    SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.6"))
    CACHE_TTL = int(os.getenv("CACHE_TTL", "300"))  # 5 minutes
    REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "30"))
    
    # Cache settings for Vercel
    CACHE_DIR = "/tmp" if os.getenv("VERCEL") else "./cache"
    
    @classmethod
    def validate(cls):
        """Validate required configuration."""
        errors = []
        
        if not cls.DB_PASSWORD:
            errors.append("DATABASE_PASSWORD not set")
        
        if not cls.GEMINI_API_KEY:
            errors.append("GEMINI_API_KEY not set")
        
        if errors:
            error_msg = ", ".join(errors)
            logger.error(f"Configuration validation failed: {error_msg}")
            return False
        
        logger.info("✅ Configuration validated successfully")
        return True
    
    @classmethod
    def get_connection_params(cls):
        """Get database connection parameters."""
        return {
            "host": cls.DB_HOST,
            "database": cls.DB_NAME,
            "user": cls.DB_USER,
            "password": cls.DB_PASSWORD,
            "port": cls.DB_PORT,
            "sslmode": "require",
            "connect_timeout": 5,
            "keepalives": 1,
            "keepalives_idle": 30,
            "keepalives_interval": 10
        }

# ========== DATABASE MANAGER ==========
class DatabaseManager:
    """Database operations with connection pooling."""
    
    @staticmethod
    @lru_cache(maxsize=1)
    def get_connection():
        """Get database connection with caching."""
        try:
            psycopg2 = get_psycopg2()
            params = Config.get_connection_params()
            conn = psycopg2.connect(**params)
            conn.autocommit = True
            logger.info("✅ Database connection established")
            return conn
        except Exception as e:
            logger.error(f"❌ Database connection failed: {e}")
            return None
    
    @staticmethod
    def execute_query(query: str, params=None):
        """Execute query safely."""
        conn = DatabaseManager.get_connection()
        if not conn:
            return None
        
        try:
            cursor = conn.cursor()
            cursor.execute(query, params or ())
            return cursor
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            return None
    
    @staticmethod
    def keyword_search(query: str, limit: int = 3) -> List[Dict]:
        """Fallback keyword search."""
        try:
            cursor = DatabaseManager.execute_query(
                """
                SELECT id, chapter, subject, class_grade, content
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
                    "id": row[0],
                    "chapter": row[1],
                    "subject": row[2],
                    "class_grade": row[3],
                    "content": row[4][:300],  # Limit content length
                    "similarity": 0.5,  # Default for keyword search
                    "source": "database"
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Keyword search failed: {e}")
            return []
    
    @staticmethod
    def get_chapter_list(limit: int = 50) -> List[str]:
        """Get list of available chapters."""
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
    def get_system_stats() -> Dict:
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

# ========== SEMANTIC SEARCH ENGINE ==========
class SemanticSearch:
    """Semantic search with FAISS (optional)."""
    
    def __init__(self):
        self.embedding_model = None
        self.faiss_index = None
        self.metadata = {}
        self.available = False
        
        # Try to initialize
        self._initialize()
    
    def _initialize(self):
        """Initialize semantic search components."""
        try:
            # Check if FAISS is available
            faiss_module = get_faiss()
            if not faiss_module:
                logger.info("Semantic search disabled: FAISS not available")
                return
            
            # Load embedding model
            SentenceTransformer = get_sentence_transformers()
            if not SentenceTransformer:
                logger.info("Semantic search disabled: SentenceTransformer not available")
                return
            
            self.embedding_model = SentenceTransformer(Config.EMBEDDING_MODEL)
            logger.info(f"✅ Loaded embedding model: {Config.EMBEDDING_MODEL}")
            
            # Load FAISS index if it exists
            faiss_path = Path(Config.FAISS_INDEX_PATH)
            if faiss_path.exists():
                try:
                    self.faiss_index = faiss_module.read_index(str(faiss_path))
                    logger.info(f"✅ Loaded FAISS index with {self.faiss_index.ntotal} vectors")
                    
                    # Load metadata
                    metadata_path = Path(Config.METADATA_PATH)
                    if metadata_path.exists():
                        with open(metadata_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            self.metadata = data.get("id_to_chunk", {})
                        logger.info(f"✅ Loaded metadata for {len(self.metadata)} chunks")
                    
                    self.available = True
                    
                except Exception as e:
                    logger.error(f"Failed to load FAISS index: {e}")
            else:
                logger.warning(f"FAISS index not found at {faiss_path}")
                
        except Exception as e:
            logger.error(f"Semantic search initialization failed: {e}")
    
    def search(self, query: str, limit: int = 3) -> List[Dict]:
        """Perform semantic search if available."""
        if not self.available or not self.faiss_index:
            return []
        
        try:
            np = get_numpy()
            if not np:
                return []
            
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
                    
                    # Convert distance to similarity
                    similarity = 1 / (1 + distance)
                    
                    if similarity >= Config.SIMILARITY_THRESHOLD:
                        results.append({
                            "id": chunk_data.get("id"),
                            "content": chunk_data.get("content", ""),
                            "chapter": chunk_data.get("chapter", ""),
                            "subject": chunk_data.get("subject", ""),
                            "class_grade": chunk_data.get("class_grade", ""),
                            "similarity": float(similarity),
                            "source": "semantic_search"
                        })
            
            return results
            
        except Exception as e:
            logger.error(f"Semantic search error: {e}")
            return []

# ========== AI ANSWER GENERATOR ==========
class AnswerGenerator:
    """Generate answers using Gemini AI with fallbacks."""
    
    def __init__(self):
        self.model = None
        self.initialized = False
        
    def initialize(self):
        """Initialize Gemini AI."""
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
            # Prepare context (limit to 2-3 chunks for token efficiency)
            context_parts = []
            for i, chunk in enumerate(context_chunks[:3], 1):
                context_parts.append(
                    f"[Source {i}]\n"
                    f"Chapter: {chunk.get('chapter', 'Unknown')}\n"
                    f"Content: {chunk.get('content', '')}\n"
                )
            
            context = "\n---\n".join(context_parts)
            
            # Create efficient prompt
            prompt = f"""You are an NCERT textbook assistant. Answer based ONLY on this context:

CONTEXT FROM NCERT:
{context}

QUESTION:
{question}

INSTRUCTIONS:
1. Answer STRICTLY based on the context above
2. Use simple, clear language
3. If information is missing, say: "This information is not covered in the provided NCERT content."
4. Keep answer concise (1-2 paragraphs)

ANSWER:"""
            
            # Generate response with timeout protection
            import threading
            
            def generate_with_timeout():
                try:
                    response = self.model.generate_content(prompt)
                    return response.text.strip()
                except Exception as e:
                    logger.error(f"Gemini generation error: {e}")
                    return None
            
            result = [None]
            thread = threading.Thread(target=lambda: result.__setitem__(0, generate_with_timeout()))
            thread.start()
            thread.join(timeout=Config.REQUEST_TIMEOUT)
            
            if thread.is_alive():
                logger.warning("Gemini API timeout")
                return "The AI response is taking too long. Please try a simpler question."
            
            answer = result[0]
            
            # Fallback if generation failed
            if not answer or len(answer) < 10:
                return f"Based on NCERT content: {context_chunks[0].get('content', '')[:150]}..."
            
            return answer
            
        except Exception as e:
            logger.error(f"Answer generation failed: {e}")
            # Smart fallback
            if context_chunks:
                return f"According to NCERT: {context_chunks[0].get('content', '')[:200]}..."
            return "I couldn't generate a detailed answer. Please try rephrasing your question."

# ========== MAIN RAG SYSTEM ==========
class NCERTRAGSystem:
    """Production RAG system with caching and fallbacks."""
    
    def __init__(self):
        logger.info("🚀 Initializing NCERT RAG System...")
        
        # Validate configuration
        if not Config.validate():
            raise ValueError("Configuration validation failed")
        
        # Initialize components
        self.db_manager = DatabaseManager()
        self.semantic_search = SemanticSearch()
        self.answer_generator = AnswerGenerator()
        
        # Simple in-memory cache with LRU-like behavior
        self.cache = {}
        self.cache_order = []
        self.max_cache_size = 100
        
        # Statistics
        self.stats = {
            "total_queries": 0,
            "semantic_searches": 0,
            "keyword_searches": 0,
            "cache_hits": 0,
            "ai_calls": 0
        }
        
        logger.info(f"✅ RAG System initialized")
        logger.info(f"   Semantic Search: {'Available' if self.semantic_search.available else 'Disabled'}")
        logger.info(f"   Gemini AI: {'Available' if self.answer_generator.initialized else 'Unavailable'}")
    
    def _get_cache_key(self, query: str) -> str:
        """Generate cache key."""
        return hashlib.sha256(query.lower().strip().encode()).hexdigest()[:16]
    
    def _add_to_cache(self, key: str, value: Dict):
        """Add item to cache with LRU-like behavior."""
        if key in self.cache:
            # Move to end (most recently used)
            self.cache_order.remove(key)
        
        self.cache[key] = value
        self.cache_order.append(key)
        
        # Remove oldest if cache is full
        if len(self.cache) > self.max_cache_size:
            oldest_key = self.cache_order.pop(0)
            del self.cache[oldest_key]
    
    def query(self, question: str, use_cache: bool = True) -> Dict[str, Any]:
        """
        Process query with semantic search and fallbacks.
        
        Args:
            question: User's question
            use_cache: Whether to use cache
            
        Returns:
            Dictionary with answer and metadata
        """
        self.stats["total_queries"] += 1
        start_time = time.time()
        
        # Validate question
        question = question.strip()
        if not question or len(question) < 3:
            return {
                "success": False,
                "answer": "Please provide a valid question (minimum 3 characters).",
                "cache_hit": False,
                "response_time": time.time() - start_time
            }
        
        # Check cache
        if use_cache:
            cache_key = self._get_cache_key(question)
            if cache_key in self.cache:
                cached = self.cache[cache_key]
                if time.time() - cached.get('_timestamp', 0) < Config.CACHE_TTL:
                    self.stats["cache_hits"] += 1
                    result = cached.copy()
                    result["cache_hit"] = True
                    result["response_time"] = time.time() - start_time
                    return result
        
        try:
            # Step 1: Try semantic search
            search_results = []
            search_type = "keyword"  # Default
            
            if self.semantic_search.available:
                search_results = self.semantic_search.search(question, limit=Config.CHUNK_LIMIT)
                if search_results:
                    self.stats["semantic_searches"] += 1
                    search_type = "semantic"
            
            # Step 2: Fallback to keyword search
            if not search_results:
                search_results = self.db_manager.keyword_search(question, limit=Config.CHUNK_LIMIT)
                self.stats["keyword_searches"] += 1
                search_type = "keyword"
            
            # Step 3: Check if we have results
            if not search_results:
                return {
                    "success": False,
                    "answer": "No relevant NCERT content found for your question.",
                    "chunks_used": 0,
                    "search_type": "none",
                    "cache_hit": False,
                    "response_time": time.time() - start_time
                }
            
            # Step 4: Generate answer
            self.stats["ai_calls"] += 1
            answer = self.answer_generator.generate(question, search_results)
            
            # Step 5: Calculate average similarity
            similarities = [r.get("similarity", 0.5) for r in search_results]
            avg_similarity = sum(similarities) / len(similarities) if similarities else 0
            
            # Step 6: Extract unique chapters
            chapters = list(set(r.get("chapter", "") for r in search_results if r.get("chapter")))
            
            # Prepare result
            result = {
                "success": True,
                "answer": answer,
                "question": question,
                "chunks_used": len(search_results),
                "chapters": chapters[:3],  # Limit to 3 chapters
                "similarity_score": round(avg_similarity, 3),
                "search_type": search_type,
                "semantic_search_available": self.semantic_search.available,
                "cache_hit": False,
                "response_time": time.time() - start_time,
                "_timestamp": time.time()  # Internal for cache
            }
            
            # Cache result
            if use_cache and result["success"]:
                cache_key = self._get_cache_key(question)
                self._add_to_cache(cache_key, result)
            
            return result
            
        except Exception as e:
            logger.error(f"Query processing error: {e}")
            return {
                "success": False,
                "answer": "An error occurred while processing your question. Please try again.",
                "chunks_used": 0,
                "search_type": "error",
                "cache_hit": False,
                "response_time": time.time() - start_time
            }
    
    def get_system_info(self) -> Dict:
        """Get comprehensive system information."""
        db_stats = self.db_manager.get_system_stats()
        
        return {
            "status": "operational",
            "timestamp": datetime.now().isoformat(),
            "database": db_stats,
            "capabilities": {
                "semantic_search": self.semantic_search.available,
                "ai_generation": self.answer_generator.initialized,
                "keyword_search": True
            },
            "configuration": {
                "model": Config.MODEL_NAME,
                "chunk_limit": Config.CHUNK_LIMIT,
                "similarity_threshold": Config.SIMILARITY_THRESHOLD,
                "cache_ttl": Config.CACHE_TTL
            },
            "statistics": self.stats,
            "cache": {
                "size": len(self.cache),
                "hit_rate": f"{(self.stats['cache_hits'] / self.stats['total_queries'] * 100 if self.stats['total_queries'] > 0 else 0):.1f}%"
            }
        }
    
    def get_chapters(self, limit: int = 50) -> List[str]:
        """Get available chapters."""
        return self.db_manager.get_chapter_list(limit)
    
    def clear_cache(self):
        """Clear query cache."""
        self.cache.clear()
        self.cache_order.clear()
        logger.info("✅ Cache cleared")
    
    def refresh_components(self):
        """Refresh system components if needed."""
        # Re-establish database connection
        DatabaseManager.get_connection.cache_clear()
        logger.info("✅ Components refreshed")

# ========== SINGLETON INSTANCE ==========
_rag_instance = None

def get_rag_system():
    """Get or create RAG system instance."""
    global _rag_instance
    if _rag_instance is None:
        try:
            _rag_instance = NCERTRAGSystem()
        except Exception as e:
            logger.error(f"Failed to initialize RAG system: {e}")
            # Create a minimal instance with error state
            class MinimalRAG:
                def query(self, question, use_cache=True):
                    return {
                        "success": False,
                        "answer": "System initialization failed. Please check configuration.",
                        "cache_hit": False,
                        "response_time": 0
                    }
                def get_system_info(self):
                    return {"status": "error", "message": "Initialization failed"}
                def get_chapters(self, limit=50):
                    return []
                def clear_cache(self):
                    pass
                def refresh_components(self):
                    pass
            
            _rag_instance = MinimalRAG()
    
    return _rag_instance

# ========== TEST FUNCTION ==========
def test_system():
    """Test the RAG system."""
    print("\n🧪 Testing RAG System...")
    
    try:
        rag = get_rag_system()
        
        # Test basic functionality
        info = rag.get_system_info()
        print(f"✅ System Status: {info.get('status', 'unknown')}")
        
        # Test a simple query
        test_questions = [
            "What is photosynthesis?",
            "Explain chemical reactions"
        ]
        
        for question in test_questions[:1]:  # Test only first to avoid rate limits
            print(f"\n🔍 Question: {question[:50]}...")
            result = rag.query(question, use_cache=False)
            
            if result["success"]:
                print(f"   ✅ Answer preview: {result['answer'][:100]}...")
                print(f"   📊 Chunks: {result['chunks_used']}, Similarity: {result['similarity_score']:.2f}")
            else:
                print(f"   ❌ Failed: {result['answer']}")
        
        print("\n✅ System test completed!")
        
    except Exception as e:
        print(f"❌ System test failed: {e}")

# ========== MAIN ENTRY ==========
if __name__ == "__main__":
    # Run test
    test_system()