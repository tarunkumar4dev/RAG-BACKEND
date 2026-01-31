"""
NCERT SMART RAG SYSTEM - PRODUCTION READY v3.0
FAISS VERSION - No Pinecone/ChromaDB
"""

import os
import sys
import logging
import time
import json
import hashlib
import threading
import uuid
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple, Optional
from functools import lru_cache
import numpy as np
import psycopg2
from psycopg2.extras import RealDictCursor, DictCursor
from psycopg2.pool import SimpleConnectionPool
import google.generativeai as genai
import faiss  # ✅ FAISS instead of Pinecone/ChromaDB

# ========== WINDOWS UNICODE FIX ==========
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# ========== GLOBAL CONFIGURATION ==========
CONFIG = {
    # Database
    "db_host": "db.dcmnzvjftmdbywrjkust.supabase.co",
    "db_port": 5432,
    "db_name": "postgres",
    "db_user": "postgres",
    "connection_timeout": 30,
    "query_timeout": 10,
    
    # Connection Pool
    "pool_minconn": 2,
    "pool_maxconn": 20,
    
    # Performance
    "max_retries": 3,
    "retry_delay": 1,
    "max_chunks_per_query": 15,
    "cache_size": 1000,
    "cache_ttl": 300,  # 5 minutes
    "chunk_limit": 10,
    
    # Gemini
    "gemini_temperature": 0.2,
    "gemini_max_tokens": 800,
    "gemini_timeout": 30,
    
    # FAISS Vector Database (NEW)
    "faiss_index_path": "./faiss_index",
    "embedding_dimension": 768,
    "faiss_nprobe": 10,  # Search parameters
    "faiss_nlist": 100,
    
    # Logging
    "log_rotation": "daily",
    "log_level": "INFO",
    
    # System
    "max_query_length": 1000,
    "min_chunk_length": 50,
    "max_chunk_length": 500,
}

# ========== ADVANCED LOGGING SETUP ==========
class ColorFormatter(logging.Formatter):
    """Color formatter for console output."""
    
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[41m',  # Red background
        'RESET': '\033[0m'
    }
    
    def format(self, record):
        log_color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        record.levelname = f"{log_color}{record.levelname}{self.COLORS['RESET']}"
        record.msg = f"{log_color}{record.msg}{self.COLORS['RESET']}"
        return super().format(record)

def setup_logging():
    """Configure advanced production logging."""
    
    # Create logs directory
    log_dir = Path(__file__).parent / "logs"
    log_dir.mkdir(exist_ok=True)
    
    # Get root logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # ========== CONSOLE HANDLER ==========
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Color formatter for console
    console_format = ColorFormatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_format)
    
    # ========== FILE HANDLER ==========
    log_file = log_dir / f"rag_system_{datetime.now().strftime('%Y%m%d')}.log"
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    
    # Detailed formatter for files
    file_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_format)
    
    # ========== ERROR HANDLER ==========
    error_file = log_dir / f"errors_{datetime.now().strftime('%Y%m%d')}.log"
    error_handler = logging.FileHandler(error_file, encoding='utf-8')
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(file_format)
    
    # Add all handlers
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    logger.addHandler(error_handler)
    
    return logger

logger = setup_logging()

# ========== FAISS VECTOR STORE MANAGER ==========
class FAISSVectorStore:
    """Local FAISS vector store for NCERT chunks."""
    
    def __init__(self, index_path: str, dimension: int = 768):
        self.index_path = Path(index_path)
        self.dimension = dimension
        self.index = None
        self.id_to_chunk = {}  # Map FAISS IDs to chunk data
        self.next_id = 0
        self.lock = threading.Lock()
        self.load_or_create()
    
    def load_or_create(self):
        """Load existing index or create new one."""
        try:
            if (self.index_path / "index.faiss").exists():
                self.index = faiss.read_index(str(self.index_path / "index.faiss"))
                
                # Load metadata
                metadata_file = self.index_path / "metadata.json"
                if metadata_file.exists():
                    with open(metadata_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        self.id_to_chunk = data.get("id_to_chunk", {})
                        self.next_id = data.get("next_id", 0)
                
                logger.info(f"✅ FAISS index loaded: {self.index.ntotal} vectors")
            else:
                # Create new index
                self.index = faiss.IndexFlatIP(self.dimension)  # Inner product similarity
                self.index_path.mkdir(exist_ok=True, parents=True)
                logger.info(f"✅ Created new FAISS index (dimension: {self.dimension})")
                
        except Exception as e:
            logger.error(f"❌ FAISS load failed: {e}")
            # Create fresh index
            self.index = faiss.IndexFlatIP(self.dimension)
            self.index_path.mkdir(exist_ok=True, parents=True)
    
    def add_chunks(self, chunks: List[Dict]):
        """Add chunks to FAISS index."""
        if not chunks:
            return
        
        with self.lock:
            # Create embeddings (simplified - using keyword frequency)
            embeddings = []
            new_chunk_data = {}
            
            for chunk in chunks:
                # Simple embedding: keyword frequency vector
                embedding = self._create_simple_embedding(chunk.get('content', ''))
                embeddings.append(embedding)
                
                # Store chunk data
                self.id_to_chunk[self.next_id] = {
                    'id': chunk.get('id'),
                    'content': chunk.get('content', '')[:500],
                    'class_grade': chunk.get('class_grade'),
                    'subject': chunk.get('subject'),
                    'chapter': chunk.get('chapter')
                }
                self.next_id += 1
            
            # Add to index
            if embeddings:
                embeddings_np = np.array(embeddings).astype('float32')
                self.index.add(embeddings_np)
                
                # Save index
                self.save()
                
                logger.info(f"✅ Added {len(chunks)} chunks to FAISS index")
    
    def _create_simple_embedding(self, text: str) -> List[float]:
        """Create simple embedding using keyword frequencies."""
        # Common educational keywords
        edu_keywords = [
            'science', 'physics', 'chemistry', 'biology', 'math', 'mathematics',
            'history', 'geography', 'english', 'hindi', 'sanskrit', 'social',
            'experiment', 'theory', 'formula', 'equation', 'process', 'system',
            'energy', 'force', 'matter', 'cell', 'organism', 'plant', 'animal',
            'chemical', 'reaction', 'acid', 'base', 'metal', 'nonmetal',
            'electric', 'current', 'circuit', 'magnet', 'light', 'sound'
        ]
        
        # Initialize vector
        embedding = [0.0] * self.dimension
        
        # Use first N dimensions for keyword frequencies
        text_lower = text.lower()
        for i, keyword in enumerate(edu_keywords[:min(self.dimension, len(edu_keywords))]):
            embedding[i] = text_lower.count(keyword) / max(1, len(text_lower.split()))
        
        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = [x / norm for x in embedding]
        
        return embedding
    
    def search(self, query: str, k: int = 10) -> List[Dict]:
        """Search for similar chunks."""
        try:
            # Create query embedding
            query_embedding = self._create_simple_embedding(query)
            query_vector = np.array([query_embedding]).astype('float32')
            
            # Search
            distances, indices = self.index.search(query_vector, min(k, self.index.ntotal))
            
            # Retrieve chunks
            results = []
            for i, idx in enumerate(indices[0]):
                if idx != -1 and idx in self.id_to_chunk:
                    chunk_data = self.id_to_chunk[idx]
                    results.append({
                        **chunk_data,
                        'similarity': float(distances[0][i]) if distances[0][i] > 0 else 0.5,
                        'faiss_id': int(idx)
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"❌ FAISS search failed: {e}")
            return []
    
    def save(self):
        """Save index and metadata."""
        try:
            faiss.write_index(self.index, str(self.index_path / "index.faiss"))
            
            metadata = {
                "id_to_chunk": self.id_to_chunk,
                "next_id": self.next_id,
                "saved_at": datetime.now().isoformat(),
                "total_vectors": self.index.ntotal
            }
            
            with open(self.index_path / "metadata.json", 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2)
                
            logger.debug(f"✅ FAISS index saved: {self.index.ntotal} vectors")
            
        except Exception as e:
            logger.error(f"❌ FAISS save failed: {e}")
    
    def get_stats(self) -> Dict:
        """Get FAISS statistics."""
        return {
            "total_vectors": self.index.ntotal if self.index else 0,
            "dimension": self.dimension,
            "chunks_stored": len(self.id_to_chunk),
            "index_path": str(self.index_path)
        }

# ========== CONNECTION POOL MANAGER ==========
class DatabasePool:
    """Advanced PostgreSQL connection pool with monitoring."""
    
    _instance = None
    _lock = threading.Lock()
    _pool = None
    _stats = {
        "connections_created": 0,
        "connections_used": 0,
        "connection_errors": 0,
        "last_reset": datetime.now()
    }
    
    @classmethod
    def get_instance(cls):
        """Singleton instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance
    
    def __init__(self):
        self._initialize_pool()
    
    def _initialize_pool(self):
        """Initialize connection pool."""
        try:
            password = os.getenv("DATABASE_PASSWORD", "").strip()
            if not password:
                raise ValueError("DATABASE_PASSWORD not set")
            
            self._pool = SimpleConnectionPool(
                minconn=CONFIG["pool_minconn"],
                maxconn=CONFIG["pool_maxconn"],
                host=CONFIG["db_host"],
                port=CONFIG["db_port"],
                user=CONFIG["db_user"],
                password=password,
                database=CONFIG["db_name"],
                sslmode='require',
                connect_timeout=CONFIG["connection_timeout"],
                keepalives=1,
                keepalives_idle=30,
                keepalives_interval=10,
                keepalives_count=5
            )
            
            logger.info(f"✅ Database pool initialized: {CONFIG['pool_minconn']}-{CONFIG['pool_maxconn']} connections")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize database pool: {e}")
            raise
    
    def get_connection(self):
        """Get connection from pool with retry logic."""
        for attempt in range(CONFIG["max_retries"]):
            try:
                conn = self._pool.getconn()
                self._stats["connections_used"] += 1
                
                # Test connection is alive
                with conn.cursor() as cursor:
                    cursor.execute("SELECT 1")
                
                return conn
                
            except Exception as e:
                self._stats["connection_errors"] += 1
                if attempt < CONFIG["max_retries"] - 1:
                    logger.warning(f"⚠️ Connection attempt {attempt + 1} failed: {e}, retrying...")
                    time.sleep(CONFIG["retry_delay"])
                else:
                    logger.error(f"❌ All connection attempts failed")
                    raise
        
        raise Exception("Failed to get database connection")
    
    def return_connection(self, conn):
        """Return connection to pool."""
        try:
            if conn:
                conn.rollback()  # Ensure clean state
                self._pool.putconn(conn)
        except Exception as e:
            logger.error(f"Error returning connection: {e}")
            try:
                conn.close()
            except:
                pass
    
    def get_stats(self):
        """Get pool statistics."""
        return {
            **self._stats,
            "pool_size": self._pool.maxconn if self._pool else 0,
            "available": self._pool.maxconn - len(self._pool._used) if self._pool else 0,
            "uptime": str(datetime.now() - self._stats["last_reset"])
        }
    
    def close(self):
        """Close all connections."""
        if self._pool:
            self._pool.closeall()
            logger.info("✅ Database pool closed")

# ========== INTELLIGENT QUERY CACHE ==========
class SmartCache:
    """Intelligent caching with TTL, LRU, and semantic similarity."""
    
    def __init__(self, max_size=1000, ttl=300):
        self.cache = {}
        self.max_size = max_size
        self.ttl = ttl
        self.access_times = {}
        self.lock = threading.Lock()
        self.hits = 0
        self.misses = 0
    
    def _get_cache_key(self, query: str) -> str:
        """Generate cache key with semantic normalization."""
        # Normalize query
        normalized = query.lower().strip()
        
        # Remove extra whitespace
        normalized = ' '.join(normalized.split())
        
        # Generate hash
        return hashlib.sha256(normalized.encode()).hexdigest()[:16]
    
    def get(self, query: str):
        """Get cached result for query."""
        key = self._get_cache_key(query)
        
        with self.lock:
            if key in self.cache:
                timestamp, result = self.cache[key]
                
                # Check TTL
                if time.time() - timestamp <= self.ttl:
                    self.hits += 1
                    self.access_times[key] = time.time()
                    return result
                else:
                    # Expired
                    del self.cache[key]
                    if key in self.access_times:
                        del self.access_times[key]
        
        self.misses += 1
        return None
    
    def set(self, query: str, result):
        """Cache result for query."""
        key = self._get_cache_key(query)
        timestamp = time.time()
        
        with self.lock:
            # Cleanup if needed
            if len(self.cache) >= self.max_size:
                self._evict_oldest()
            
            self.cache[key] = (timestamp, result)
            self.access_times[key] = timestamp
    
    def _evict_oldest(self):
        """Evict least recently used entries."""
        if not self.access_times:
            return
        
        # Get oldest accessed keys
        sorted_keys = sorted(self.access_times.items(), key=lambda x: x[1])
        keys_to_remove = sorted_keys[:max(1, len(sorted_keys) // 10)]  # Remove 10%
        
        for key, _ in keys_to_remove:
            if key in self.cache:
                del self.cache[key]
            if key in self.access_times:
                del self.access_times[key]
    
    def clear(self):
        """Clear cache."""
        with self.lock:
            self.cache.clear()
            self.access_times.clear()
            self.hits = 0
            self.misses = 0
    
    def stats(self):
        """Get cache statistics."""
        with self.lock:
            hit_rate = (self.hits / (self.hits + self.misses) * 100) if (self.hits + self.misses) > 0 else 0
            return {
                "size": len(self.cache),
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": f"{hit_rate:.1f}%",
                "max_size": self.max_size,
                "ttl": self.ttl
            }

# ========== GEMINI AI MANAGER ==========
class GeminiManager:
    """Manage Gemini AI with fallback strategies."""
    
    def __init__(self):
        self.models = []
        self.current_model = None
        self.fallback_mode = False
        self.initialize()
    
    def initialize(self):
        """Initialize Gemini with multiple model fallbacks."""
        try:
            api_key = os.getenv("GEMINI_API_KEY", "").strip()
            if not api_key:
                logger.warning("⚠️ GEMINI_API_KEY not set, using fallback mode")
                self.fallback_mode = True
                return
            
            # Configure Gemini
            genai.configure(api_key=api_key)
            
            # Try models in order of preference
            model_priority = [
                "gemini-2.0-flash",
                "gemini-1.5-flash", 
                "gemini-1.5-pro",
                "gemini-pro"
            ]
            
            for model_name in model_priority:
                try:
                    logger.info(f"🔧 Testing model: {model_name}")
                    
                    # Quick test
                    model = genai.GenerativeModel(model_name)
                    response = model.generate_content(
                        "Hello", 
                        generation_config={
                            "max_output_tokens": 1,
                            "temperature": 0.1
                        }
                    )
                    
                    if response.text:
                        self.current_model = model_name
                        logger.info(f"✅ Using Gemini model: {self.current_model}")
                        return
                        
                except Exception as e:
                    logger.debug(f"Model {model_name} failed: {str(e)[:100]}")
                    continue
            
            logger.warning("⚠️ No Gemini model available, using fallback mode")
            self.fallback_mode = True
            
        except Exception as e:
            logger.error(f"❌ Gemini initialization failed: {e}")
            self.fallback_mode = True
    
    def generate_answer(self, prompt: str, context: str) -> str:
        """Generate answer using Gemini."""
        if self.fallback_mode or not self.current_model:
            raise Exception("Gemini not available")
        
        try:
            model = genai.GenerativeModel(self.current_model)
            
            # Enhanced prompt engineering
            enhanced_prompt = f"""# NCERT Expert Assistant
You are an expert NCERT tutor. Answer the student's question using ONLY the provided NCERT content.

## NCERT CONTEXT:
{context}

## STUDENT'S QUESTION:
{prompt}

## INSTRUCTIONS:
1. Answer STRICTLY based on the NCERT content above
2. Use simple, student-friendly language
3. Be concise but complete
4. If information is not in context, say: "This specific information is not available in the NCERT textbook."
5. Do NOT add any external knowledge
6. Reference the class and subject if applicable
7. Structure answer logically

## ANSWER:"""
            
            response = model.generate_content(
                enhanced_prompt,
                generation_config={
                    "temperature": CONFIG["gemini_temperature"],
                    "max_output_tokens": CONFIG["gemini_max_tokens"],
                    "top_p": 0.8,
                    "top_k": 40
                },
                safety_settings={
                    "HARM_CATEGORY_HARASSMENT": "BLOCK_NONE",
                    "HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE",
                    "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE",
                    "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE"
                }
            )
            
            if response.prompt_feedback and response.prompt_feedback.block_reason:
                logger.warning(f"⚠️ Response blocked: {response.prompt_feedback}")
                raise Exception("Response blocked by safety settings")
            
            return response.text.strip()
            
        except Exception as e:
            logger.error(f"❌ Gemini generation failed: {e}")
            raise
    
    def is_available(self) -> bool:
        """Check if Gemini is available."""
        return not self.fallback_mode and self.current_model is not None

# ========== HYBRID CHUNK RETRIEVER ==========
class HybridRetriever:
    """Hybrid retrieval using both database and FAISS."""
    
    def __init__(self, db_pool: DatabasePool, faiss_store: FAISSVectorStore):
        self.db_pool = db_pool
        self.faiss_store = faiss_store
    
    def retrieve(self, query: str, limit: int = 10) -> List[Dict]:
        """Retrieve chunks using hybrid approach."""
        all_chunks = []
        seen_ids = set()
        
        # ===== STRATEGY 1: FAISS Semantic Search =====
        faiss_results = self.faiss_store.search(query, k=limit * 2)
        for result in faiss_results:
            chunk_id = result.get('id')
            if chunk_id and chunk_id not in seen_ids:
                seen_ids.add(chunk_id)
                all_chunks.append(result)
                if len(all_chunks) >= limit:
                    break
        
        logger.debug(f"FAISS found {len(faiss_results)} results, using {len(all_chunks)}")
        
        # ===== STRATEGY 2: Database Keyword Fallback =====
        if len(all_chunks) < limit:
            remaining = limit - len(all_chunks)
            db_chunks = self._database_keyword_search(query, remaining)
            
            for chunk in db_chunks:
                chunk_id = chunk.get('id')
                if chunk_id and chunk_id not in seen_ids:
                    seen_ids.add(chunk_id)
                    all_chunks.append(chunk)
                    
                    # Also add to FAISS for future searches
                    self.faiss_store.add_chunks([chunk])
        
        # Sort by similarity
        all_chunks.sort(key=lambda x: x.get('similarity', 0), reverse=True)
        
        return all_chunks[:limit]
    
    def _database_keyword_search(self, query: str, limit: int = 10) -> List[Dict]:
        """Fallback to database keyword search."""
        conn = None
        try:
            conn = self.db_pool.get_connection()
            
            # Extract simple keywords
            words = [w.lower() for w in query.split() if len(w) > 3]
            if not words:
                return []
            
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                # Build keyword conditions
                keyword_conditions = []
                params = []
                
                for word in words[:5]:  # Use first 5 words
                    keyword_conditions.append("content ILIKE %s")
                    params.append(f"%{word}%")
                
                where_clause = " OR ".join(keyword_conditions)
                
                cursor.execute(f"""
                    SELECT id, class_grade, subject, chapter, content
                    FROM ncert_chunks 
                    WHERE {where_clause}
                    ORDER BY LENGTH(content) DESC
                    LIMIT %s
                """, params + [limit * 2])
                
                rows = cursor.fetchall()
                
                # Convert to dicts and add similarity score
                chunks = []
                for row in rows:
                    chunk = dict(row)
                    # Calculate simple similarity based on keyword matches
                    content_lower = chunk['content'].lower()
                    matches = sum(1 for word in words if word in content_lower)
                    chunk['similarity'] = min(matches / len(words), 1.0)
                    chunks.append(chunk)
                
                return chunks
                
        except Exception as e:
            logger.error(f"❌ Database keyword search failed: {e}")
            return []
        finally:
            if conn:
                self.db_pool.return_connection(conn)
    
    def preload_faiss(self, batch_size: int = 1000):
        """Preload database chunks into FAISS."""
        logger.info("🔄 Preloading chunks into FAISS...")
        
        conn = None
        try:
            conn = self.db_pool.get_connection()
            
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute("SELECT COUNT(*) as total FROM ncert_chunks")
                total = cursor.fetchone()['total']
                logger.info(f"📊 Total chunks in database: {total}")
                
                offset = 0
                loaded = 0
                
                while True:
                    cursor.execute("""
                        SELECT id, class_grade, subject, chapter, content
                        FROM ncert_chunks 
                        ORDER BY id
                        LIMIT %s OFFSET %s
                    """, [batch_size, offset])
                    
                    rows = cursor.fetchall()
                    if not rows:
                        break
                    
                    chunks = [dict(row) for row in rows]
                    self.faiss_store.add_chunks(chunks)
                    
                    loaded += len(chunks)
                    offset += batch_size
                    
                    if loaded % 5000 == 0:
                        logger.info(f"✅ Loaded {loaded}/{total} chunks into FAISS")
                
                logger.info(f"🎉 FAISS preload complete: {loaded} chunks")
                
        except Exception as e:
            logger.error(f"❌ FAISS preload failed: {e}")
        finally:
            if conn:
                self.db_pool.return_connection(conn)

# ========== MAIN RAG SYSTEM ==========
class NCERTRAGSystem:
    """Production-ready NCERT RAG System with FAISS."""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        """Singleton pattern."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(NCERTRAGSystem, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.initialized = False
            self.db_pool = None
            self.faiss_store = None
            self.cache = None
            self.gemini = None
            self.retriever = None
            self.stats = {
                "total_queries": 0,
                "gemini_queries": 0,
                "fallback_queries": 0,
                "cache_hits": 0,
                "faiss_searches": 0,
                "total_response_time": 0,
                "start_time": datetime.now()
            }
            self._initialize()
    
    def _initialize(self):
        """Initialize all components."""
        try:
            logger.info("🚀 Initializing NCERT Smart RAG System v3.0 (FAISS)...")
            
            # Load environment
            self._load_environment()
            
            # Initialize components
            self.db_pool = DatabasePool.get_instance()
            self.faiss_store = FAISSVectorStore(
                index_path=CONFIG["faiss_index_path"],
                dimension=CONFIG["embedding_dimension"]
            )
            self.cache = SmartCache(
                max_size=CONFIG["cache_size"],
                ttl=CONFIG["cache_ttl"]
            )
            self.gemini = GeminiManager()
            self.retriever = HybridRetriever(self.db_pool, self.faiss_store)
            
            # Preload FAISS in background
            threading.Thread(target=self.retriever.preload_faiss, daemon=True).start()
            
            # Verify system
            self._verify_system()
            
            self.initialized = True
            logger.info("✅ NCERT RAG System (FAISS) initialized successfully!")
            
        except Exception as e:
            logger.error(f"❌ Initialization failed: {e}", exc_info=True)
            raise
    
    def _load_environment(self):
        """Load and validate environment variables."""
        env_path = Path(__file__).parent / '.env'
        
        if env_path.exists():
            logger.info("📁 Loading environment from .env file")
            with open(env_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and '=' in line and not line.startswith('#'):
                        key, value = line.split('=', 1)
                        os.environ[key.strip()] = value.strip().strip('"').strip("'")
        
        # Validate required variables
        required = ["DATABASE_PASSWORD"]
        for var in required:
            if not os.getenv(var):
                logger.warning(f"⚠️ Required variable not set: {var}")
    
    def _verify_system(self):
        """Verify all system components are working."""
        logger.info("🔍 Verifying system components...")
        
        # Test database
        conn = None
        try:
            conn = self.db_pool.get_connection()
            with conn.cursor() as cursor:
                cursor.execute("SELECT COUNT(*) FROM ncert_chunks")
                count = cursor.fetchone()[0]
                logger.info(f"✓ Database: {count} chunks")
                
                cursor.execute("SELECT COUNT(DISTINCT chapter) FROM ncert_chunks")
                chapters = cursor.fetchone()[0]
                logger.info(f"✓ Chapters: {chapters}")
        finally:
            if conn:
                self.db_pool.return_connection(conn)
        
        # Test FAISS
        faiss_stats = self.faiss_store.get_stats()
        logger.info(f"✓ FAISS: {faiss_stats['total_vectors']} vectors")
        
        # Test Gemini
        if self.gemini.is_available():
            logger.info(f"✓ Gemini: {self.gemini.current_model}")
        else:
            logger.info("✓ Gemini: Fallback mode")
        
        # Test cache
        cache_stats = self.cache.stats()
        logger.info(f"✓ Cache: {cache_stats['size']}/{cache_stats['max_size']} entries")
    
    def query(self, question: str, use_cache: bool = True) -> Dict[str, Any]:
        """
        Process a query and return detailed response.
        
        Args:
            question: User's question
            use_cache: Whether to use cache
        
        Returns:
            Dictionary with answer and metadata
        """
        if not self.initialized:
            raise Exception("System not initialized")
        
        start_time = time.time()
        self.stats["total_queries"] += 1
        
        # Generate query ID
        query_id = str(uuid.uuid4())[:8]
        
        # Check cache
        if use_cache:
            cached_result = self.cache.get(question)
            if cached_result:
                self.stats["cache_hits"] += 1
                response_time = time.time() - start_time
                self.stats["total_response_time"] += response_time
                
                logger.info(f"✅ Query [{query_id}] from cache in {response_time:.2f}s")
                
                return {
                    "query_id": query_id,
                    "answer": cached_result["answer"],
                    "chunks_used": cached_result["chunks_used"],
                    "chapters": cached_result.get("chapters", []),
                    "cache_hit": True,
                    "response_time": response_time,
                    "model_used": cached_result.get("model_used", "cache"),
                    "similarity_score": cached_result.get("similarity_score", 0)
                }
        
        try:
            # Retrieve chunks
            chunks = self.retriever.retrieve(question, limit=CONFIG["chunk_limit"])
            
            if not chunks:
                answer = "I couldn't find relevant information in my NCERT knowledge base."
                chunks_count = 0
                chapters = []
                model_used = "none"
                avg_similarity = 0
            else:
                # Extract chapters
                chapters = list(set(chunk.get('chapter', 'Unknown') for chunk in chunks if chunk.get('chapter')))
                
                # Calculate average similarity
                avg_similarity = sum(chunk.get('similarity', 0) for chunk in chunks) / len(chunks)
                
                # Generate answer
                try:
                    if self.gemini.is_available():
                        # Prepare context for Gemini
                        context = self._prepare_context(chunks)
                        answer = self.gemini.generate_answer(question, context)
                        model_used = self.gemini.current_model
                        self.stats["gemini_queries"] += 1
                    else:
                        raise Exception("Gemini not available")
                        
                except Exception as e:
                    # Fallback to simple concatenation
                    answer = self._generate_fallback_answer(chunks)
                    model_used = "fallback"
                    self.stats["fallback_queries"] += 1
                    logger.warning(f"⚠️ Using fallback answer: {str(e)[:100]}")
                
                chunks_count = len(chunks)
            
            response_time = time.time() - start_time
            self.stats["total_response_time"] += response_time
            
            # Prepare result
            result = {
                "query_id": query_id,
                "answer": answer,
                "chunks_used": chunks_count,
                "chapters": chapters[:5],  # Limit to 5 chapters
                "cache_hit": False,
                "response_time": response_time,
                "model_used": model_used,
                "similarity_score": avg_similarity
            }
            
            # Cache the result
            if use_cache and chunks_count > 0:
                self.cache.set(question, result)
            
            logger.info(f"✅ Query [{query_id}] processed in {response_time:.2f}s, using {chunks_count} chunks")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Query failed: {e}", exc_info=True)
            return {
                "query_id": query_id,
                "answer": "I encountered an error while processing your question. Please try again.",
                "chunks_used": 0,
                "chapters": [],
                "cache_hit": False,
                "response_time": time.time() - start_time,
                "model_used": "error",
                "similarity_score": 0
            }
    
    def _prepare_context(self, chunks: List[Dict]) -> str:
        """Prepare context for Gemini."""
        context_parts = []
        
        for i, chunk in enumerate(chunks[:5], 1):  # Use top 5 chunks
            context_parts.append(
                f"[SOURCE {i}]\n"
                f"Class: {chunk.get('class_grade', 'Unknown')}\n"
                f"Subject: {chunk.get('subject', 'Unknown')}\n"
                f"Chapter: {chunk.get('chapter', 'Unknown')}\n"
                f"Content: {chunk['content']}\n"
            )
        
        return "\n" + "="*60 + "\n".join(context_parts) + "="*60
    
    def _generate_fallback_answer(self, chunks: List[Dict]) -> str:
        """Generate fallback answer when Gemini fails."""
        if not chunks:
            return "I couldn't find relevant information to answer this question."
        
        # Group by chapter
        grouped = {}
        for chunk in chunks[:5]:
            chapter = chunk.get('chapter', 'Unknown')
            if chapter not in grouped:
                grouped[chapter] = []
            grouped[chapter].append(chunk)
        
        # Build answer
        answer_parts = ["Based on NCERT content:\n"]
        
        for i, (chapter, chunk_list) in enumerate(grouped.items(), 1):
            chunk = chunk_list[0]
            answer_parts.append(
                f"{i}. **{chapter}**\n"
                f"   Subject: {chunk.get('subject', 'Unknown')}\n"
                f"   Class: {chunk.get('class_grade', 'Unknown')}\n"
                f"   {chunk['content'][:200].strip()}"
            )
            if len(chunk['content']) > 200:
                answer_parts[-1] += "..."
            answer_parts[-1] += "\n"
        
        answer_parts.append("\n*Refer to NCERT textbook for complete information.*")
        
        return "\n".join(answer_parts)
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics."""
        # Database stats
        conn = None
        db_stats = {}
        try:
            conn = self.db_pool.get_connection()
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute("""
                    SELECT 
                        COUNT(*) as total_chunks,
                        COUNT(DISTINCT chapter) as unique_chapters,
                        COUNT(DISTINCT subject) as unique_subjects,
                        COUNT(DISTINCT class_grade) as unique_classes
                    FROM ncert_chunks
                """)
                db_stats = cursor.fetchone()
        except Exception as e:
            logger.error(f"Failed to get DB stats: {e}")
        finally:
            if conn:
                self.db_pool.return_connection(conn)
        
        # FAISS stats
        faiss_stats = self.faiss_store.get_stats()
        
        # Calculate averages
        avg_response_time = 0
        if self.stats["total_queries"] > 0:
            avg_response_time = self.stats["total_response_time"] / self.stats["total_queries"]
        
        cache_stats = self.cache.stats()
        
        return {
            "system": {
                "version": "3.0.0",
                "uptime": str(datetime.now() - self.stats["start_time"]),
                "status": "operational" if self.initialized else "error",
                "initialized": self.initialized
            },
            "database": db_stats,
            "faiss": faiss_stats,
            "ai": {
                "gemini_available": self.gemini.is_available(),
                "current_model": self.gemini.current_model,
                "gemini_queries": self.stats["gemini_queries"],
                "fallback_queries": self.stats["fallback_queries"]
            },
            "performance": {
                "total_queries": self.stats["total_queries"],
                "cache_hits": self.stats["cache_hits"],
                "cache_hit_rate": cache_stats["hit_rate"],
                "avg_response_time": f"{avg_response_time:.2f}s",
                "cache_size": cache_stats["size"]
            },
            "configuration": {
                "max_chunks": CONFIG["chunk_limit"],
                "cache_ttl": CONFIG["cache_ttl"],
                "pool_size": CONFIG["pool_maxconn"]
            }
        }
    
    def clear_cache(self):
        """Clear query cache."""
        self.cache.clear()
        logger.info("✅ Cache cleared")
    
    def close(self):
        """Clean shutdown."""
        # Save FAISS index
        self.faiss_store.save()
        
        if self.db_pool:
            self.db_pool.close()
        logger.info("✅ System shut down cleanly")

# ========== INTERACTIVE MODE ==========
class InteractiveMode:
    """Interactive question-answering interface."""
    
    def __init__(self):
        self.rag = NCERTRAGSystem()
        self.running = True
    
    def display_banner(self):
        """Display system banner."""
        print("\n" + "="*60)
        print("🚀 NCERT SMART RAG SYSTEM - FAISS EDITION v3.0")
        print("="*60)
        
        stats = self.rag.get_system_stats()
        db_stats = stats["database"]
        faiss_stats = stats["faiss"]
        
        print("\n📈 SYSTEM STATS:")
        print(f"   • Database: {db_stats.get('total_chunks', 0)} NCERT chunks")
        print(f"   • FAISS: {faiss_stats.get('total_vectors', 0)} vectors")
        print(f"   • Gemini: {'✅ Available' if stats['ai']['gemini_available'] else '❌ Disabled'}")
        print(f"   • Cache: {stats['performance']['cache_size']} entries ({CONFIG['cache_ttl']}s TTL)")
        
        print("\n💬 COMMANDS:")
        print("  Type 'stats' for system info")
        print("  Type 'cache' to clear cache")
        print("  Type 'test' to run tests")
        print("  Type 'exit' to quit")
        print("-" * 60)
    
    def handle_command(self, command: str):
        """Handle special commands."""
        command = command.strip().lower()
        
        if command == 'stats':
            stats = self.rag.get_system_stats()
            print("\n📊 DETAILED SYSTEM INFO:")
            print(f"   • System: {stats['system']['version']} ({stats['system']['status']})")
            print(f"   • Uptime: {stats['system']['uptime']}")
            print(f"   • Total Queries: {stats['performance']['total_queries']}")
            print(f"   • Cache Hit Rate: {stats['performance']['cache_hit_rate']}")
            print(f"   • Avg Response Time: {stats['performance']['avg_response_time']}")
            print(f"   • Gemini Model: {stats['ai']['current_model'] or 'Not available'}")
            print(f"   • FAISS Vectors: {stats['faiss']['total_vectors']}")
            return True
        
        elif command == 'cache':
            self.rag.clear_cache()
            print("\n✅ Cache cleared successfully!")
            return True
        
        elif command == 'test':
            self.run_quick_test()
            return True
        
        elif command in ['exit', 'quit']:
            print("\n👋 Goodbye!")
            self.running = False
            return True
        
        return False
    
    def run_quick_test(self):
        """Run quick system test."""
        print("\n🧪 Running quick test...")
        
        test_questions = [
            "What is photosynthesis?",
            "Explain chemical reactions"
        ]
        
        for question in test_questions:
            print(f"\n   Testing: '{question[:30]}...'")
            start_time = time.time()
            result = self.rag.query(question, use_cache=False)
            response_time = time.time() - start_time
            
            print(f"   ✅ {result['chunks_used']} chunks, {result['model_used']}, {response_time:.2f}s")
        
        print("\n✅ Test completed!")
    
    def format_answer(self, result: Dict[str, Any]):
        """Format answer for display."""
        print(f"\n📚 Answer:")
        print(f"{result['answer']}\n")
        
        print("📊 Sources:")
        print(f"   • Chunks used: {result['chunks_used']}")
        
        if result['chapters']:
            chapters_str = ', '.join(result['chapters'])
            print(f"   • Chapters: {chapters_str[:80]}...")
        
        print(f"   • Avg relevance: {result['similarity_score']*100:.1f}%")
        print(f"   • Response time: {result['response_time']:.2f}s")
        
        model_name = result['model_used']
        if 'gemini' in model_name:
            print(f"   • Powered by: Gemini AI")
        elif model_name == 'cache':
            print(f"   • Served from: Cache")
        else:
            print(f"   • Source: NCERT Database + FAISS")
    
    def run(self):
        """Run interactive mode."""
        self.display_banner()
        
        while self.running:
            try:
                # Get user input
                user_input = input("\n❓ Question: ").strip()
                
                if not user_input:
                    continue
                
                # Check for commands
                if self.handle_command(user_input):
                    continue
                
                # Process question
                print("\n🤔 Processing...")
                
                # Get answer
                result = self.rag.query(user_input)
                
                # Display answer
                self.format_answer(result)
                
            except KeyboardInterrupt:
                print("\n\n⚠️ Type 'exit' to quit.")
                continue
            
            except Exception as e:
                print(f"\n❌ Error: {e}")
                logger.error(f"Interactive mode error: {e}")
                continue

# ========== MAIN ENTRY POINT ==========
def main():
    """Main entry point."""
    # Display startup banner
    print("\n" + "="*60)
    print("🚀 NCERT SMART RAG SYSTEM v3.0 - FAISS EDITION")
    print("="*60)
    print("Initializing...\n")
    
    try:
        # Start interactive mode
        interactive = InteractiveMode()
        interactive.run()
        
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        print(f"\n❌ System error: {e}")
    
    finally:
        # Ensure clean shutdown
        rag = NCERTRAGSystem._instance
        if rag:
            rag.close()
        print("\n✅ System shut down cleanly")

# ========== DIRECT TEST FUNCTION ==========
def test_direct_question(question: str):
    """Test a direct question."""
    print(f"\n🔍 Testing: {question}")
    print("-" * 60)
    
    rag = NCERTRAGSystem()
    
    try:
        result = rag.query(question, use_cache=False)
        
        print(f"\n📚 Answer:")
        print(result['answer'])
        
        print(f"\n📊 Stats:")
        print(f"   • Chunks used: {result['chunks_used']}")
        print(f"   • Response time: {result['response_time']:.2f}s")
        print(f"   • Model: {result['model_used']}")
        print(f"   • Similarity: {result['similarity_score']*100:.1f}%")
        
    finally:
        rag.close()

# ========== ENTRY POINTS ==========
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="NCERT RAG System")
    parser.add_argument('--mode', choices=['interactive', 'test', 'question'], 
                       default='interactive', help='Run mode')
    parser.add_argument('--question', type=str, help='Direct question to answer')
    
    args = parser.parse_args()
    
    if args.mode == 'question' and args.question:
        test_direct_question(args.question)
    elif args.mode == 'test':
        # Run test mode
        rag = NCERTRAGSystem()
        try:
            test_questions = [
                "What is photosynthesis?",
                "Explain chemical reactions",
                "What are acids and bases?",
                "Describe metals and non-metals",
                "What is electricity?"
            ]
            
            for question in test_questions:
                test_direct_question(question)
                print("\n" + "="*60)
                
        finally:
            rag.close()
    else:
        # Default to interactive mode
        main()