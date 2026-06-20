"""
🚀 NCERT ULTRA PRO - SIMPLE & WORKING VERSION
Production-Grade RAG System with Gemini AI
"""

import os
import sys
import time
import hashlib
import logging
import threading
import re
import random
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime
from collections import OrderedDict, defaultdict
import traceback

# ========== THIRD-PARTY IMPORTS ==========
try:
    import psycopg2
    from psycopg2.pool import ThreadedConnectionPool
    from psycopg2.extras import RealDictCursor
    PSYCOPG2_AVAILABLE = True
except ImportError:
    PSYCOPG2_AVAILABLE = False
    print("❌ Install: pip install psycopg2-binary")

try:
    # Using the NEW google.genai package
    from google import genai
    from google.genai import types
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("❌ Install: pip install google-genai")

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("⚠️  Install: pip install python-dotenv")

# ========== SIMPLE CONFIGURATION ==========
class Config:
    """Simple configuration class."""
    
    # Database
    DB_HOST = os.getenv("DB_HOST", "localhost")
    DB_NAME = os.getenv("DB_NAME", "postgres")
    DB_USER = os.getenv("DB_USER", "postgres")
    DB_PASSWORD = os.getenv("DB_PASSWORD", "")
    DB_PORT = int(os.getenv("DB_PORT", "5432"))
    DB_POOL_MIN = 2
    DB_POOL_MAX = 10
    
    # AI Models - Latest Gemini models
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
    
    # Try these models in order (Gemini 3 Flash included)
    MODEL_PRIORITY = [
        "gemini-2.0-flash",      # Fast and reliable
        "gemini-1.5-pro",        # Most accurate
        "gemini-1.5-flash",      # Legacy but good
        "models/gemini-3-flash", # Latest Gemini 3
        "gemini-3-flash-001",    # Alternate name
    ]
    
    # Performance
    MAX_CHUNKS_PER_QUERY = 8
    MAX_TOKENS_RESPONSE = 1000
    REQUEST_TIMEOUT_SECONDS = 10
    SEARCH_TIMEOUT_SECONDS = 3
    
    # Caching
    CACHE_ENABLED = True
    CACHE_MAX_SIZE = 500
    CACHE_TTL_SECONDS = 1800  # 30 minutes
    
    # Logging
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    
    # Monitoring
    SLOW_QUERY_THRESHOLD = 1.5  # seconds
    
    # Safety
    MAX_QUERY_LENGTH = 500
    MIN_QUERY_LENGTH = 2

# ========== SIMPLE LOGGING ==========
class SimpleLogger:
    """Simple logging setup."""
    
    def __init__(self):
        # Create logger
        self.logger = logging.getLogger("NCERT_SYSTEM")
        self.logger.setLevel(getattr(logging, Config.LOG_LEVEL))
        
        # Console handler
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(message)s',
            datefmt='%H:%M:%S'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
    
    def info(self, msg: str):
        self.logger.info(msg)
    
    def warning(self, msg: str):
        self.logger.warning(msg)
    
    def error(self, msg: str):
        self.logger.error(msg)
    
    def debug(self, msg: str):
        self.logger.debug(msg)

logger = SimpleLogger()

# ========== DATABASE MANAGER ==========
class DatabaseManager:
    """Database connection and search."""
    
    _pool = None
    
    @classmethod
    def get_pool(cls):
        """Get or create connection pool."""
        if not PSYCOPG2_AVAILABLE:
            return None
        
        if cls._pool is None:
            try:
                cls._pool = ThreadedConnectionPool(
                    minconn=Config.DB_POOL_MIN,
                    maxconn=Config.DB_POOL_MAX,
                    host=Config.DB_HOST,
                    database=Config.DB_NAME,
                    user=Config.DB_USER,
                    password=Config.DB_PASSWORD,
                    port=Config.DB_PORT,
                    sslmode="require",
                    connect_timeout=3,
                )
                logger.info(f"Database pool created ({Config.DB_POOL_MIN}-{Config.DB_POOL_MAX} connections)")
            except Exception as e:
                logger.error(f"Failed to create database pool: {e}")
                return None
        
        return cls._pool
    
    @classmethod
    def get_connection(cls):
        """Get database connection."""
        pool = cls.get_pool()
        if not pool:
            return None
        
        try:
            conn = pool.getconn()
            # Test connection
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
            return conn
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            return None
    
    @classmethod
    def release_connection(cls, conn):
        """Release connection back to pool."""
        if cls._pool and conn:
            try:
                cls._pool.putconn(conn)
            except:
                pass
    
    @classmethod
    def search_content(cls, keywords: List[str], chapters: List[str]) -> List[Dict]:
        """Search for relevant content."""
        conn = None
        start_time = time.time()
        
        try:
            conn = cls.get_connection()
            if not conn:
                return []
            
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Set timeout
                cur.execute(f"SET statement_timeout = {Config.SEARCH_TIMEOUT_SECONDS * 1000}")
                
                all_results = []
                seen_ids = set()
                
                # Strategy 1: Chapter match
                if chapters:
                    for chapter in chapters[:3]:
                        cur.execute("""
                            SELECT id, chapter, subject, class_grade, content
                            FROM ncert_chunks 
                            WHERE chapter ILIKE %s
                            ORDER BY LENGTH(content) ASC
                            LIMIT 2
                        """, [f"%{chapter}%"])
                        
                        for row in cur.fetchall():
                            if row['id'] not in seen_ids:
                                row_dict = dict(row)
                                row_dict['relevance'] = 0.9
                                all_results.append(row_dict)
                                seen_ids.add(row_dict['id'])
                
                # Strategy 2: Keyword search
                if keywords:
                    for keyword in keywords[:5]:
                        cur.execute("""
                            SELECT id, chapter, subject, class_grade, content
                            FROM ncert_chunks 
                            WHERE content ILIKE %s
                            ORDER BY LENGTH(content) ASC
                            LIMIT 2
                        """, [f"%{keyword}%"])
                        
                        for row in cur.fetchall():
                            if row['id'] not in seen_ids:
                                row_dict = dict(row)
                                content_lower = row_dict['content'].lower()
                                relevance = 0.7
                                if keyword in content_lower:
                                    relevance = 0.8
                                row_dict['relevance'] = relevance
                                all_results.append(row_dict)
                                seen_ids.add(row_dict['id'])
                
                # Strategy 3: Fallback to Science
                if len(all_results) < 3:
                    cur.execute("""
                        SELECT id, chapter, subject, class_grade, content
                        FROM ncert_chunks 
                        WHERE subject ILIKE '%science%'
                        ORDER BY RANDOM()
                        LIMIT 3
                    """)
                    
                    for row in cur.fetchall():
                        if row['id'] not in seen_ids:
                            row_dict = dict(row)
                            row_dict['relevance'] = 0.4
                            all_results.append(row_dict)
                            seen_ids.add(row_dict['id'])
                
                # Sort by relevance
                all_results.sort(key=lambda x: x.get('relevance', 0), reverse=True)
                
                # Deduplicate chapters
                final_results = []
                seen_chapters = set()
                
                for result in all_results:
                    chapter = result.get('chapter', '')
                    if chapter not in seen_chapters or result['relevance'] > 0.8:
                        final_results.append(result)
                        seen_chapters.add(chapter)
                    
                    if len(final_results) >= Config.MAX_CHUNKS_PER_QUERY:
                        break
                
                search_time = time.time() - start_time
                logger.info(f"Found {len(final_results)} chunks in {search_time:.2f}s")
                
                return final_results
                
        except Exception as e:
            logger.error(f"Search error: {e}")
            return []
        finally:
            if conn:
                cls.release_connection(conn)

# ========== QUERY ANALYZER ==========
class QueryAnalyzer:
    """Analyze user queries."""
    
    # Chapter mapping
    CHAPTER_KEYWORDS = {
        # Biology
        "photosynthesis": ["Life Processes"],
        "respiration": ["Life Processes"],
        "digestion": ["Life Processes"],
        "reproduction": ["How do Organisms Reproduce"],
        "heredity": ["Heredity and Evolution"],
        "evolution": ["Heredity and Evolution"],
        "ecosystem": ["Our Environment"],
        "environment": ["Our Environment"],
        "catenation": ["Carbon and its Compounds"],
        "carbon": ["Carbon and its Compounds"],
        
        # Chemistry
        "chemical": ["Chemical Reactions and Equations"],
        "reaction": ["Chemical Reactions and Equations"],
        "acid": ["Acids, Bases and Salts"],
        "base": ["Acids, Bases and Salts"],
        "acetic": ["Acids, Bases and Salts"],
        "glacial": ["Acids, Bases and Salts"],
        "metal": ["Metals and Non-metals"],
        "copper": ["Metals and Non-metals"],
        "oxide": ["Metals and Non-metals"],
        "galvanization": ["Metals and Non-metals"],
        "corrosion": ["Metals and Non-metals"],
        
        # Physics
        "electricity": ["Electricity"],
        "current": ["Electricity"],
        "magnetic": ["Magnetic Effects of Electric Current"],
        "light": ["Light - Reflection and Refraction"],
        "eye": ["The Human Eye and Colourful World"],
        "force": ["Force and Laws of Motion"],
        "motion": ["Force and Laws of Motion"],
        
        # Hindi
        "प्रकाश": ["Life Processes"],
        "संश्लेषण": ["Life Processes"],
        "श्वसन": ["Life Processes"],
        "विद्युत": ["Electricity"],
        "चुंबकीय": ["Magnetic Effects of Electric Current"],
    }
    
    @classmethod
    def analyze(cls, query: str) -> Dict[str, Any]:
        """Analyze query."""
        query = query.strip()[:Config.MAX_QUERY_LENGTH]
        query_lower = query.lower()
        
        # Extract keywords
        words = re.findall(r'\b[a-zA-Z]{3,}\b', query_lower)
        stop_words = {"what", "is", "are", "the", "a", "an", "in", "on", "at", "to"}
        keywords = [w for w in words if w not in stop_words]
        
        # Find relevant chapters
        relevant_chapters = []
        for keyword in keywords:
            if keyword in cls.CHAPTER_KEYWORDS:
                relevant_chapters.extend(cls.CHAPTER_KEYWORDS[keyword])
        
        # Check full query
        for term, chapters in cls.CHAPTER_KEYWORDS.items():
            if term in query_lower:
                relevant_chapters.extend(chapters)
        
        # Remove duplicates
        relevant_chapters = list(set(relevant_chapters))[:5]
        
        # Detect language
        has_hindi = any('\u0900' <= char <= '\u097F' for char in query)
        
        return {
            "original": query,
            "keywords": keywords,
            "relevant_chapters": relevant_chapters,
            "has_hindi": has_hindi,
        }

# ========== GEMINI AI MANAGER ==========
class GeminiAIManager:
    """Manage Gemini AI models."""
    
    def __init__(self):
        self.client = None
        self.available_models = []
        self.primary_model = None
        self._initialize()
    
    def _initialize(self):
        """Initialize Gemini."""
        if not GEMINI_AVAILABLE or not Config.GEMINI_API_KEY:
            logger.warning("Gemini not available - check API key")
            return
        
        try:
            # Create client
            self.client = genai.Client(api_key=Config.GEMINI_API_KEY)
            
            logger.info("Testing Gemini models...")
            
            # Test each model
            for model_name in Config.MODEL_PRIORITY:
                try:
                    logger.info(f"  Testing: {model_name}")
                    
                    # Quick test
                    response = self.client.models.generate_content(
                        model=model_name,
                        contents="Hello",
                        config=types.GenerateContentConfig(max_output_tokens=5)
                    )
                    
                    self.available_models.append(model_name)
                    
                    if not self.primary_model:
                        self.primary_model = model_name
                        logger.info(f"    ✅ Set as primary")
                    else:
                        logger.info(f"    ✅ Available")
                        
                except Exception as e:
                    error_msg = str(e)
                    if "404" in error_msg:
                        logger.info(f"    ❌ Model not found")
                    else:
                        logger.info(f"    ❌ Error: {str(e)[:50]}")
            
            if self.available_models:
                logger.info(f"✅ Gemini ready with {len(self.available_models)} models")
                logger.info(f"   Primary: {self.primary_model}")
            else:
                logger.error("❌ No Gemini models available")
                
        except Exception as e:
            logger.error(f"Gemini init failed: {e}")
    
    def generate(self, prompt: str, model_name: str = None) -> Tuple[Optional[str], str]:
        """Generate response."""
        if not self.client or not self.available_models:
            return None, "no_models"
        
        model_to_use = model_name or self.primary_model
        
        try:
            start_time = time.time()
            
            response = self.client.models.generate_content(
                model=model_to_use,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.2,
                    top_p=0.8,
                    max_output_tokens=Config.MAX_TOKENS_RESPONSE,
                )
            )
            
            duration = time.time() - start_time
            logger.info(f"Gemini response in {duration:.2f}s using {model_to_use}")
            
            return response.text.strip(), model_to_use
            
        except Exception as e:
            logger.error(f"Gemini error ({model_to_use}): {str(e)[:80]}")
            return None, model_to_use
    
    def generate_with_fallback(self, prompt: str) -> Tuple[Optional[str], str]:
        """Generate with fallback."""
        # Try primary model
        response, model_used = self.generate(prompt, self.primary_model)
        if response:
            return response, model_used
        
        # Try other models
        for model_name in self.available_models:
            if model_name != self.primary_model:
                response, model_used = self.generate(prompt, model_name)
                if response:
                    logger.info(f"Used fallback model: {model_name}")
                    return response, model_used
        
        return None, "all_failed"
    
    def is_available(self) -> bool:
        """Check if AI is available."""
        return len(self.available_models) > 0

# ========== ANSWER GENERATOR ==========
class AnswerGenerator:
    """Generate answers."""
    
    def __init__(self, gemini_manager: GeminiAIManager):
        self.gemini = gemini_manager
    
    def generate(self, question: str, search_results: List[Dict]) -> Dict[str, Any]:
        """Generate answer."""
        start_time = time.time()
        
        if not search_results:
            return self._no_results_response(question)
        
        # Prepare context
        context = self._prepare_context(search_results)
        
        # Try Gemini first
        if self.gemini.is_available():
            prompt = self._create_prompt(question, context)
            answer, model_used = self.gemini.generate_with_fallback(prompt)
            
            if answer:
                confidence = self._calculate_confidence(answer, search_results)
                
                return {
                    "answer": answer,
                    "confidence": confidence,
                    "source": "gemini",
                    "model_used": model_used,
                    "chunks_used": len(search_results),
                    "chapters": list(set(r.get('chapter', '') for r in search_results if r.get('chapter')))[:3],
                }
        
        # Fallback answer
        return self._fallback_answer(search_results)
    
    def _prepare_context(self, search_results: List[Dict]) -> str:
        """Prepare context from search results."""
        context_parts = []
        
        for i, result in enumerate(search_results[:5], 1):
            chapter = result.get('chapter', 'Unknown')
            content = result.get('content', '')
            relevance = result.get('relevance', 0.5)
            
            # Clean content
            content = content.replace('\n', ' ').strip()
            
            context_parts.append(
                f"[Source {i}: {chapter}]\n"
                f"Relevance: {relevance:.1f}\n"
                f"Content: {content[:400]}"
            )
        
        return "\n\n" + "\n---\n".join(context_parts)
    
    def _create_prompt(self, question: str, context: str) -> str:
        """Create prompt for Gemini."""
        return f"""You are an expert NCERT tutor. Answer using ONLY the provided NCERT content.

QUESTION: {question}

NCERT CONTENT:
{context}

INSTRUCTIONS:
1. Answer directly from NCERT content
2. Use simple language for students
3. Structure: Definition → Explanation → Examples
4. Keep concise (100-200 words)
5. If incomplete information, say so
6. NEVER add information not in extracts
7. Mention relevant NCERT chapters

ANSWER:"""
    
    def _calculate_confidence(self, answer: str, search_results: List[Dict]) -> float:
        """Calculate confidence."""
        confidence = 0.5
        
        # Answer length
        if 80 < len(answer) < 400:
            confidence += 0.2
        
        # Search relevance
        if search_results:
            avg_relevance = sum(r.get('relevance', 0.5) for r in search_results[:3]) / min(3, len(search_results))
            confidence += avg_relevance * 0.3
        
        return min(1.0, max(0.1, confidence))
    
    def _fallback_answer(self, search_results: List[Dict]) -> Dict[str, Any]:
        """Generate fallback answer."""
        # Sort by relevance
        search_results.sort(key=lambda x: x.get('relevance', 0), reverse=True)
        
        best_result = search_results[0]
        chapter = best_result.get('chapter', 'NCERT')
        content = best_result.get('content', '')
        
        # Extract meaningful sentences
        sentences = [s.strip() for s in content.split('. ') if s.strip() and len(s) > 20]
        
        if len(sentences) > 1:
            answer = f"According to NCERT {chapter}: {'. '.join(sentences[:2])}."
        else:
            answer = f"From NCERT {chapter}: {content[:250]}..."
        
        return {
            "answer": answer,
            "confidence": 0.3,
            "source": "fallback",
            "model_used": "content",
            "chunks_used": 1,
            "chapters": [chapter],
        }
    
    def _no_results_response(self, question: str) -> Dict[str, Any]:
        """No results response."""
        suggestions = [
            "What is photosynthesis?",
            "Explain chemical reactions",
            "What is galvanization?",
            "Describe human eye",
            "What are acids and bases?",
            "Explain electricity",
            "What is heredity?",
        ]
        
        random_suggestions = random.sample(suggestions, 3)
        suggestions_text = " | ".join(random_suggestions)
        
        return {
            "answer": f"No NCERT content found for '{question}'. Try: {suggestions_text}",
            "confidence": 0.1,
            "source": "no_results",
            "model_used": "none",
            "chunks_used": 0,
            "chapters": [],
        }

# ========== CACHE ==========
class SimpleCache:
    """Simple cache with TTL."""
    
    def __init__(self, max_size: int = 500, ttl: int = 1800):
        self.max_size = max_size
        self.ttl = ttl
        self.cache = OrderedDict()
    
    def _make_key(self, query: str) -> str:
        """Create cache key."""
        return hashlib.md5(query.lower().encode()).hexdigest()[:12]
    
    def get(self, query: str) -> Optional[Any]:
        """Get from cache."""
        if not Config.CACHE_ENABLED:
            return None
        
        key = self._make_key(query)
        
        if key not in self.cache:
            return None
        
        data, timestamp = self.cache[key]
        
        if time.time() - timestamp > self.ttl:
            del self.cache[key]
            return None
        
        # Move to end (most recently used)
        self.cache.move_to_end(key)
        return data
    
    def set(self, query: str, data: Any):
        """Set cache entry."""
        if not Config.CACHE_ENABLED:
            return
        
        key = self._make_key(query)
        
        # Evict if needed
        if len(self.cache) >= self.max_size:
            self.cache.popitem(last=False)
        
        self.cache[key] = (data, time.time())
        self.cache.move_to_end(key)
    
    def clear(self):
        """Clear cache."""
        self.cache.clear()
        logger.info("Cache cleared")
    
    def size(self) -> int:
        """Get cache size."""
        return len(self.cache)

# ========== MAIN SYSTEM ==========
class NCERTSystem:
    """Main NCERT RAG system."""
    
    def __init__(self):
        print("\n" + "="*60)
        print("🚀 NCERT ULTRA PRO - SIMPLE & WORKING")
        print("="*60)
        
        # Initialize components
        self.query_analyzer = QueryAnalyzer()
        self.gemini_manager = GeminiAIManager()
        self.answer_generator = AnswerGenerator(self.gemini_manager)
        self.cache = SimpleCache(max_size=Config.CACHE_MAX_SIZE, ttl=Config.CACHE_TTL_SECONDS)
        
        # Display info
        self._display_info()
    
    def _display_info(self):
        """Display system information."""
        print("\n📊 SYSTEM STATUS:")
        print(f"   Database: {'✅ Available' if PSYCOPG2_AVAILABLE else '❌ Not available'}")
        print(f"   Gemini AI: {'✅ Available' if self.gemini_manager.is_available() else '❌ Not available'}")
        
        if self.gemini_manager.is_available():
            print(f"   Primary Model: {self.gemini_manager.primary_model}")
        
        print(f"   Cache: {self.cache.size()}/{Config.CACHE_MAX_SIZE}")
        print("="*60)
    
    def query(self, question: str) -> Dict[str, Any]:
        """Process query."""
        total_start = time.time()
        
        # Validate
        if len(question) < Config.MIN_QUERY_LENGTH:
            return self._error_response("Query too short")
        
        if len(question) > Config.MAX_QUERY_LENGTH:
            question = question[:Config.MAX_QUERY_LENGTH]
        
        # Check cache
        cached = self.cache.get(question)
        if cached:
            cached["from_cache"] = True
            cached["response_time"] = time.time() - total_start
            logger.info(f"Cache hit: {question[:50]}...")
            return cached
        
        logger.info(f"Processing: {question[:50]}...")
        
        try:
            # Analyze query
            analysis = self.query_analyzer.analyze(question)
            
            if analysis['keywords']:
                logger.info(f"Keywords: {analysis['keywords'][:3]}")
            
            # Search
            search_start = time.time()
            search_results = DatabaseManager.search_content(
                keywords=analysis['keywords'],
                chapters=analysis['relevant_chapters']
            )
            search_time = time.time() - search_start
            
            # Generate answer
            gen_start = time.time()
            answer_data = self.answer_generator.generate(
                question=analysis['original'],
                search_results=search_results
            )
            gen_time = time.time() - gen_start
            
            # Prepare response
            total_time = time.time() - total_start
            
            response = {
                "success": True,
                "answer": answer_data["answer"],
                "confidence": answer_data["confidence"],
                "metadata": {
                    "chunks_used": answer_data["chunks_used"],
                    "chapters": answer_data.get("chapters", []),
                    "source": answer_data["source"],
                    "model": answer_data.get("model_used", "none"),
                },
                "timing": {
                    "total": total_time,
                    "search": search_time,
                    "generation": gen_time,
                },
                "from_cache": False,
            }
            
            # Cache
            self.cache.set(question, response)
            
            if total_time > Config.SLOW_QUERY_THRESHOLD:
                logger.warning(f"Slow query: {total_time:.2f}s")
            
            return response
            
        except Exception as e:
            logger.error(f"Error: {e}")
            return self._error_response("System error")
    
    def _error_response(self, message: str) -> Dict[str, Any]:
        """Error response."""
        return {
            "success": False,
            "answer": f"Error: {message}",
            "from_cache": False,
            "timing": {"total": 0}
        }
    
    def clear_cache(self):
        """Clear cache."""
        self.cache.clear()

# ========== INTERACTIVE MODE ==========
def interactive_mode():
    """Interactive command line."""
    print("\n🎮 INTERACTIVE MODE")
    print("Commands: exit, clear, test, help")
    print("-" * 60)
    
    system = NCERTSystem()
    
    while True:
        try:
            question = input("\n❓ Question: ").strip()
            
            if not question:
                continue
            
            if question.lower() in ['exit', 'quit', 'q']:
                print("\n👋 Goodbye!")
                break
            
            if question.lower() == 'clear':
                system.clear_cache()
                print("✅ Cache cleared")
                continue
            
            if question.lower() == 'test':
                test_questions = [
                    "What is photosynthesis?",
                    "Explain chemical reactions",
                    "What is galvanization?",
                    "What is glacial acetic acid?",
                    "What is catenation?",
                    "Color of copper oxide",
                ]
                
                print("\n🧪 Quick tests:")
                for q in test_questions:
                    result = system.query(q)
                    status = "✅" if result["success"] else "❌"
                    conf = f"{result.get('confidence', 0):.0%}" if "confidence" in result else ""
                    print(f"  {status} {q[:30]}... {conf}")
                    time.sleep(0.3)
                continue
            
            if question.lower() == 'help':
                print("\n📖 HELP:")
                print("  • Ask NCERT Science questions")
                print("  • System uses Gemini AI")
                print("  • Commands: exit, clear, test, help")
                continue
            
            # Process query
            result = system.query(question)
            
            # Display result
            print("\n" + "="*60)
            
            if result["success"]:
                print("📗 ANSWER:")
                print("-" * 60)
                print(result["answer"])
                print("-" * 60)
                
                print(f"\n📊 Details:")
                print(f"  • Confidence: {result['confidence']:.0%}")
                print(f"  • Source: {result['metadata']['source']}")
                
                if result['metadata']['model'] != 'none':
                    print(f"  • Model: {result['metadata']['model']}")
                
                if result['metadata']['chapters']:
                    print(f"  • Chapters: {', '.join(result['metadata']['chapters'][:2])}")
                
                if result.get('from_cache'):
                    print(f"  • ⚡ From cache")
                
                timing = result['timing']
                print(f"\n⏱️  Timing:")
                print(f"  • Total: {timing['total']:.2f}s")
                print(f"  • Search: {timing['search']:.2f}s")
                print(f"  • Generation: {timing['generation']:.2f}s")
            else:
                print(f"❌ {result['answer']}")
            
            print("="*60)
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n⚠️  Error: {str(e)[:50]}")

# ========== MAIN ==========
def main():
    """Main entry point."""
    os.system('cls' if os.name == 'nt' else 'clear')
    
    print("\n" + "="*60)
    print("🚀 NCERT ULTRA PRO - WORKING VERSION")
    print("="*60)
    
    # Check dependencies
    if not PSYCOPG2_AVAILABLE or not GEMINI_AVAILABLE:
        print("\n❌ Missing dependencies. Install:")
        print("   pip install psycopg2-binary google-genai python-dotenv")
        sys.exit(1)
    
    # Check API key
    if not Config.GEMINI_API_KEY:
        print("⚠️  WARNING: GEMINI_API_KEY not set")
        print("   Create .env file or set environment variable")
    
    # Start
    interactive_mode()

if __name__ == "__main__":
    main()