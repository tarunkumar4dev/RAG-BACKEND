"""
NCERT RAG API SERVER - VERCEL PRODUCTION READY
Fast, scalable Flask API optimized for serverless deployment
"""

import os
import sys
import json
import time
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

# Import RAG system
try:
    from rag_system import NCERTRAGSystem, CONFIG
    RAG_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Warning: Could not import RAG system: {e}")
    RAG_AVAILABLE = False

# ========== LOGGING SETUP ==========
def setup_logging():
    """Setup production logging for Vercel."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

# ========== FLASK APP SETUP ==========
app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False  # Disable for production
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB max

# Enable CORS
CORS(app, resources={
    r"/*": {
        "origins": ["*"],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization", "X-API-Key"]
    }
})

# Rate limiting with memory storage (serverless compatible)
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["100 per day", "20 per hour"],
    storage_uri="memory://",
    strategy="fixed-window"  # Serverless compatible
)

# ========== RAG SYSTEM WRAPPER (Serverless Optimized) ==========
class ServerlessRAGSystem:
    """RAG system optimized for serverless environments."""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ServerlessRAGSystem, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        """Initialize with lazy loading for serverless."""
        self.start_time = datetime.now()
        self.rag = None
        self.initialized = False
        self.stats = {
            "total_requests": 0,
            "successful_queries": 0,
            "failed_queries": 0,
            "total_response_time": 0,
            "cache_hits": 0
        }
    
    def _ensure_initialized(self):
        """Lazy initialization for cold starts."""
        if not self.initialized and RAG_AVAILABLE:
            try:
                logger.info("🚀 Initializing NCERT RAG System (lazy load)...")
                self.rag = NCERTRAGSystem()
                self.initialized = True
                logger.info("✅ RAG System initialized")
            except Exception as e:
                logger.error(f"❌ Failed to initialize RAG system: {e}")
                raise
    
    def query(self, question: str, use_cache: bool = True) -> Dict[str, Any]:
        """Process query with lazy initialization."""
        start_time = time.time()
        self.stats["total_requests"] += 1
        
        try:
            # Ensure system is initialized
            self._ensure_initialized()
            
            if not self.initialized:
                raise Exception("RAG system not available")
            
            # Process query
            result = self.rag.query(question, use_cache)
            
            # Update stats
            response_time = time.time() - start_time
            self.stats["total_response_time"] += response_time
            
            if result["chunks_used"] > 0:
                self.stats["successful_queries"] += 1
            else:
                self.stats["failed_queries"] += 1
            
            if result.get("cache_hit"):
                self.stats["cache_hits"] += 1
            
            # Add API metadata
            result["api_metadata"] = {
                "request_id": result["query_id"],
                "timestamp": datetime.now().isoformat(),
                "response_time": response_time,
                "version": "2.0.0",
                "serverless": True
            }
            
            return result
            
        except Exception as e:
            self.stats["failed_queries"] += 1
            logger.error(f"Query failed: {e}")
            
            # Return fallback response
            return {
                "query_id": "error_" + str(int(time.time())),
                "answer": "I'm currently unable to process your request. Please try again later.",
                "chunks_used": 0,
                "chapters": [],
                "cache_hit": False,
                "response_time": time.time() - start_time,
                "model_used": "error",
                "similarity_score": 0
            }
    
    def get_api_stats(self) -> Dict[str, Any]:
        """Get API statistics."""
        # Calculate averages
        avg_response_time = 0
        if self.stats["total_requests"] > 0:
            avg_response_time = self.stats["total_response_time"] / self.stats["total_requests"]
        
        rag_stats = {}
        if self.initialized and self.rag:
            try:
                rag_stats = self.rag.get_system_stats()
            except:
                rag_stats = {"error": "Failed to get RAG stats"}
        
        return {
            "api": {
                "total_requests": self.stats["total_requests"],
                "successful_queries": self.stats["successful_queries"],
                "failed_queries": self.stats["failed_queries"],
                "success_rate": (
                    self.stats["successful_queries"] / self.stats["total_requests"] * 100 
                    if self.stats["total_requests"] > 0 else 0
                ),
                "cache_hits": self.stats["cache_hits"],
                "avg_response_time": f"{avg_response_time:.2f}s",
                "uptime": str(datetime.now() - self.start_time),
                "initialized": self.initialized
            },
            "rag_system": rag_stats,
            "environment": {
                "rag_available": RAG_AVAILABLE,
                "serverless": True,
                "timestamp": datetime.now().isoformat()
            }
        }

# Initialize serverless RAG system
rag_system = ServerlessRAGSystem()

# ========== ERROR HANDLERS ==========
@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "error": "Not Found",
        "message": "The requested endpoint does not exist",
        "status": 404,
        "timestamp": datetime.now().isoformat()
    }), 404

@app.errorhandler(429)
def ratelimit_handler(error):
    return jsonify({
        "error": "Rate Limit Exceeded",
        "message": "Too many requests. Please try again later.",
        "status": 429,
        "timestamp": datetime.now().isoformat()
    }), 429

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {error}")
    return jsonify({
        "error": "Internal Server Error",
        "message": "An unexpected error occurred",
        "status": 500,
        "timestamp": datetime.now().isoformat()
    }), 500

# ========== MIDDLEWARE ==========
@app.before_request
def before_request():
    """Log incoming requests."""
    if request.path not in ['/health', '/favicon.ico', '/robots.txt']:
        logger.info(f"📥 {request.method} {request.path}")

@app.after_request
def after_request(response):
    """Add security headers."""
    # Security headers
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
    
    # CORS headers
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
    
    return response

# ========== API ENDPOINTS ==========
@app.route('/', methods=['GET'])
@limiter.exempt
def index():
    """API information endpoint."""
    return jsonify({
        "api": "NCERT RAG API",
        "version": "2.0.0",
        "description": "AI-powered NCERT textbook question answering system",
        "deployment": "Vercel Serverless",
        "status": "operational" if rag_system.initialized else "initializing",
        "endpoints": {
            "GET /": "API information",
            "GET /health": "Health check",
            "GET /status": "System status",
            "GET /stats": "API statistics",
            "GET /chapters": "List available chapters",
            "POST /query": "Ask questions",
            "POST /batch-query": "Ask multiple questions (max 5)"
        },
        "limits": {
            "queries_per_minute": 10,
            "queries_per_hour": 20,
            "batch_size": 5
        },
        "documentation": "https://github.com/yourusername/ncert-rag-api",
        "timestamp": datetime.now().isoformat()
    })

@app.route('/health', methods=['GET'])
@limiter.exempt
def health_check():
    """Health check endpoint."""
    try:
        stats = rag_system.get_api_stats()
        
        return jsonify({
            "status": "healthy" if rag_system.initialized else "initializing",
            "timestamp": datetime.now().isoformat(),
            "components": {
                "api": "operational",
                "rag_system": "available" if rag_system.initialized else "unavailable",
                "database": "connected" if rag_system.initialized else "unknown",
                "gemini": "available" if rag_system.initialized else "unknown"
            },
            "version": "2.0.0",
            "uptime": stats["api"]["uptime"]
        })
    except Exception as e:
        return jsonify({
            "status": "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "error": str(e),
            "components": {
                "api": "operational",
                "rag_system": "error",
                "database": "unknown",
                "gemini": "unknown"
            }
        }), 500

@app.route('/status', methods=['GET'])
@limiter.exempt
def status():
    """System status endpoint."""
    stats = rag_system.get_api_stats()
    
    return jsonify({
        "status": "operational" if rag_system.initialized else "initializing",
        "rag_system": {
            "initialized": rag_system.initialized,
            "available": RAG_AVAILABLE,
            **stats.get("rag_system", {})
        },
        "api": stats["api"],
        "environment": stats["environment"],
        "timestamp": datetime.now().isoformat()
    })

@app.route('/stats', methods=['GET'])
@limiter.exempt
def get_stats():
    """Get API statistics."""
    return jsonify(rag_system.get_api_stats())

@app.route('/chapters', methods=['GET'])
@limiter.limit("5 per minute")
def list_chapters():
    """List all available chapters."""
    try:
        rag_system._ensure_initialized()
        
        if not rag_system.initialized:
            return jsonify({
                "error": "System not initialized",
                "message": "Please try again in a moment"
            }), 503
        
        # Try to get chapters from RAG system
        if hasattr(rag_system.rag, 'list_chapters_sync'):
            chapters = rag_system.rag.list_chapters_sync()
        else:
            # Fallback if method doesn't exist
            chapters = []
        
        return jsonify({
            "success": True,
            "count": len(chapters),
            "chapters": chapters[:50],  # Limit to 50
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Failed to list chapters: {e}")
        return jsonify({
            "success": False,
            "error": "Failed to retrieve chapters",
            "message": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500

@app.route('/query', methods=['POST', 'OPTIONS'])
@limiter.limit("10 per minute")
def query():
    """
    Query endpoint - Ask questions to NCERT RAG system.
    
    Request JSON:
    {
        "question": "What is photosynthesis?",
        "use_cache": true
    }
    """
    # Handle OPTIONS for CORS
    if request.method == 'OPTIONS':
        return '', 200
    
    try:
        # Get request data
        data = request.get_json(silent=True)
        
        if not data:
            return jsonify({
                "success": False,
                "error": "Invalid request",
                "message": "Request body must be JSON",
                "timestamp": datetime.now().isoformat()
            }), 400
        
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
        
        # Process query
        logger.info(f"Processing query: '{question[:50]}...'")
        result = rag_system.query(question, use_cache)
        
        # Prepare response
        response = {
            "success": result["chunks_used"] > 0,
            "question": question,
            "answer": result["answer"],
            "metadata": {
                "query_id": result["query_id"],
                "chunks_used": result["chunks_used"],
                "chapters": result["chapters"][:5],  # Limit chapters
                "similarity_score": float(result["similarity_score"]),
                "response_time": result["response_time"],
                "model_used": result["model_used"],
                "cache_hit": result.get("cache_hit", False)
            },
            "api_metadata": result.get("api_metadata", {}),
            "timestamp": datetime.now().isoformat()
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Query endpoint error: {e}", exc_info=True)
        return jsonify({
            "success": False,
            "error": "Internal server error",
            "message": str(e)[:100],
            "timestamp": datetime.now().isoformat()
        }), 500

@app.route('/batch-query', methods=['POST', 'OPTIONS'])
@limiter.limit("5 per minute")
def batch_query():
    """
    Batch query endpoint - Ask multiple questions at once.
    
    Request JSON:
    {
        "questions": ["What is photosynthesis?", "Explain chemical reactions"],
        "use_cache": true
    }
    """
    # Handle OPTIONS for CORS
    if request.method == 'OPTIONS':
        return '', 200
    
    try:
        data = request.get_json(silent=True)
        
        if not data:
            return jsonify({
                "success": False,
                "error": "Invalid request",
                "message": "Request body must be JSON"
            }), 400
        
        questions = data.get('questions', [])
        
        if not isinstance(questions, list):
            return jsonify({
                "success": False,
                "error": "Invalid request", 
                "message": "Questions must be an array"
            }), 400
        
        if len(questions) > 5:
            return jsonify({
                "success": False,
                "error": "Invalid request",
                "message": "Maximum 5 questions per batch"
            }), 400
        
        use_cache = data.get('use_cache', True)
        
        results = []
        for question in questions:
            if isinstance(question, str) and question.strip():
                try:
                    result = rag_system.query(question.strip(), use_cache)
                    results.append({
                        "question": question,
                        "answer": result["answer"],
                        "chunks_used": result["chunks_used"],
                        "success": result["chunks_used"] > 0,
                        "response_time": result["response_time"]
                    })
                except Exception as e:
                    results.append({
                        "question": question,
                        "answer": f"Error processing question: {str(e)[:50]}",
                        "chunks_used": 0,
                        "success": False,
                        "error": str(e)[:100]
                    })
            else:
                results.append({
                    "question": question,
                    "answer": "Invalid question format",
                    "chunks_used": 0,
                    "success": False,
                    "error": "Question must be a non-empty string"
                })
        
        return jsonify({
            "success": True,
            "count": len(results),
            "results": results,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Batch query error: {e}")
        return jsonify({
            "success": False,
            "error": "Internal server error",
            "message": str(e)
        }), 500

# ========== UTILITY ENDPOINTS ==========
@app.route('/favicon.ico')
@limiter.exempt
def favicon():
    return '', 204

@app.route('/robots.txt')
@limiter.exempt  
def robots():
    return Response("User-agent: *\nAllow: /\nDisallow: /admin/", mimetype='text/plain')

# ========== VERCEL COMPATIBILITY ==========
# Vercel requires this variable name
application = app

# Optional: Simple handler if needed
def handler(event, context):
    """Vercel serverless handler - Simplified version."""
    # Let Flask handle everything
    return application

# ========== LOCAL DEVELOPMENT ==========
if __name__ == '__main__':
    # Print startup banner
    print("\n" + "="*60)
    print("🚀 NCERT RAG API SERVER - LOCAL DEVELOPMENT")
    print("="*60)
    print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🌐 http://localhost:5000")
    print(f"🔧 RAG System Available: {RAG_AVAILABLE}")
    print("="*60)
    
    # Run Flask development server
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=False,
        threaded=True
    )