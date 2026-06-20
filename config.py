"""
CONFIGURATION FOR NCERT RAG SYSTEM
Production-ready configuration optimized for Vercel
"""

import os
import sys
import logging
from typing import Dict, Any, Optional
from pathlib import Path

# ========== LOGGING SETUP ==========
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# ========== ENVIRONMENT DETECTION ==========
def get_environment() -> str:
    """Detect current environment."""
    if os.getenv("VERCEL_ENV"):
        env = os.getenv("VERCEL_ENV")
    elif os.getenv("ENVIRONMENT"):
        env = os.getenv("ENVIRONMENT")
    elif os.getenv("FLASK_ENV"):
        env = os.getenv("FLASK_ENV")
    else:
        env = "production"
    
    # Map to standard environments
    if env in ["production", "prod"]:
        return "production"
    elif env in ["preview", "staging", "test"]:
        return "staging"
    else:
        return "development"

ENVIRONMENT = get_environment()
IS_PRODUCTION = ENVIRONMENT == "production"
IS_VERCEL = bool(os.getenv("VERCEL"))

# ========== DATABASE CONFIGURATION ==========
# Get values with proper defaults
DATABASE_CONFIG = {
    "host": os.getenv("DB_HOST") or "db.dcmnzvjftmdbywrjkust.supabase.co",
    "port": int(os.getenv("DB_PORT") or "5432"),
    "database": os.getenv("DB_NAME") or "postgres",
    "user": os.getenv("DB_USER") or "postgres",
    "password": os.getenv("DB_PASSWORD") or "",
    
    # Connection settings optimized for serverless
    "connection_timeout": 10,
    "query_timeout": 5,
    "sslmode": "require",
    
    # Pooling settings for Vercel
    "pool_minconn": 1 if IS_VERCEL else 2,
    "pool_maxconn": 5 if IS_VERCEL else 20,
    
    # Keepalive settings for persistent connections
    "keepalives": 1,
    "keepalives_idle": 30,
    "keepalives_interval": 10,
    "keepalives_count": 3,
    
    # Application name for monitoring
    "application_name": f"ncert-rag-{ENVIRONMENT}"
}

# ========== AI & GEMINI CONFIGURATION ==========
# Get Gemini API key from either GEMINI_API_KEY or GEMINI_API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GEMINI_API") or ""

AI_CONFIG = {
    # Gemini API
    "gemini_api_key": GEMINI_API_KEY,
    
    # Model selection - using most reliable for production
    "gemini_model": os.getenv("GEMINI_MODEL") or "gemini-1.5-flash",
    "fallback_models": ["gemini-1.5-flash", "gemini-1.5-pro"],
    
    # Model parameters
    "temperature": float(os.getenv("GEMINI_TEMPERATURE") or "0.2"),
    "max_output_tokens": int(os.getenv("GEMINI_MAX_TOKENS") or "1024"),
    "timeout": int(os.getenv("GEMINI_TIMEOUT") or "30"),
    "top_p": 0.95,
    "top_k": 40,
    
    # Embedding model (for future semantic search)
    "embedding_model": os.getenv("EMBEDDING_MODEL") or "all-MiniLM-L6-v2",
    "embedding_dimension": 384,  # For MiniLM-L6-v2
    
    # Cache settings
    "response_cache_ttl": int(os.getenv("CACHE_TTL") or "300"),  # 5 minutes
    "max_cache_size": int(os.getenv("MAX_CACHE_SIZE") or "100")
}

# ========== SYSTEM CONFIGURATION ==========
SYSTEM_CONFIG = {
    # Environment
    "environment": ENVIRONMENT,
    "is_production": IS_PRODUCTION,
    "is_vercel": IS_VERCEL,
    
    # Application
    "app_name": "NCERT-RAG-API",
    "version": "2.0.0",
    "api_version": "v1",
    
    # Performance settings
    "max_concurrent_requests": int(os.getenv("MAX_CONCURRENT_REQUESTS") or "10"),
    "request_timeout": int(os.getenv("REQUEST_TIMEOUT") or "30"),
    "max_retries": int(os.getenv("MAX_RETRIES") or "3"),
    "retry_delay": float(os.getenv("RETRY_DELAY") or "1.0"),
    
    # Query processing
    "max_query_length": int(os.getenv("MAX_QUERY_LENGTH") or "500"),
    "min_query_length": 3,
    "max_chunks_per_query": int(os.getenv("MAX_CHUNKS_PER_QUERY") or "5"),
    "similarity_threshold": float(os.getenv("SIMILARITY_THRESHOLD") or "0.6"),
    
    # CORS settings
    "cors_origins": (os.getenv("CORS_ORIGINS") or "*").split(","),
    "cors_methods": ["GET", "POST", "OPTIONS"],
    "cors_headers": ["Content-Type", "Authorization"],
    
    # Security
    "rate_limit_enabled": os.getenv("RATE_LIMIT_ENABLED", "true").lower() == "true",
    "rate_limit_requests": int(os.getenv("RATE_LIMIT_REQUESTS") or "100"),
    "rate_limit_period": int(os.getenv("RATE_LIMIT_PERIOD") or "900"),  # 15 minutes
    
    # Logging
    "log_level": os.getenv("LOG_LEVEL") or ("INFO" if IS_PRODUCTION else "DEBUG"),
    "log_format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "enable_request_logging": os.getenv("ENABLE_REQUEST_LOGGING", "true").lower() == "true",
    
    # Health checks
    "health_check_interval": int(os.getenv("HEALTH_CHECK_INTERVAL") or "300"),  # 5 minutes
    "max_response_size": int(os.getenv("MAX_RESPONSE_SIZE") or "1048576"),  # 1MB
    
    # Cache paths for Vercel
    "cache_dir": "/tmp" if IS_VERCEL else "./cache",
    "faiss_cache_dir": "/tmp/faiss_cache" if IS_VERCEL else "./faiss_cache"
}

# ========== VALIDATION ==========
def validate_configuration() -> Dict[str, Any]:
    """
    Validate configuration and return validation results.
    Returns: Dictionary with validation status and issues
    """
    issues = []
    warnings = []
    
    # Required variables - check with actual values
    if not DATABASE_CONFIG["password"]:
        issues.append("Missing required environment variable: DB_PASSWORD")
    
    if not AI_CONFIG["gemini_api_key"]:
        issues.append("Missing required environment variable: GEMINI_API_KEY")
    
    # Database connection validation
    if not DATABASE_CONFIG["host"]:
        issues.append("Missing database host")
    
    # Port validation
    if not 1 <= DATABASE_CONFIG["port"] <= 65535:
        issues.append(f"Invalid database port: {DATABASE_CONFIG['port']}")
    
    # Gemini model validation
    valid_models = {"gemini-1.5-flash", "gemini-1.5-pro", "gemini-2.0-flash"}
    if AI_CONFIG["gemini_model"] not in valid_models:
        warnings.append(f"Gemini model '{AI_CONFIG['gemini_model']}' might not be optimal")
        # Auto-correct to a valid model
        AI_CONFIG["gemini_model"] = "gemini-1.5-flash"
    
    # Performance validation
    if SYSTEM_CONFIG["max_chunks_per_query"] > 10:
        warnings.append("High max_chunks_per_query may impact performance")
    
    if SYSTEM_CONFIG["max_concurrent_requests"] > 20 and IS_VERCEL:
        warnings.append("High concurrent requests may exceed Vercel limits")
    
    # Security warnings for development
    if not IS_PRODUCTION and SYSTEM_CONFIG["cors_origins"] == ["*"]:
        warnings.append("CORS is set to allow all origins (not recommended for production)")
    
    validation_result = {
        "is_valid": len(issues) == 0,
        "environment": ENVIRONMENT,
        "issues": issues,
        "warnings": warnings,
        "required_vars_missing": len([i for i in issues if "Missing required" in i])
    }
    
    return validation_result

# ========== CONFIGURATION LOADER ==========
def load_configuration() -> Dict[str, Any]:
    """
    Load and validate configuration.
    Returns: Complete configuration dictionary
    """
    # Create cache directories
    for dir_path in [SYSTEM_CONFIG["cache_dir"], SYSTEM_CONFIG["faiss_cache_dir"]]:
        try:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.warning(f"Could not create cache directory {dir_path}: {e}")
    
    # Validate configuration
    validation = validate_configuration()
    
    if validation["issues"]:
        logger.error(f"Configuration issues found: {validation['issues']}")
        
    if validation["warnings"]:
        for warning in validation["warnings"]:
            logger.warning(warning)
    
    # Log configuration status
    logger.info(f"Configuration loaded for {ENVIRONMENT} environment")
    logger.info(f"Database: {'✅' if DATABASE_CONFIG['password'] else '❌'}")
    logger.info(f"Gemini AI: {'✅' if AI_CONFIG['gemini_api_key'] else '❌'}")
    logger.info(f"Vercel: {'✅' if IS_VERCEL else '❌'}")
    
    return {
        "database": DATABASE_CONFIG,
        "ai": AI_CONFIG,
        "system": SYSTEM_CONFIG,
        "validation": validation
    }

# ========== EXPORT CONFIGURATION ==========
# Load configuration immediately
CONFIG = load_configuration()

# Helper functions for easy access
def get_config() -> Dict[str, Any]:
    """Get complete configuration."""
    return CONFIG

def get_database_config() -> Dict[str, Any]:
    """Get database configuration."""
    return CONFIG["database"]

def get_ai_config() -> Dict[str, Any]:
    """Get AI configuration."""
    return CONFIG["ai"]

def get_system_config() -> Dict[str, Any]:
    """Get system configuration."""
    return CONFIG["system"]

def get_validation() -> Dict[str, Any]:
    """Get validation results."""
    return CONFIG.get("validation", {})

def is_production() -> bool:
    """Check if running in production."""
    return IS_PRODUCTION

def is_vercel() -> bool:
    """Check if running on Vercel."""
    return IS_VERCEL

def get_gemini_api_key() -> str:
    """Get Gemini API key."""
    return AI_CONFIG["gemini_api_key"]

def get_db_password() -> str:
    """Get database password."""
    return DATABASE_CONFIG["password"]

# ========== ENVIRONMENT-SPECIFIC OVERRIDES ==========
def get_database_connection_string() -> Optional[str]:
    """Get database connection string based on environment."""
    db_config = get_database_config()
    
    if IS_VERCEL:
        # Use connection pooling for Vercel
        return None  # Let psycopg2 handle it
    
    # Standard connection string for other environments
    return f"postgresql://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}?sslmode=require"

def get_gemini_model() -> str:
    """Get appropriate Gemini model based on environment."""
    ai_config = get_ai_config()
    
    if IS_PRODUCTION:
        return ai_config["gemini_model"]
    else:
        # Use cheaper/faster model for non-production
        return "gemini-1.5-flash"

# ========== TEST FUNCTION ==========
def test_configuration() -> Dict[str, Any]:
    """Test configuration and return results."""
    print("\n" + "="*60)
    print("🔧 CONFIGURATION TEST")
    print("="*60)
    
    validation = get_validation()
    
    print(f"✅ Environment: {ENVIRONMENT}")
    print(f"✅ Vercel: {'Yes' if IS_VERCEL else 'No'}")
    print(f"✅ Production: {'Yes' if IS_PRODUCTION else 'No'}")
    
    print(f"\n📊 CONFIGURATION VALUES:")
    print(f"   DB_HOST: {DATABASE_CONFIG['host'][:30]}...")
    print(f"   DB_USER: {DATABASE_CONFIG['user']}")
    print(f"   DB_PASSWORD: {'Set' if DATABASE_CONFIG['password'] else 'NOT SET'}")
    print(f"   GEMINI_API_KEY: {'Set' if AI_CONFIG['gemini_api_key'] else 'NOT SET'}")
    print(f"   GEMINI_MODEL: {AI_CONFIG['gemini_model']}")
    print(f"   MAX_CHUNKS_PER_QUERY: {SYSTEM_CONFIG['max_chunks_per_query']}")
    print(f"   CACHE_DIR: {SYSTEM_CONFIG['cache_dir']}")
    
    print(f"\n✅ Valid: {'Yes' if validation['is_valid'] else 'No'}")
    
    if not validation["is_valid"]:
        print("\n❌ ISSUES:")
        for issue in validation["issues"]:
            print(f"   - {issue}")
    
    if validation["warnings"]:
        print("\n⚠️ WARNINGS:")
        for warning in validation["warnings"]:
            print(f"   - {warning}")
    
    print("="*60)
    
    return validation

# Run test if executed directly
if __name__ == "__main__":
    test_configuration()