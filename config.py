"""
CONFIGURATION FOR NCERT RAG SYSTEM
Production-ready configuration with FAISS vector database
"""

import os
from pathlib import Path
from typing import Dict, Any

# ========== DATABASE CONFIGURATION ==========
DATABASE_CONFIG = {
    "host": "db.dcmnzvjftmdbywrjkust.supabase.co",
    "port": 5432,
    "database": "postgres",
    "user": "postgres",
    "password": "",  # Will be loaded from environment
    "connection_timeout": 30,
    "query_timeout": 10,
    "sslmode": "require",
    "pool_minconn": 2,
    "pool_maxconn": 20
}

# ========== AI CONFIGURATION ==========
AI_CONFIG = {
    "gemini_api_key": "",  # Will be loaded from environment
    
    # Model settings
    "gemini_temperature": 0.2,
    "gemini_max_tokens": 800,
    "gemini_timeout": 30,
    
    # FAISS Vector Database (NO Pinecone needed)
    "vector_db": "faiss",
    "faiss_index_path": "./faiss_index",
    "embedding_dimension": 768,
    
    # Model priorities
    "model_priority": [
        "gemini-2.0-flash",
        "gemini-1.5-flash", 
        "gemini-1.5-pro"
    ]
}

# ========== SYSTEM CONFIGURATION ==========
SYSTEM_CONFIG = {
    # Performance
    "max_retries": 3,
    "retry_delay": 1,
    "max_chunks_per_query": 10,
    
    # Query limits
    "max_query_length": 500,
    "min_chunk_length": 50,
    "max_chunk_length": 300,
    
    # Logging
    "log_level": "INFO",
    "log_to_file": False  # Simple logging for Vercel
}

# ========== LOAD ENVIRONMENT VARIABLES ==========
def load_environment_variables() -> bool:
    """
    Load environment variables from .env file or system environment.
    Returns: True if all required variables are loaded
    """
    # Try to load from .env file
    env_path = Path(__file__).parent / '.env'
    if env_path.exists():
        try:
            with open(env_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and '=' in line and not line.startswith('#'):
                        key, value = line.split('=', 1)
                        key = key.strip()
                        value = value.strip().strip('"').strip("'")
                        os.environ[key] = value
        except Exception as e:
            print(f"⚠️ Warning: Could not load .env file: {e}")
    
    # Load database password
    db_password = os.getenv("DATABASE_PASSWORD", "")
    if db_password:
        DATABASE_CONFIG["password"] = db_password
    
    # Load AI keys
    gemini_key = os.getenv("GEMINI_API_KEY", "")
    if gemini_key:
        AI_CONFIG["gemini_api_key"] = gemini_key
    
    # Check required variables
    required_vars = []
    if not DATABASE_CONFIG["password"]:
        required_vars.append("DATABASE_PASSWORD")
    if not AI_CONFIG["gemini_api_key"]:
        required_vars.append("GEMINI_API_KEY")
    
    if required_vars:
        print(f"⚠️ Missing environment variables: {', '.join(required_vars)}")
        print("💡 Set these in Vercel Environment Variables or .env file")
        return False
    
    return True

# ========== EXPORT CONFIGURATION ==========
# Combine all configs
CONFIG = {
    "database": DATABASE_CONFIG,
    "ai": AI_CONFIG,
    "system": SYSTEM_CONFIG
}

# Load environment on import
ENV_LOADED = load_environment_variables()

# ========== HELPER FUNCTIONS ==========
def get_config() -> Dict[str, Any]:
    """Get complete configuration."""
    return CONFIG

def get_database_config() -> Dict[str, Any]:
    """Get database configuration."""
    return DATABASE_CONFIG

def get_ai_config() -> Dict[str, Any]:
    """Get AI configuration."""
    return AI_CONFIG

def get_system_config() -> Dict[str, Any]:
    """Get system configuration."""
    return SYSTEM_CONFIG

def is_configured() -> bool:
    """Check if system is properly configured."""
    return ENV_LOADED

# ========== TEST FUNCTION ==========
def test_configuration():
    """Test configuration loading."""
    print("\n" + "="*60)
    print("🔧 CONFIGURATION TEST")
    print("="*60)
    
    print(f"✅ Environment loaded: {ENV_LOADED}")
    print(f"✅ Database configured: {bool(DATABASE_CONFIG['password'])}")
    print(f"✅ Gemini configured: {bool(AI_CONFIG['gemini_api_key'])}")
    print(f"✅ Vector DB: FAISS (No API key needed)")
    
    if not ENV_LOADED:
        print("\n⚠️ IMPORTANT: Set these environment variables:")
        print("   - DATABASE_PASSWORD")
        print("   - GEMINI_API_KEY")
    
    print("="*60)

# Run test if executed directly
if __name__ == "__main__":
    test_configuration()