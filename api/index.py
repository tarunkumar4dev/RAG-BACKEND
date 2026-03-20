import sys
import os

# Add test-generator-backend to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'test-generator-backend'))

from app.main import app