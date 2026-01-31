# api/index.py
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Import your Flask app from api_server.py
from api_server import app

# Vercel requires this variable name
application = app

# Optional: Simple handler
def handler(event, context):
    return application