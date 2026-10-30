"""
Production Financial Research Assistant Agentic RAG System
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

try:
    import torch
except ImportError:
    pass

__version__ = "1.0.0"
