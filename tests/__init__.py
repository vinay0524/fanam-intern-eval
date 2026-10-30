import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

try:
    import torch
except ImportError:
    pass

try:
    import faiss
except ImportError:
    pass
