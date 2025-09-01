# test_install.py
import faiss
import pdfplumber
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

print("All libraries imported successfully!")

# quick embedding test
model = SentenceTransformer("all-MiniLM-L6-v2")
emb = model.encode("Hello finance world!")
print("Embedding shape:", emb.shape)
