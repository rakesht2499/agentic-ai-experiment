# nltk_initializer.py
import os
import nltk

# Use a safe temporary directory for NLTK data
NLTK_DATA_DIR = "/tmp/nltk_data"
os.makedirs(NLTK_DATA_DIR, exist_ok=True)

# Tell NLTK to use this directory
nltk.data.path.append(NLTK_DATA_DIR)

# Ensure 'punkt' tokenizer is available
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", download_dir=NLTK_DATA_DIR)
