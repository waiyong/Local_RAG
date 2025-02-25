# config.py

# Llama 3.2 GGUF Model Path
MODEL_PATH = "./quantized_model/Llama-3.1-8B-Instruct-Q5KM.gguf"

# Embedding Model Path
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
EMBEDDING_MODEL_PATH = "./embedding_model"

# Document Directory
DOCUMENT_PATH = "./Document"

# Retrieval settings
TOP_K = 3  # Number of retrieved documents

# Chainlit settings
HOST = "0.0.0.0"
PORT = 8000
