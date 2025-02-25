from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from config import DOCUMENT_PATH, EMBEDDING_MODEL_NAME, EMBEDDING_MODEL_PATH, TOP_K
from llm_loader import load_llm

def build_rag_pipeline():
    """Loads documents, initializes embeddings, and sets up retrieval."""

    # Load LLM
    llm = load_llm()
    Settings.llm = llm  # Assign model globally in LlamaIndex

    # Load embedding model from local storage
    Settings.embed_model = HuggingFaceEmbedding(
        model_name=EMBEDDING_MODEL_NAME,
        cache_folder=EMBEDDING_MODEL_PATH
    )

    # Load and index documents
    documents = SimpleDirectoryReader(DOCUMENT_PATH).load_data()
    index = VectorStoreIndex.from_documents(documents)

    # Create retriever for document search
    retriever = index.as_retriever(similarity_top_k=TOP_K)

    return retriever, llm, index
