from llama_index.llms.llama_cpp import LlamaCPP
from config import MODEL_PATH

def load_llm():
    """Loads the Llama 3.2 model with Metal backend."""
    return LlamaCPP(
        model_path=MODEL_PATH,
        temperature=0.1,
        max_new_tokens=512,
        context_window=8192,
        generate_kwargs={},
        model_kwargs={"n_gpu_layers": -1, "verbose":False}  # Full Metal acceleration
    )