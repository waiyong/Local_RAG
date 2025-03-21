import chainlit as cl
import chromadb
from llama_index.core import (
    VectorStoreIndex,
    Settings,
    StorageContext,
    PromptTemplate
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.callbacks import CallbackManager
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.chat_engine import CondenseQuestionChatEngine
from config import EMBEDDING_MODEL_NAME, EMBEDDING_MODEL_PATH, TOP_K,MAX_TOKENS_GENERATE,MEMORY_LENGTH
from llm_loader import load_llm


# Initialize ChromaDB Storage
chroma_client = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = chroma_client.get_or_create_collection("financial_reports_2")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# Load embedding model from local storage
Settings.embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    cache_folder="./embedding_cache"
)
index = VectorStoreIndex.from_vector_store(
    vector_store=vector_store,
    storage_context=storage_context
)

custom_prompt = PromptTemplate(
    """\
Rewrite the user's follow-up question as a standalone question.

1. Include all relevant past context.
2. Keep it natural and grammatically correct.
3. If already standalone, return it unchanged.

<Chat History>
{chat_history}

<User's Follow-Up Question>
{question}

<Rewritten Standalone Question>
"""
)


response_prompt = PromptTemplate(
    """\
You are an AI assistant providing structured responses.

### **Instructions:**
- Answer clearly and concisely.
- Summarize retrieved context to avoid duplication.
- Summarize the key facts efficiently.
- If the context lacks enough details, say: "I don’t have enough information."
- Format responses in natural sentences.

<Retrieved Context>
{context}

<User's Query>
{question}

### **AI Response:**
"""
)

@cl.on_chat_start
async def start():
    """Initialize the chat session when the user starts chatting."""

    # Load LLM
    llm = load_llm()
    Settings.llm = llm

    # Initialize the CallbackManager with Chainlit's callback handler
    callback_manager = CallbackManager([cl.LlamaIndexCallbackHandler()])


    query_engine = index.as_query_engine(
        response_mode="compact",
        response_prompt=response_prompt,
        similarity_top_k=TOP_K,
        max_tokens = MAX_TOKENS_GENERATE,
        service_context=callback_manager,
        streaming=True
    )

    memory = ChatMemoryBuffer.from_defaults(token_limit=MEMORY_LENGTH)

    chat_engine = CondenseQuestionChatEngine.from_defaults(
        query_engine=query_engine,
        memory=memory,
        #condense_question_prompt=custom_prompt,
        verbose=False,
    )

    # Store chat engine in Chainlit session
    cl.user_session.set("chat_engine", chat_engine)

    # Send initial greeting message
    await cl.Message(
        author="Assistant", content="Hello! I'm your AI assistant. How can I help you today?"
    ).send()


@cl.on_message
async def handle_message(message: cl.Message):
    """Handles user input, retrieves documents, and generates AI responses."""
    
    chat_engine = cl.user_session.get("chat_engine")
    if not chat_engine:
        await cl.Message(content="Chat session not initialized. Please restart the chat.").send()
        return
    
    msg = cl.Message(content="", author="Assistant")  # Placeholder for response

    response_stream = chat_engine.stream_chat(message.content)
    final_response = ""
    # Stream the response token by token
    for token in response_stream.response_gen:
        final_response += token
        await msg.stream_token(token)


    await msg.send()