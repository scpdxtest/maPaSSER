"""
MultiAgent System API Server with RAG Support
Supports multiple LLM models including Mistral-7B, Llama-3.1-8B, and Granite-3.3-8B
Integrates with ChromaDB for Retrieval-Augmented Generation
Supports both Ollama and HuggingFace embeddings
WITH PROPER CANCELLATION SUPPORT
WITH PROPER CANCELLATION SUPPORT AND PERFORMANCE METRICS
"""

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList
import asyncio
from asyncio import Task, CancelledError
import logging
import os
import time
from typing import Optional, List, Dict
import argparse
from datetime import datetime
import httpx
from langchain_community.embeddings import OllamaEmbeddings
from langchain.schema import Document
from sentence_transformers import SentenceTransformer
import numpy as np
import signal
import sys
import threading
import tempfile

# ============================================================
# === DISK SPACE CONFIGURATION (macOS) ===
# ============================================================

# 1. Choose your preferred volume for models and cache
CACHE_VOLUME = "/Volumes/YourExternalDrive"  # Change this!

# Option 1: Use external drive (recommended for space)
# CACHE_VOLUME = "/Volumes/ExternalSSD"

# Option 2: Use main disk (if you have space)
# CACHE_VOLUME = os.path.expanduser("~")

# Option 3: Use specific mounted volume
# CACHE_VOLUME = "/Volumes/MachineLearning"

# 2. HuggingFace model cache
HF_CACHE_DIR = f"{CACHE_VOLUME}/ml_cache/huggingface"
os.environ["HF_HOME"] = HF_CACHE_DIR
os.environ["TRANSFORMERS_CACHE"] = f"{HF_CACHE_DIR}/transformers"
os.environ["HF_DATASETS_CACHE"] = f"{HF_CACHE_DIR}/datasets"

# 3. PyTorch cache
TORCH_CACHE_DIR = f"{CACHE_VOLUME}/ml_cache/torch"
os.environ["TORCH_HOME"] = TORCH_CACHE_DIR
os.environ["TORCH_EXTENSIONS_DIR"] = f"{TORCH_CACHE_DIR}/extensions"

# 4. Temporary files directory
TEMP_DIR = f"{CACHE_VOLUME}/ml_cache/tmp"
os.environ["TMPDIR"] = TEMP_DIR
os.environ["TEMP"] = TEMP_DIR
os.environ["TMP"] = TEMP_DIR

# 5. Create all necessary directories
for directory in [HF_CACHE_DIR, TORCH_CACHE_DIR, TEMP_DIR]:
    os.makedirs(directory, exist_ok=True)
    print(f"📁 Cache directory: {directory}")

# Set Python's tempfile module to use our custom temp dir
tempfile.tempdir = TEMP_DIR

# 6. MPS memory management
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

# ============================================================
# === END DISK SPACE CONFIGURATION ===
# ============================================================

# === Added: instrumentation helpers ===
def _now_ms():
    """Get current time in milliseconds"""
    return int(time.perf_counter() * 1000)

def _count_tokens_approx(text: str, tokenizer=None) -> int:
    """Count tokens approximately"""
    if not text:
        return 0
    if tokenizer is not None:
        try:
            return len(tokenizer.encode(text, add_special_tokens=False))
        except Exception:
            pass
    return max(1, len(str(text).split()))

def _agent_metrics_init(model_key: str, strategy: str, num_agents: int, 
                       context_mode: str, similarity_threshold: float, 
                       k: int, context_window: int) -> Dict:
    """Initialize agent metrics dictionary"""
    return {
        "model_key": model_key,
        "strategy": strategy,
        "num_agents": num_agents,
        "context_mode": context_mode,
        "similarity_threshold": similarity_threshold,
        "k": k,
        "context_window": context_window,
        "retrieval_ms": 0,
        "gen_ms": 0,
        "consensus_ms": 0,
        "end_to_end_ms": 0,
        "prompt_tokens_total": 0,
        "completion_tokens_total": 0,
        "tokens_total": 0,
        "messages_total": 0,
        "turns_total": 0,
        "agents_participated": 0,
        "consensus_rounds": 0,
        "disagreement_rate": None,
        "consensus_method": None,
    }

async def _post_record(backend_url: str, payload: Dict):
    """Post metrics record to backend (best-effort)"""
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            r = await client.post(f"{backend_url}/record", json=payload)
            r.raise_for_status()
    except Exception as e:
        # best-effort logging; don't crash
        print(f"[WARN] backend /record failed: {e}")

# Add helper function for token estimation on server side
def estimateTokens(text: str) -> int:
    """Estimate token count from text (same as frontend)"""
    if not text:
        return 0
    return max(1, len(text) // 4)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set environment variable to avoid tokenizer warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

app = FastAPI(title="MultiAgent System API with RAG", version="2.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Port configuration
DEFAULT_API_PORT = 8004

# Global variables
llm_model = None
llm_tokenizer = None
llm_device = None
current_model_name = None

# Global flag for graceful shutdown
shutdown_event = asyncio.Event()

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    logger.warning("🛑 Received interrupt signal (Ctrl+C)")
    logger.warning("🛑 Marking all tasks for cancellation...")
    
    # Mark all active tasks as cancelled
    for task_id in list(cancellation_tracker._cancelled_tasks):
        cancellation_tracker.mark_cancelled(task_id)
    
    shutdown_event.set()
    logger.info("✅ Shutdown initiated. Waiting for tasks to complete...")

# Register signal handler
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# Cancellation tracking
class CancellationTracker:
    """Track cancellation state across threads"""
    def __init__(self):
        self._cancelled_tasks = set()
        self._lock = threading.Lock()
    
    def mark_cancelled(self, task_id):
        with self._lock:
            self._cancelled_tasks.add(task_id)
            logger.info(f"🛑 Task {task_id} marked for cancellation")
    
    def is_cancelled(self, task_id):
        with self._lock:
            return task_id in self._cancelled_tasks
    
    def remove(self, task_id):
        with self._lock:
            self._cancelled_tasks.discard(task_id)

# Global cancellation tracker
cancellation_tracker = CancellationTracker()

# Custom stopping criteria
class CancellationStoppingCriteria(StoppingCriteria):
    """Stopping criteria that checks for task cancellation"""
    def __init__(self, task_id: str):
        self.task_id = task_id
        self.check_interval = 10  # Check every 10 tokens
        self.token_count = 0
    
    def __call__(self, input_ids, scores, **kwargs):
        self.token_count += 1
        
        # Check cancellation every N tokens (not every token for performance)
        if self.token_count % self.check_interval == 0:
            if cancellation_tracker.is_cancelled(self.task_id):
                logger.warning(f"🛑 Generation stopped for task {self.task_id} at token {self.token_count}")
                return True
        
        return False

# Model mapping - Python 3.9 Compatible
MODEL_MAPPING = {
    # Mistral - Fully supported
    "mistral-7b": "mistralai/Mistral-7B-Instruct-v0.3",
    "mistral-7b-v2": "mistralai/Mistral-7B-Instruct-v0.2",
    "mistral-nemo": "mistralai/Mistral-Nemo-Instruct-2407",
    
    # Llama 3.2 - Now fully supported with transformers 4.46+
    "llama-3.2-3b": "meta-llama/Llama-3.2-3B-Instruct",
    "llama-3.2-1b": "meta-llama/Llama-3.2-1B-Instruct",
    
    # Llama 3.1 - Now supported with transformers 4.46+ on Python 3.9
    "llama-3.1-8b": "meta-llama/Llama-3.1-8B-Instruct",
    
    # Granite - Now fully supported with transformers 4.46+
    "granite-3.3-8b": "ibm-granite/granite-3.3-8b-instruct",

    # Lightweight models - Fully supported
    "phi-2": "microsoft/phi-2",
    "falcon-7b": "tiiuae/falcon-7b-instruct",
    "distilgpt2": "distilgpt2",
    "gpt2": "gpt2",
}

# Model context windows - all modern models support 128K
MODEL_CONTEXT_WINDOWS = {
    "mistralai/Mistral-7B-Instruct-v0.2": 128000,
    "mistralai/Mistral-7B-Instruct-v0.3": 128000,
    "mistralai/Mistral-Nemo-Instruct-2407": 128000,
    "meta-llama/Llama-3.2-3B-Instruct": 128000,
    "meta-llama/Llama-3.2-1B-Instruct": 128000,
    "meta-llama/Llama-3.1-8B-Instruct": 128000,
    "ibm-granite/granite-3.3-8b-instruct": 128000,
    "microsoft/phi-2": 2048,
    "tiiuae/falcon-7b-instruct": 2048,
    "distilgpt2": 1024,
    "gpt2": 1024,
}

def get_model_context_window(model_name: str, custom_size: Optional[int] = None) -> int:
    """Get context window size for a model"""
    default_size = MODEL_CONTEXT_WINDOWS.get(model_name, 4096)
    
    # If custom size provided, use it (capped at model's max)
    if custom_size and custom_size > 0:
        return min(custom_size, default_size)
    
    return default_size

class MultiAgentRequest(BaseModel):
    query: str
    context: Optional[str] = ""
    model: str = "codegen-2b"
    strategy: str = "collaborative"
    num_agents: int = 3
    context_size: Optional[int] = None  # NEW: Custom context size
    # RAG parameters
    use_rag: bool = False
    chroma_url: Optional[str] = "http://127.0.0.1:8000"
    collection_name: Optional[str] = None
    embedding_model: Optional[str] = "mistral"
    embedding_source: Optional[str] = "ollama"
    ollama_url: Optional[str] = "http://127.0.0.1:11434"
    top_k_docs: Optional[int] = 7

class RAGRetriever:
    """RAG Retriever supporting both Ollama and HuggingFace embeddings"""
    def __init__(self, chroma_url: str, collection_name: str, 
                 embedding_model: str = "mistral",
                 embedding_source: str = "ollama",
                 ollama_url: str = "http://127.0.0.1:11434"):
        self.chroma_url = chroma_url.rstrip('/')
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model
        self.embedding_source = embedding_source
        self.ollama_url = ollama_url
        self.collection_id = None
        self.collection_dimension = None
        self.embedding_model = None
        self.http_client = httpx.AsyncClient(timeout=30.0)
        
    async def initialize(self):
        """Initialize the RAG retriever"""
        try:
            logger.info(f"Connecting to ChromaDB at {self.chroma_url}")
            
            # List collections
            list_url = f"{self.chroma_url}/api/v2/tenants/default_tenant/databases/default_database/collections"
            
            try:
                response = await self.http_client.get(list_url)
                response.raise_for_status()
                collections = response.json()
                
                logger.info(f"Found {len(collections)} collections")
                
                # Find our collection
                target_collection = None
                for col in collections:
                    if col.get("name") == self.collection_name:
                        target_collection = col
                        self.collection_id = col.get("id")
                        metadata = col.get("metadata") or {}
                        self.collection_dimension = metadata.get("dimension")
                        break
                
                if not target_collection:
                    available = [c.get("name") for c in collections]
                    logger.error(f"Collection '{self.collection_name}' not found. Available: {available}")
                    return False
                
                logger.info(f"✅ Found collection: {self.collection_name} (ID: {self.collection_id})")
                
            except Exception as e:
                logger.error(f"Failed to list collections: {e}")
                return False
            
            # Load embedding model based on source
            if self.embedding_source == "ollama":
                logger.info(f"Loading Ollama embedding model: {self.embedding_model_name}")
                self.embedding_model = OllamaEmbeddings(
                    model=self.embedding_model_name,
                    base_url=self.ollama_url
                )
                
                # Test embedding dimension
                try:
                    test_embedding = await asyncio.to_thread(
                        self.embedding_model.embed_query,
                        "test"
                    )
                    embedding_dim = len(test_embedding)
                    logger.info(f"   Ollama model '{self.embedding_model_name}' produces: {embedding_dim} dimensions")
                    
                    # Detect collection dimension if not in metadata
                    if not self.collection_dimension:
                        await self._detect_collection_dimension()
                    
                    # Check for dimension mismatch
                    if self.collection_dimension and embedding_dim != self.collection_dimension:
                        logger.error(f"❌ DIMENSION MISMATCH!")
                        logger.error(f"   Collection expects: {self.collection_dimension} dimensions")
                        logger.error(f"   Ollama model '{self.embedding_model_name}' produces: {embedding_dim} dimensions")
                        return False
                        
                except Exception as e:
                    logger.error(f"Failed to test Ollama embedding: {e}")
                    return False
                    
            else:  # huggingface
                logger.info(f"Loading HuggingFace embedding model: {self.embedding_model_name}")
                self.embedding_model = SentenceTransformer(self.embedding_model_name)
                
                # Test embedding dimension
                try:
                    test_embedding = self.embedding_model.encode("test", convert_to_numpy=True)
                    embedding_dim = len(test_embedding)
                    logger.info(f"   HuggingFace model produces: {embedding_dim} dimensions")
                    
                    # Detect collection dimension if not in metadata
                    if not self.collection_dimension:
                        await self._detect_collection_dimension()
                    
                    # Check for dimension mismatch
                    if self.collection_dimension and embedding_dim != self.collection_dimension:
                        logger.error(f"❌ DIMENSION MISMATCH!")
                        logger.error(f"   Collection expects: {self.collection_dimension} dimensions")
                        logger.error(f"   Embedding model '{self.embedding_model_name}' produces: {embedding_dim} dimensions")
                        return False
                        
                except Exception as e:
                    logger.warning(f"Could not test embedding dimension: {e}")
            
            logger.info(f"✅ RAG retriever initialized for collection: {self.collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize RAG retriever: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return False
    
    async def _detect_collection_dimension(self):
        """Detect collection dimension by fetching a sample embedding"""
        try:
            logger.info("   Attempting to detect collection dimension...")
            get_url = f"{self.chroma_url}/api/v2/tenants/default_tenant/databases/default_database/collections/{self.collection_id}/get"
            get_response = await self.http_client.post(
                get_url,
                json={"limit": 1, "include": ["embeddings"]}
            )
            if get_response.status_code == 200:
                get_data = get_response.json()
                if get_data.get("embeddings") and len(get_data["embeddings"]) > 0:
                    stored_embedding = get_data["embeddings"][0]
                    self.collection_dimension = len(stored_embedding)
                    logger.info(f"   Detected collection dimension: {self.collection_dimension}")
        except Exception as e:
            logger.warning(f"Could not detect collection dimension: {e}")
    
    async def get_relevant_documents(self, query: str, k: int = 7) -> List[Document]:
        """Retrieve relevant documents for a query"""
        try:
            if not self.collection_id:
                logger.error("Collection not initialized")
                return []
            
            # Generate query embedding
            logger.info(f"Generating embedding for query with {self.embedding_model_name} ({self.embedding_source})")
            
            if self.embedding_source == "ollama":
                query_embedding = await asyncio.to_thread(
                    self.embedding_model.embed_query,
                    query
                )
            else:  # huggingface
                query_embedding = await asyncio.to_thread(
                    self.embedding_model.encode,
                    query,
                    convert_to_numpy=True
                )
                query_embedding = query_embedding.tolist()
            
            # Verify dimension matches
            if self.collection_dimension and len(query_embedding) != self.collection_dimension:
                logger.error(f"❌ Embedding dimension mismatch!")
                logger.error(f"   Query embedding: {len(query_embedding)} dimensions")
                logger.error(f"   Collection expects: {self.collection_dimension} dimensions")
                return []
            
            # Query ChromaDB
            query_url = f"{self.chroma_url}/api/v2/tenants/default_tenant/databases/default_database/collections/{self.collection_id}/query"
            
            query_payload = {
                "query_embeddings": [query_embedding],
                "n_results": k,
                "include": ["documents", "metadatas", "distances"]
            }
            
            logger.info(f"Querying ChromaDB for top {k} documents (embedding dim: {len(query_embedding)})")
            
            response = await self.http_client.post(
                query_url, 
                json=query_payload,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code != 200:
                logger.error(f"❌ ChromaDB query failed with status {response.status_code}")
                logger.error(f"   Response: {response.text[:500]}")
                response.raise_for_status()
            
            results = response.json()
            
            # Convert to LangChain Document format
            documents = []
            if results and "documents" in results and len(results["documents"]) > 0:
                docs = results["documents"][0]
                metadatas = results.get("metadatas", [[]])[0]
                distances = results.get("distances", [[]])[0]
                
                for i, doc_text in enumerate(docs):
                    if not doc_text:
                        continue
                        
                    metadata = metadatas[i] if i < len(metadatas) else {}
                    distance = distances[i] if i < len(distances) else 0.0
                    
                    metadata["similarity_score"] = round(1.0 - distance, 3)
                    
                    documents.append(Document(
                        page_content=doc_text,
                        metadata=metadata
                    ))
            
            logger.info(f"✅ Retrieved {len(documents)} documents for query")
            return documents
            
        except httpx.HTTPStatusError as e:
            logger.error(f"ChromaDB HTTP {e.response.status_code}: {e.response.text[:500]}")
            return []
        except Exception as e:
            logger.error(f"RAG retrieval error: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return []
    
    async def close(self):
        """Close HTTP client"""
        await self.http_client.aclose()

class Agent:
    """Individual agent with RAG capabilities and cancellation support"""
    def __init__(self, agent_id: int, role: str, model, tokenizer, device, retriever: Optional[RAGRetriever] = None):
        self.agent_id = agent_id
        self.role = role
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.retriever = retriever
        self.response = None
        self.confidence = 0.0
        self.rag_docs_used = 0
        self.task_id = None
    
    async def process_query(self, query: str, context: str, strategy: str, task_id: str = None, custom_context_size: Optional[int] = None) -> Dict:
        """Process query with optional RAG support - WITH PROPER CANCELLATION"""
        self.task_id = task_id or f"agent_{self.agent_id}_{id(self)}"
        
        try:
            # Check if task is cancelled at the start
            if cancellation_tracker.is_cancelled(self.task_id):
                logger.warning(f"Agent {self.agent_id} detected cancellation at start")
                raise asyncio.CancelledError()
            
            # Retrieve relevant documents if RAG is enabled
            rag_context = ""
            retrieved_docs = []
            
            if self.retriever:
                if cancellation_tracker.is_cancelled(self.task_id):
                    logger.warning(f"Agent {self.agent_id} cancelled before RAG")
                    raise asyncio.CancelledError()
                
                try:
                    if self.role in ["Analyzer", "Researcher", "Expert"]:
                        k = 15
                    elif self.role in ["Synthesizer", "Manager", "Reviewer"]:
                        k = 10
                    else:
                        k = 12
                    
                    retrieved_docs = await self.retriever.get_relevant_documents(query, k=k)
                    self.rag_docs_used = len(retrieved_docs)
                    
                    if retrieved_docs:
                        doc_summaries = []
                        total_chars = 0
                        max_chars = 15000
                        
                        for i, doc in enumerate(retrieved_docs):
                            content = doc.page_content[:1200]
                            similarity = doc.metadata.get('similarity_score', 0.0)
                            doc_summary = f"[Source {i+1}] (Relevance: {similarity:.2f})\n{content}"
                            
                            if total_chars + len(doc_summary) > max_chars:
                                break
                            
                            doc_summaries.append(doc_summary)
                            total_chars += len(doc_summary)
                        
                        rag_context = "\n\n".join(doc_summaries)
                        logger.info(f"Agent {self.agent_id} ({self.role}) using {len(doc_summaries)} docs ({total_chars} chars)")
                    
                except Exception as e:
                    logger.error(f"RAG retrieval error for Agent {self.agent_id}: {e}")
            
            if cancellation_tracker.is_cancelled(self.task_id):
                logger.warning(f"Agent {self.agent_id} cancelled before prompt building")
                raise asyncio.CancelledError()
            
            # Build prompts (same as before)
            if rag_context:
                prompt = f"""Based on the following knowledge base sources, provide a comprehensive answer to the question.
    
    Knowledge Base:
    {rag_context}
    
    Additional Context: {context if context else "None provided"}
    
    Question: {query}
    
    Role: You are {self.role}. Analyze the sources above and provide a detailed, well-structured answer.
    
    Answer:"""
            else:
                if context and len(context) > 50:
                    prompt = f"""Using the information provided below, answer the question comprehensively.
    
    Information:
    {context}
    
    Question: {query}
    
    Role: You are {self.role}. Provide a detailed, well-structured answer.
    
    Answer:"""
                else:
                    prompt = f"""Question: {query}
    
    Role: You are {self.role}. Provide a detailed, comprehensive answer based on your knowledge.
    
    Answer:"""
            
            model_name = getattr(self.model.config, '_name_or_path', 'unknown')
            
            # Use custom context size if provided
            max_context = get_model_context_window(model_name, custom_context_size)
            max_input_length = int(max_context * 0.9)
            
            # Improved logging
            context_info = f"Agent {self.agent_id} using context window: {max_context} tokens (max input: {max_input_length})"
            if custom_context_size:
                context_info += f" [CUSTOM: {custom_context_size}]"
            else:
                context_info += f" [AUTO]"
            logger.info(context_info)
                        
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=max_input_length,
                padding=True
            )
            
            input_length = inputs['input_ids'].shape[1]
            logger.info(f"Agent {self.agent_id} prompt tokenized: {input_length} tokens (of {max_input_length} max)")
            
            if self.device != "cpu":
                try:
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                except Exception as e:
                    logger.warning(f"Failed to move inputs to {self.device}: {e}")
            
            remaining_context = max_context - input_length
            
            if rag_context or (context and len(context) > 50):
                max_new_tokens = min(800, int(remaining_context * 0.8))
                min_new_tokens = 150
            else:
                max_new_tokens = min(500, int(remaining_context * 0.8))
                min_new_tokens = 100
            
            logger.info(f"Agent {self.agent_id} generation: min={min_new_tokens}, max={max_new_tokens} tokens")
            
            if cancellation_tracker.is_cancelled(self.task_id):
                logger.warning(f"Agent {self.agent_id} cancelled before model.generate()")
                raise asyncio.CancelledError()
            
            # NEW: Add cancellation stopping criteria
            stopping_criteria = StoppingCriteriaList([
                CancellationStoppingCriteria(self.task_id)
            ])
            
            generation_kwargs = {
                "max_new_tokens": max_new_tokens,
                "min_new_tokens": min_new_tokens,
                "temperature": 0.7,
                "do_sample": True,
                "top_p": 0.9,
                "top_k": 40,
                "repetition_penalty": 1.15,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
                "no_repeat_ngram_size": 3,
                "stopping_criteria": stopping_criteria,
            }
            
            if self.device == "mps":
                generation_kwargs.update({
                    "max_new_tokens": max(200, max_new_tokens - 100),
                    "min_new_tokens": 80,
                })
            
            logger.info(f"Agent {self.agent_id} generating with cancellation support (task_id: {self.task_id})...")
            
            # NEW: Run generation in thread pool with periodic cancellation checks
            def run_generation():
                """Run generation in thread - checking cancellation every 0.1s"""
                with torch.no_grad():
                    return self.model.generate(**inputs, **generation_kwargs)
            
            # Create generation task
            generation_task = asyncio.create_task(
                asyncio.to_thread(run_generation)
            )
            
            # Poll for cancellation while generation runs
            while not generation_task.done():
                if cancellation_tracker.is_cancelled(self.task_id):
                    logger.warning(f"🛑 Agent {self.agent_id} cancellation detected during generation!")
                    generation_task.cancel()
                    try:
                        await generation_task
                    except asyncio.CancelledError:
                        logger.info(f"✅ Agent {self.agent_id} generation task cancelled")
                    raise asyncio.CancelledError()
                
                # Check every 100ms
                await asyncio.sleep(0.1)
            
            # Get result
            try:
                outputs = await generation_task
            except asyncio.CancelledError:
                logger.warning(f"Agent {self.agent_id} generation was cancelled")
                raise
            
            # Check if generation was cancelled
            if cancellation_tracker.is_cancelled(self.task_id):
                logger.warning(f"Agent {self.agent_id} generation was cancelled")
                raise asyncio.CancelledError()
            
            generated_length = outputs.shape[1] - input_length
            logger.info(f"Agent {self.agent_id} generated {generated_length} new tokens")
            
            # Better extraction - decode ONLY the new tokens
            generated_part = None
            
            try:
                new_tokens = outputs[0][input_length:]
                generated_part = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
                logger.info(f"Agent {self.agent_id} extracted new tokens directly")
            except Exception as e:
                logger.warning(f"Agent {self.agent_id} failed to extract new tokens: {e}")
            
            if not generated_part or len(generated_part.strip()) < 20:
                full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                if "Answer:" in full_response:
                    parts = full_response.split("Answer:")
                    if len(parts) > 1:
                        generated_part = parts[-1].strip()
                        logger.info(f"Agent {self.agent_id} extracted after 'Answer:' marker")
            
            if not generated_part or len(generated_part.strip()) < 20:
                full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                prompt_text = self.tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True)
                if prompt_text in full_response:
                    prompt_end_index = full_response.find(prompt_text) + len(prompt_text)
                    generated_part = full_response[prompt_end_index:].strip()
                    logger.info(f"Agent {self.agent_id} removed exact prompt text")
            
            if not generated_part or len(generated_part.strip()) < 20:
                full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                cutoff = int(len(full_response) * 0.3)
                generated_part = full_response[cutoff:].strip()
                logger.warning(f"Agent {self.agent_id} using fallback extraction (last 70%)")
            
            generated_part = generated_part.strip()
            
            cleanup_patterns = [
                "Knowledge Base:",
                "Additional Context:",
                "Information:",
                "Question:",
                "Role: You are",
                f"Role: {self.role}",
                "Based on the following",
                "Using the information provided",
                "Provide a detailed"
            ]
            
            for pattern in cleanup_patterns:
                if generated_part.startswith(pattern):
                    lines = generated_part.split('\n')
                    clean_lines = []
                    skip_lines = True
                    
                    for line in lines:
                        if skip_lines and line.strip() and not any(p in line for p in cleanup_patterns):
                            skip_lines = False
                        
                        if not skip_lines:
                            clean_lines.append(line)
                    
                    if clean_lines:
                        generated_part = '\n'.join(clean_lines).strip()
                        logger.info(f"Agent {self.agent_id} removed prompt fragments")
                    break
            
            word_count = len(generated_part.split())
            sentence_count = len([s for s in generated_part.split('.') if s.strip()])
            
            logger.info(f"Agent {self.agent_id} final: {word_count} words, {sentence_count} sentences")
            
            base_confidence = min(100, word_count * 1.2 + 50)
            rag_boost = 15 if rag_context else 0
            quality_bonus = min(20, sentence_count * 3)
            length_penalty = -10 if word_count < 20 else 0
            confidence = min(100, max(30, base_confidence + rag_boost + quality_bonus + length_penalty))
            
            self.response = generated_part
            self.confidence = int(confidence)
            
            logger.info(f"✅ Agent {self.agent_id} complete: {word_count} words, {confidence}% confidence")
            
            return {
                "agent_id": self.agent_id,
                "agent_name": f"Agent {self.agent_id} ({self.role})",
                "response": generated_part,
                "confidence": int(confidence),
                "reasoning": f"Analysis from {self.role} perspective" + 
                        (f" using {self.rag_docs_used} retrieved documents" if self.retriever else ""),
                "rag_docs_used": self.rag_docs_used,
                "rag_enabled": bool(self.retriever),
                "word_count": word_count,
                "retrieved_sources": [
                    {
                        "content": doc.page_content[:300] + "...",
                        "metadata": doc.metadata
                    }
                    for doc in retrieved_docs[:5]
                ] if retrieved_docs else []
            }
            
        except asyncio.CancelledError:
            logger.warning(f"🛑 Agent {self.agent_id} processing cancelled (task_id: {self.task_id})")
            cancellation_tracker.remove(self.task_id)
            raise
        except Exception as e:
            logger.error(f"Agent {self.agent_id} processing error: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {
                "agent_id": self.agent_id,
                "agent_name": f"Agent {self.agent_id} ({self.role})",
                "response": f"Error processing query: {str(e)}",
                "confidence": 0,
                "reasoning": "Processing failed",
                "rag_docs_used": 0,
                "word_count": 0
            }
        finally:
            if self.task_id:
                cancellation_tracker.remove(self.task_id)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='MultiAgent System Server with RAG')
    parser.add_argument('--port', type=int, default=DEFAULT_API_PORT,
                       help=f'API server port (default: {DEFAULT_API_PORT})')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                       help='API server host (default: 0.0.0.0)')
    # NEW: Cache location arguments
    parser.add_argument('--cache-dir', type=str, default=None,
                       help='Directory for model cache (default: ~/.cache)')
    parser.add_argument('--temp-dir', type=str, default=None,
                       help='Directory for temporary files (default: system temp)')
    return parser.parse_args()

def get_optimal_device():
    """Get the best available device"""
    if torch.backends.mps.is_available():
        logger.info("MPS (Metal Performance Shaders) is available - using Apple Silicon GPU")
        return "mps"
    elif torch.cuda.is_available():
        logger.info(f"CUDA is available - using GPU: {torch.cuda.get_device_name()}")
        return "cuda"
    else:
        logger.info("Using CPU (no GPU acceleration available)")
        return "cpu"

def get_model_name_from_key(model_key):
    """Map UI model keys to actual model names"""
    return MODEL_MAPPING.get(model_key, "Salesforce/codegen-2B-nl")

def load_llm_model(model_name="Salesforce/codegen-2B-nl"):
    """Load LLM model with enhanced error handling"""
    global llm_model, llm_tokenizer, llm_device, current_model_name
    
    if llm_model is not None and current_model_name == model_name:
        logger.info(f"Model already loaded: {model_name}")
        return llm_model, llm_tokenizer
    
    logger.info(f"🔄 Loading LLM model: {model_name}")
    
    device = get_optimal_device()
    llm_device = device
    
    # Model-specific configurations
    model_configs = {
        "mistralai/Mistral-7B-Instruct-v0.2": {
            "use_fast": False,
            "trust_remote_code": True,
            "torch_dtype": torch.float32 if device == "cpu" else torch.float16,
            "low_cpu_mem_usage": True,
        },
        "mistralai/Mistral-7B-Instruct-v0.3": {
            "use_fast": False,
            "trust_remote_code": True,
            "torch_dtype": torch.float32 if device == "cpu" else torch.float16,
            "low_cpu_mem_usage": True,
        },
        "meta-llama/Llama-3.1-8B-Instruct": {
            "use_fast": True,
            "trust_remote_code": True,
            "torch_dtype": torch.float32 if device == "cpu" else torch.float16,
            "low_cpu_mem_usage": True,
        },
        "meta-llama/Llama-3.2-3B-Instruct": {
            "use_fast": True,
            "trust_remote_code": True,
            "torch_dtype": torch.float32 if device == "cpu" else torch.float16,
            "low_cpu_mem_usage": True,
        },
        "meta-llama/Llama-3.2-1B-Instruct": {
            "use_fast": True,
            "trust_remote_code": True,
            "torch_dtype": torch.float32 if device == "cpu" else torch.float16,
            "low_cpu_mem_usage": False,
        },
        "ibm-granite/granite-3.3-8b-instruct": {
            "use_fast": False,
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
        },
    }
    
    config = model_configs.get(model_name, {
        "use_fast": True,
        "trust_remote_code": True,
        "torch_dtype": torch.float32 if device == "cpu" else torch.float16,
    })
    
    try:
        logger.info(f"Loading tokenizer for {model_name}...")
        
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                use_fast=config.get("use_fast", True),
                trust_remote_code=config.get("trust_remote_code", True),
                padding_side='left',
            )
            logger.info(f"✅ Loaded tokenizer (fast={config.get('use_fast', True)})")
        except Exception as fast_error:
            logger.warning(f"Fast tokenizer failed: {fast_error}")
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                use_fast=False,
                trust_remote_code=True,
                padding_side='left',
            )
            logger.info("✅ Loaded slow tokenizer")
        
        if tokenizer.pad_token is None:
            if tokenizer.eos_token:
                tokenizer.pad_token = tokenizer.eos_token
                logger.info(f"Set pad_token to eos_token: {tokenizer.eos_token}")
            else:
                tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                logger.info("Added custom [PAD] token")
        
        torch_dtype = config.get("torch_dtype", torch.float32)
        logger.info(f"Loading model with dtype: {torch_dtype} on device: {device}")
        
        model_kwargs = {
            "torch_dtype": torch_dtype,
            "trust_remote_code": config.get("trust_remote_code", True),
            "device_map": None,
        }
        
        if device == "cpu" and config.get("low_cpu_mem_usage", False):
            model_kwargs["low_cpu_mem_usage"] = True
        
        if "mistral" in model_name.lower():
            model_kwargs.update({"use_cache": True})
        
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        
        logger.info(f"Moving model to {device}...")
        model.to(device)
        model.eval()
        
        if tokenizer.pad_token != tokenizer.eos_token:
            model.resize_token_embeddings(len(tokenizer))
        
        llm_model = model
        llm_tokenizer = tokenizer
        current_model_name = model_name
        
        param_count = sum(p.numel() for p in model.parameters()) / 1e6
        logger.info(f"✅ Successfully loaded: {model_name}")
        logger.info(f"   Device: {device}, Parameters: ~{param_count:.1f}M")
        
        return model, tokenizer
        
    except Exception as e:
        logger.error(f"❌ Failed to load model {model_name}: {e}")
        fallback_models = [
            "mistralai/Mistral-7B-Instruct-v0.3",
            "Salesforce/codegen-2B-nl",
            "distilgpt2",
        ]
        
        for fallback in fallback_models:
            if fallback != model_name:
                logger.info(f"🔄 Attempting fallback to {fallback}...")
                try:
                    return load_llm_model(fallback)
                except Exception as fallback_error:
                    continue
        
        raise RuntimeError(f"Failed to load any model. Last error: {e}")

def create_agents(num_agents: int, strategy: str, model, tokenizer, device, 
                  retriever: Optional[RAGRetriever] = None) -> List[Agent]:
    """Create multiple agents with optional RAG support"""
    agents = []
    
    if strategy == "collaborative":
        roles = ["Analyzer", "Critic", "Synthesizer", "Validator", "Reviewer", 
                "Expert", "Specialist", "Researcher", "Consultant", "Evaluator"]
    elif strategy == "sequential":
        roles = ["Analyzer", "Processor", "Reviewer", "Synthesizer", "Validator", 
                "Finalizer", "Checker", "Refiner", "Enhancer", "Completer"]
    elif strategy == "competitive":
        roles = ["Expert A", "Expert B", "Expert C", "Specialist A", "Specialist B", 
                "Analyst A", "Analyst B", "Researcher A", "Researcher B", "Authority"]
    elif strategy == "hierarchical":
        roles = ["Manager"] + [f"Specialist {i}" for i in range(1, 10)]
    else:
        roles = [f"Agent {i}" for i in range(1, 11)]
    
    for i in range(num_agents):
        role = roles[i % len(roles)]
        agent = Agent(i + 1, role, model, tokenizer, device, retriever)
        agents.append(agent)
    
    return agents

# Update strategy functions to track consensus metrics
# Update strategy functions to track consensus metrics
async def collaborative_strategy(agents: List[Agent], query: str, context: str, 
                                websocket: WebSocket = None, task_id: str = None, 
                                custom_context_size: Optional[int] = None,
                                _metrics: Dict = None, _agents_seen: set = None,
                                tokenizer = None) -> Dict:
    """Collaborative strategy with metrics tracking and safe WebSocket handling"""
    logger.info("Executing collaborative strategy")
    
    # Safe send helper
    async def safe_ws_send(message_dict):
        if websocket:
            try:
                await websocket.send_json(message_dict)
                return True
            except Exception as e:
                logger.warning(f"WebSocket send failed: {e}")
                return False
        return False
    
    await safe_ws_send({
        "type": "progress",
        "progress": 45,
        "message": "Agents retrieving documents from knowledge base..."
    })
    
    if task_id and cancellation_tracker.is_cancelled(task_id):
        logger.warning("Collaborative strategy cancelled before agent tasks")
        raise asyncio.CancelledError()
    
    # Pass custom_context_size to each agent
    tasks = [agent.process_query(query, context, "collaborative", f"{task_id}_agent{agent.agent_id}", custom_context_size) 
             for agent in agents]
    
    await safe_ws_send({
        "type": "progress",
        "progress": 55,
        "message": "Agents generating responses..."
    })
    
    try:
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        valid_responses = [r for r in responses if not isinstance(r, (asyncio.CancelledError, Exception))]
        
        if not valid_responses:
            logger.warning("All agent tasks were cancelled or failed")
            raise asyncio.CancelledError()
            
    except asyncio.CancelledError:
        logger.warning("🛑 Agent tasks cancelled in collaborative strategy")
        for task in tasks:
            if not task.done():
                task.cancel()
        raise
    
    logger.info(f"All {len(valid_responses)} agents completed processing")
    
    # Track consensus metrics
    _t_cons = _now_ms()
    if _metrics:
        _metrics["consensus_rounds"] += 1
        _metrics["consensus_method"] = "collaborative"
        
        # Calculate disagreement rate based on response similarity
        if valid_responses and len(valid_responses) > 1:
            unique_responses = set()
            for r in valid_responses:
                response_text = r.get("response", "").strip()
                if response_text:
                    # Use first 100 chars as signature for uniqueness check
                    unique_responses.add(response_text[:100])
            
            if len(unique_responses) > 1:
                _metrics["disagreement_rate"] = 1.0 - (1.0 / len(unique_responses))
            else:
                _metrics["disagreement_rate"] = 0.0
        else:
            _metrics["disagreement_rate"] = 0.0
        
        _metrics["consensus_ms"] += _now_ms() - _t_cons
    
    # Track participating agents
    if _agents_seen is not None:
        for agent in agents:
            _agents_seen.add(f"agent_{agent.agent_id}")
    
    # Track completion tokens for each agent
    if tokenizer and _metrics:
        for r in valid_responses:
            _metrics["completion_tokens_total"] += _count_tokens_approx(r.get("response", ""), tokenizer)
    
    formatted_responses = []
    for r in valid_responses:
        agent_header = f"**{r['agent_name']}** (Confidence: {r['confidence']}%)"
        if r.get('rag_docs_used', 0) > 0:
            agent_header += f" [Used {r['rag_docs_used']} docs]"
        formatted_responses.append(f"{agent_header}\n{r['response']}")
    
    all_responses = "\n\n---\n\n".join(formatted_responses)
    avg_confidence = sum(r["confidence"] for r in valid_responses) / len(valid_responses)
    total_words = sum(r.get("word_count", 0) for r in valid_responses)
    
    final_answer = f"""# Collaborative Multi-Agent Analysis

{all_responses}

---

## Consensus Summary
Based on {len(agents)} agents working collaboratively:
- Average Confidence: {avg_confidence:.1f}%
- Total Analysis: {total_words} words
- Strategy: Diverse perspectives combined for comprehensive analysis
"""
    
    return {
        "final_answer": final_answer,
        "agent_responses": valid_responses,
        "consensus_score": int(avg_confidence),
        "strategy": "Collaborative"
    }

async def sequential_strategy(agents: List[Agent], query: str, context: str, 
                             websocket: WebSocket = None, 
                             custom_context_size: Optional[int] = None,
                             _metrics: Dict = None, _agents_seen: set = None,
                             tokenizer = None) -> Dict:
    """Sequential strategy with metrics tracking and safe WebSocket handling"""
    logger.info("Executing sequential strategy")
    
    # Safe send helper
    async def safe_ws_send(message_dict):
        if websocket:
            try:
                await websocket.send_json(message_dict)
                return True
            except Exception as e:
                logger.warning(f"WebSocket send failed: {e}")
                return False
        return False
    
    responses = []
    accumulated_context = context
    
    for idx, agent in enumerate(agents):
        await safe_ws_send({
            "type": "progress",
            "progress": min(45 + (idx * 10), 70),
            "message": f"Agent {agent.agent_id} ({agent.role}) processing..."
        })
        
        # Calculate dynamic context size for each agent
        if custom_context_size:
            scale_factor = 1.0 + (idx * 0.5)
            agent_context_size = int(custom_context_size * min(scale_factor, 2.0))
            
            model_name = getattr(agent.model.config, '_name_or_path', 'unknown')
            max_model_size = MODEL_CONTEXT_WINDOWS.get(model_name, 4096)
            agent_context_size = min(agent_context_size, max_model_size)
            
            logger.info(f"Agent {agent.agent_id} sequential context: {agent_context_size} tokens (scaled by {scale_factor:.1f}x)")
        else:
            accumulated_tokens = estimateTokens(accumulated_context) + estimateTokens(query)
            
            if accumulated_tokens < 2000:
                agent_context_size = 4096
            elif accumulated_tokens < 5000:
                agent_context_size = 8192
            elif accumulated_tokens < 10000:
                agent_context_size = 16384
            elif accumulated_tokens < 20000:
                agent_context_size = 32768
            else:
                agent_context_size = 65536
            
            logger.info(f"Agent {agent.agent_id} auto-sized context: {agent_context_size} tokens (accumulated: ~{accumulated_tokens} tokens)")
        
        response = await agent.process_query(query, accumulated_context, "sequential", None, agent_context_size)
        responses.append(response)
        
        # Track participating agents
        if _agents_seen is not None:
            _agents_seen.add(f"agent_{agent.agent_id}")
        
        # Track completion tokens
        if tokenizer and _metrics:
            _metrics["completion_tokens_total"] += _count_tokens_approx(response.get("response", ""), tokenizer)
        
        # Add previous agent's response to accumulated context
        accumulated_context += f"\n\n[Previous Agent {agent.agent_id}]: {response['response'][:500]}"
    
    # Track consensus metrics
    _t_cons = _now_ms()
    if _metrics:
        _metrics["consensus_rounds"] = len(agents)  # Each agent builds on previous
        _metrics["consensus_method"] = "sequential"
        _metrics["turns_total"] = len(agents)
        
        # Sequential has no disagreement as each builds on previous
        _metrics["disagreement_rate"] = 0.0
        _metrics["consensus_ms"] += _now_ms() - _t_cons
    
    final_answer = responses[-1]["response"]
    avg_confidence = sum(r["confidence"] for r in responses) / len(responses)
    
    return {
        "final_answer": final_answer,
        "agent_responses": responses,
        "consensus_score": int(avg_confidence),
        "strategy": "Sequential"
    }

async def competitive_strategy(agents: List[Agent], query: str, context: str, 
                              websocket: WebSocket = None, 
                              custom_context_size: Optional[int] = None,
                              _metrics: Dict = None, _agents_seen: set = None,
                              tokenizer = None) -> Dict:
    """Competitive strategy with metrics tracking and safe WebSocket handling"""
    logger.info("Executing competitive strategy")
    
    # Safe send helper
    async def safe_ws_send(message_dict):
        if websocket:
            try:
                await websocket.send_json(message_dict)
                return True
            except Exception as e:
                logger.warning(f"WebSocket send failed: {e}")
                return False
        return False
    
    await safe_ws_send({
        "type": "progress",
        "progress": 50,
        "message": "Agents competing to provide best answer..."
    })
    
    tasks = [agent.process_query(query, context, "competitive", None, custom_context_size) for agent in agents]
    
    try:
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        valid_responses = [r for r in responses if not isinstance(r, (asyncio.CancelledError, Exception))]
        
        if not valid_responses:
            logger.warning("All agent tasks were cancelled or failed in competitive strategy")
            raise asyncio.CancelledError()
    except asyncio.CancelledError:
        logger.warning("🛑 Agent tasks cancelled in competitive strategy")
        for task in tasks:
            if not task.done():
                task.cancel()
        raise
    
    # Track participating agents
    if _agents_seen is not None:
        for agent in agents:
            _agents_seen.add(f"agent_{agent.agent_id}")
    
    # Track completion tokens
    if tokenizer and _metrics:
        for r in valid_responses:
            _metrics["completion_tokens_total"] += _count_tokens_approx(r.get("response", ""), tokenizer)
    
    # Track consensus metrics
    _t_cons = _now_ms()
    if _metrics:
        _metrics["consensus_rounds"] += 1
        _metrics["consensus_method"] = "competitive"
        
        # Calculate disagreement by confidence variance
        confidences = [r["confidence"] for r in valid_responses]
        if len(confidences) > 1:
            avg_conf = sum(confidences) / len(confidences)
            variance = sum((c - avg_conf) ** 2 for c in confidences) / len(confidences)
            # Normalize variance to 0-1 range (max variance is 2500 for 0-100 range)
            _metrics["disagreement_rate"] = min(1.0, variance / 2500.0)
        else:
            _metrics["disagreement_rate"] = 0.0
        
        _metrics["consensus_ms"] += _now_ms() - _t_cons
    
    best_response = max(valid_responses, key=lambda x: x["confidence"])
    
    return {
        "final_answer": best_response["response"],
        "agent_responses": valid_responses,
        "consensus_score": best_response["confidence"],
        "strategy": "Competitive"
    }

async def hierarchical_strategy(agents: List[Agent], query: str, context: str, 
                               websocket: WebSocket = None, 
                               custom_context_size: Optional[int] = None,
                               _metrics: Dict = None, _agents_seen: set = None,
                               tokenizer = None) -> Dict:
    """Hierarchical strategy with metrics tracking and safe WebSocket handling"""
    logger.info("Executing hierarchical strategy")
    
    # Safe send helper
    async def safe_ws_send(message_dict):
        if websocket:
            try:
                await websocket.send_json(message_dict)
                return True
            except Exception as e:
                logger.warning(f"WebSocket send failed: {e}")
                return False
        return False
    
    specialists = agents[:-1]
    manager = agents[-1]
    
    await safe_ws_send({
        "type": "progress",
        "progress": 50,
        "message": "Specialist agents analyzing query..."
    })
    
    specialist_tasks = [agent.process_query(query, context, "hierarchical", None, custom_context_size) for agent in specialists]
    
    try:
        specialist_responses = await asyncio.gather(*specialist_tasks, return_exceptions=True)
        valid_specialist_responses = [r for r in specialist_responses if not isinstance(r, (asyncio.CancelledError, Exception))]
        
        if not valid_specialist_responses:
            logger.warning("All specialist tasks were cancelled or failed")
            raise asyncio.CancelledError()
    except asyncio.CancelledError:
        logger.warning("🛑 Specialist tasks cancelled in hierarchical strategy")
        for task in specialist_tasks:
            if not task.done():
                task.cancel()
        raise
    
    # Track specialist agents
    if _agents_seen is not None:
        for agent in specialists:
            _agents_seen.add(f"agent_{agent.agent_id}")
    
    # Track specialist completion tokens
    if tokenizer and _metrics:
        for r in valid_specialist_responses:
            _metrics["completion_tokens_total"] += _count_tokens_approx(r.get("response", ""), tokenizer)
    
    await safe_ws_send({
        "type": "progress",
        "progress": 65,
        "message": "Manager synthesizing specialist insights..."
    })
    
    # Build manager context
    specialist_summaries = [f"[{r['agent_name']}]: {r['response'][:400]}" 
                           for r in valid_specialist_responses]
    manager_context = context + "\n\nSpecialist Inputs:\n" + "\n\n".join(specialist_summaries)
    
    # Manager needs larger context
    if custom_context_size:
        manager_context_size = int(custom_context_size * 1.5)
        model_name = getattr(manager.model.config, '_name_or_path', 'unknown')
        max_model_size = MODEL_CONTEXT_WINDOWS.get(model_name, 4096)
        manager_context_size = min(manager_context_size, max_model_size)
        logger.info(f"Manager using expanded context: {manager_context_size} tokens (1.5x specialist)")
    else:
        accumulated_tokens = estimateTokens(manager_context) + estimateTokens(query)
        if accumulated_tokens < 5000:
            manager_context_size = 8192
        elif accumulated_tokens < 10000:
            manager_context_size = 16384
        elif accumulated_tokens < 20000:
            manager_context_size = 32768
        else:
            manager_context_size = 65536
        logger.info(f"Manager auto-sized context: {manager_context_size} tokens")
    
    manager_response = await manager.process_query(query, manager_context, "hierarchical", None, manager_context_size)
    
    # Track manager agent
    if _agents_seen is not None:
        _agents_seen.add(f"agent_{manager.agent_id}")
    
    # Track manager completion tokens
    if tokenizer and _metrics:
        _metrics["completion_tokens_total"] += _count_tokens_approx(manager_response.get("response", ""), tokenizer)
    
    # Track consensus metrics
    _t_cons = _now_ms()
    if _metrics:
        _metrics["consensus_rounds"] = 2  # Specialists + Manager synthesis
        _metrics["consensus_method"] = "hierarchical"
        _metrics["turns_total"] = 2
        
        # Calculate specialist disagreement
        if len(valid_specialist_responses) > 1:
            unique_responses = set()
            for r in valid_specialist_responses:
                response_text = r.get("response", "").strip()
                if response_text:
                    unique_responses.add(response_text[:100])
            
            if len(unique_responses) > 1:
                _metrics["disagreement_rate"] = 1.0 - (1.0 / len(unique_responses))
            else:
                _metrics["disagreement_rate"] = 0.0
        else:
            _metrics["disagreement_rate"] = 0.0
        
        _metrics["consensus_ms"] += _now_ms() - _t_cons
    
    all_responses = valid_specialist_responses + [manager_response]
    
    return {
        "final_answer": manager_response["response"],
        "agent_responses": all_responses,
        "consensus_score": manager_response["confidence"],
        "strategy": "Hierarchical"
    }

def calculate_metrics(responses: List[Dict], final_answer: str) -> Dict:
    """Calculate performance metrics"""
    avg_confidence = sum(r["confidence"] for r in responses) / len(responses) if responses else 50
    avg_word_count = sum(r.get("word_count", 0) for r in responses) / len(responses) if responses else 0
    
    quality_score = min(10, int(avg_confidence / 10))
    agreement_rate = int(avg_confidence * 0.8)
    coherence = min(100, int(avg_word_count * 2 + 50))
    coverage = min(100, int(len(responses) * 15 + 40))
    
    return {
        "agreement_rate": agreement_rate,
        "quality_score": quality_score,
        "coherence": coherence,
        "coverage": coverage,
        "avg_response_length": int(avg_word_count)
    }

def analyze_strategy(strategy: str, metrics: Dict) -> Dict:
    """Analyze strategy performance"""
    strategies_info = {
        "Collaborative": {
            "strengths": "Diverse perspectives, comprehensive coverage, balanced consensus",
            "weaknesses": "May be slower due to parallel processing overhead",
            "recommendations": "Best for complex queries requiring multiple viewpoints"
        },
        "Sequential": {
            "strengths": "Builds on previous insights, progressive refinement",
            "weaknesses": "Earlier agent errors can propagate",
            "recommendations": "Good for step-by-step problem solving"
        },
        "Competitive": {
            "strengths": "Selects highest confidence answer, quality-focused",
            "weaknesses": "Ignores alternative perspectives",
            "recommendations": "Best when accuracy is critical"
        },
        "Hierarchical": {
            "strengths": "Organized structure, clear coordination",
            "weaknesses": "Manager bottleneck, dependent on final synthesis",
            "recommendations": "Good for complex tasks requiring oversight"
        }
    }
    
    return strategies_info.get(strategy, {
        "strengths": "Standard approach",
        "weaknesses": "No specific optimizations",
        "recommendations": "Consider specialized strategies for better results"
    })

@app.get("/chromadb/collections")
async def get_chromadb_collections(chroma_url: str = "http://127.0.0.1:8000"):
    """Fetch available ChromaDB collections"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{chroma_url}/api/v1/collections")
            collections = response.json()
            return {"collections": [c["name"] for c in collections]}
    except Exception as e:
        logger.error(f"Failed to fetch ChromaDB collections: {e}")
        raise HTTPException(status_code=500, detail=f"ChromaDB connection error: {str(e)}")

async def process_multiagent_request(data: dict, websocket: WebSocket = None, 
                                    ws_active_ref: list = None, 
                                    last_activity_ref: list = None):
    """Process multiagent request with metrics tracking"""
    from typing import Set
    
    # Initialize metrics
    _run_t0 = _now_ms()
    _agents_seen: Set[str] = set()

    async def safe_send(message_dict):
        """Safely send message only if WebSocket is still connected"""
        if websocket and (ws_active_ref is None or ws_active_ref[0]):
            try:
                await websocket.send_json(message_dict)
                # Update last activity time
                if last_activity_ref is not None:
                    last_activity_ref[0] = time.time()
                return True
            except Exception as e:
                logger.warning(f"Failed to send WebSocket message: {e}")
                if ws_active_ref:
                    ws_active_ref[0] = False
                return False
        return False

    try:
        query = data.get("query", "")
        context = data.get("context", "")
        model_key = data.get("model", "codegen-2b")
        strategy = data.get("strategy", "collaborative")
        num_agents = data.get("num_agents", 3)
        context_size = data.get("context_size", None)
        
        use_rag = data.get("use_rag", False)
        chroma_url = data.get("chroma_url", "http://127.0.0.1:8000")
        collection_name = data.get("collection_name", None)
        embedding_model = data.get("embedding_model", "mistral")
        embedding_source = data.get("embedding_source", "ollama")
        ollama_url = data.get("ollama_url", "http://127.0.0.1:11434")
        top_k_docs = data.get("top_k_docs", 7)
        
        # Initialize metrics
        _metrics = _agent_metrics_init(
            model_key=model_key,
            strategy=strategy,
            num_agents=num_agents,
            context_mode="rag" if use_rag else "direct",
            similarity_threshold=0.0,
            k=top_k_docs,
            context_window=context_size or 4096
        )

        if not query:
            await safe_send({"type": "error", "message": "Query is required"})
            return None
        
        logger.info(f"🎯 MultiAgent request: {num_agents} agents, "
                f"{strategy} strategy, model: {model_key}, RAG: {use_rag}"
                + (f", context_size: {context_size}" if context_size else ""))
        
        start_time = time.time()
        
        # RAG initialization
        local_retriever = None
        if use_rag and collection_name:
            _t_retr = _now_ms()
            await safe_send({
                "type": "progress",
                "progress": 10,
                "message": f"Initializing RAG with {embedding_model} ({embedding_source})..."
            })
            
            local_retriever = RAGRetriever(
                chroma_url, collection_name, embedding_model,
                embedding_source, ollama_url
            )
            success = await local_retriever.initialize()
            _metrics["retrieval_ms"] += _now_ms() - _t_retr
            
            if not success:
                await safe_send({
                    "type": "error",
                    "message": f"Failed to initialize RAG with collection '{collection_name}'"
                })
                return local_retriever
        
        await safe_send({
            "type": "progress",
            "progress": 15,
            "message": f"Loading {model_key} model..."
        })
        
        model_name = get_model_name_from_key(model_key)
        model, tokenizer = load_llm_model(model_name)
        
        # Track prompt tokens
        _metrics["prompt_tokens_total"] += _count_tokens_approx(query, tokenizer)
        _metrics["prompt_tokens_total"] += _count_tokens_approx(context, tokenizer)
        
        await safe_send({
            "type": "progress",
            "progress": 25,
            "message": f"Creating {num_agents} agents" + 
                    (" with RAG support..." if use_rag else "...")
        })
        
        agents = create_agents(num_agents, strategy, model, tokenizer, llm_device, local_retriever)
        
        await safe_send({
            "type": "progress",
            "progress": 40,
            "message": f"Executing {strategy} strategy..."
        })
        
        # Execute strategy
        _t_gen = _now_ms()
        
        if strategy == "collaborative":
            results = await collaborative_strategy(
                agents, query, context, 
                websocket, 
                None, context_size, _metrics, _agents_seen, tokenizer
            )
        elif strategy == "sequential":
            results = await sequential_strategy(
                agents, query, context, 
                websocket, 
                context_size, _metrics, _agents_seen, tokenizer
            )
        elif strategy == "competitive":
            results = await competitive_strategy(
                agents, query, context, 
                websocket, 
                context_size, _metrics, _agents_seen, tokenizer
            )
        elif strategy == "hierarchical":
            results = await hierarchical_strategy(
                agents, query, context, 
                websocket, 
                context_size, _metrics, _agents_seen, tokenizer
            )
        else:
            results = await collaborative_strategy(
                agents, query, context, 
                websocket, 
                None, context_size, _metrics, _agents_seen, tokenizer
            )
        
        _metrics["gen_ms"] += _now_ms() - _t_gen
        
        # Track completion tokens
        if "final_answer" in results:
            _metrics["completion_tokens_total"] += _count_tokens_approx(results["final_answer"], tokenizer)
        
        _metrics["tokens_total"] = _metrics["prompt_tokens_total"] + _metrics["completion_tokens_total"]
        _metrics["agents_participated"] = len(_agents_seen)
        _metrics["messages_total"] = len(results.get("agent_responses", []))
        
        if use_rag:
            total_docs = sum(r.get("rag_docs_used", 0) for r in results["agent_responses"])
            results["rag_info"] = {
                "enabled": True,
                "collection": collection_name,
                "embedding_model": embedding_model,
                "embedding_source": embedding_source,
                "total_documents_retrieved": total_docs,
                "avg_docs_per_agent": round(total_docs / num_agents, 1) if num_agents > 0 else 0
            }
        
        await safe_send({
            "type": "progress",
            "progress": 80,
            "message": "Calculating metrics..."
        })
        
        metrics = calculate_metrics(results["agent_responses"], results["final_answer"])
        strategy_analysis = analyze_strategy(results["strategy"], metrics)
        
        processing_time = time.time() - start_time
        _metrics["end_to_end_ms"] = _now_ms() - _run_t0
        
        results["performance_metrics"] = _metrics
        
        await safe_send({"type": "progress", "progress": 100, "message": "Complete!"})
        
        sent = await safe_send({
            "type": "complete",
            "results": {
                **results,
                "num_agents": num_agents,
                "processing_time": round(processing_time, 2),
                "metrics": metrics,
                "strategy_analysis": strategy_analysis,
                "performance_metrics": _metrics
            }
        })
        
        if sent:
            logger.info(f"✅ Query completed in {processing_time:.2f}s")
        else:
            logger.warning(f"⚠️ Processing completed but client disconnected")
        
        return local_retriever
        
    except CancelledError:
        logger.warning(f"🛑 Processing cancelled")
        raise
        
    except Exception as e:
        logger.error(f"Processing error: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        
        await safe_send({"type": "error", "message": str(e)})
        raise

@app.websocket("/multiagent-query")
async def multiagent_query_endpoint(websocket: WebSocket):
    """WebSocket endpoint with RAG support and proper cancellation"""
    logger.info(f"📞 WebSocket connection attempt from {websocket.client}")
    
    try:
        await websocket.accept()
        logger.info("✅ WebSocket connection accepted")
    except Exception as e:
        logger.error(f"Failed to accept WebSocket: {e}")
        return
    
    retriever = None
    keepalive_task: Task = None
    processing_task: Task = None
    task_id = f"ws_{id(websocket)}_{int(time.time())}"
    ws_active = [True]  # Use list so it's mutable in nested functions
    last_activity = [time.time()]  # Track last activity
    
    async def send_keepalive():
        """Send periodic keepalive pings"""
        try:
            while ws_active[0]:
                await asyncio.sleep(10)  # Check every 10 seconds
                if not ws_active[0]:
                    break
                
                # Only send keepalive if no recent activity
                time_since_activity = time.time() - last_activity[0]
                if time_since_activity > 8:  # 8 seconds of inactivity
                    try:
                        await websocket.send_json({
                            "type": "keepalive",
                            "timestamp": int(time.time()),
                            "message": "Processing..."
                        })
                        last_activity[0] = time.time()
                        logger.debug("💓 Sent keepalive ping")
                    except Exception as e:
                        logger.warning(f"Keepalive ping failed: {e}")
                        ws_active[0] = False
                        break
        except CancelledError:
            logger.debug("Keepalive task cancelled")
        except Exception as e:
            logger.error(f"Keepalive error: {e}")
            ws_active[0] = False
    
    try:
        # Start keepalive task
        keepalive_task = asyncio.create_task(send_keepalive())
        logger.info("✅ Keepalive task started")
        
        logger.info("⏳ Waiting for client to send request data...")
        
        try:
            data = await asyncio.wait_for(websocket.receive_json(), timeout=30.0)
            last_activity[0] = time.time()  # Update activity time
            logger.info(f"📨 Received request data:")
            logger.info(f"   - Query: {data.get('query', '')[:100]}...")
            logger.info(f"   - Model: {data.get('model', 'unknown')}")
            logger.info(f"   - Strategy: {data.get('strategy', 'unknown')}")
            logger.info(f"   - Agents: {data.get('num_agents', 0)}")
            logger.info(f"   - RAG: {data.get('use_rag', False)}")
        except asyncio.TimeoutError:
            logger.error("❌ Timeout waiting for initial message from client (30s)")
            ws_active[0] = False
            await websocket.close(code=1000, reason="Timeout")
            return
        except Exception as recv_error:
            logger.error(f"❌ Error receiving initial message: {recv_error}")
            ws_active[0] = False
            await websocket.close(code=1011, reason=f"Receive error: {recv_error}")
            return
        
        if not data or not isinstance(data, dict):
            logger.error(f"❌ Invalid data received: {type(data)}")
            ws_active[0] = False
            await websocket.close(code=1003, reason="Invalid data format")
            return
        
        if not data.get("query"):
            logger.error("❌ No query in request")
            ws_active[0] = False
            await websocket.close(code=1003, reason="Missing query")
            return
        
        logger.info(f"✅ Request validated, starting processing...")
        
        # Process request with ws_active flag and last_activity tracker
        processing_task = asyncio.create_task(
            process_multiagent_request(data, websocket, ws_active, last_activity)
        )
        
        try:
            retriever = await processing_task
        except CancelledError:
            logger.warning("Processing task cancelled")
            cancellation_tracker.mark_cancelled(task_id)
        except Exception as e:
            logger.error(f"Processing task error: {e}")
        
    except WebSocketDisconnect:
        logger.warning("WebSocket disconnected by client")
        ws_active[0] = False
        cancellation_tracker.mark_cancelled(task_id)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        ws_active[0] = False
        cancellation_tracker.mark_cancelled(task_id)
    finally:
        ws_active[0] = False
        logger.info(f"🧹 Cleaning up task {task_id}")
        
        if keepalive_task and not keepalive_task.done():
            keepalive_task.cancel()
            try:
                await keepalive_task
            except CancelledError:
                pass
        
        if processing_task and not processing_task.done():
            processing_task.cancel()
            try:
                await processing_task
            except CancelledError:
                pass
        
        if retriever:
            try:
                await retriever.close()
            except Exception as e:
                logger.warning(f"Error closing retriever: {e}")
        
        cancellation_tracker.remove(task_id)
        logger.info(f"✅ Cleanup complete for task {task_id}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "MultiAgent System API with RAG",
        "device": llm_device if llm_device else "not initialized",
        "model": current_model_name if current_model_name else "not loaded",
        "rag_support": True,
        "embedding_sources": ["ollama", "huggingface"],
        "cancellation_support": True
    }

@app.get("/models")
async def get_available_models():
    """Get list of available models"""
    return [{"key": key, "name": name} for key, name in MODEL_MAPPING.items()]

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "MultiAgent System API with RAG",
        "version": "2.0.0",
        "endpoints": {
            "health": "/health",
            "models": "/models",
            "websocket": "/multiagent-query"
        },
        "features": ["RAG", "Multiple LLMs", "Cancellation Support"]
    }

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("🛑 Server shutting down...")
    logger.info("✅ Cleanup complete")

# Update the if __name__ == "__main__" block (at the end)

if __name__ == "__main__":
    import uvicorn
    args = parse_arguments()

    # Apply cache directory settings if provided
    if args.cache_dir:
        cache_dir = os.path.expanduser(args.cache_dir)
        os.makedirs(cache_dir, exist_ok=True)
        os.environ["HF_HOME"] = cache_dir
        os.environ["TRANSFORMERS_CACHE"] = f"{cache_dir}/transformers"
        os.environ["TORCH_HOME"] = f"{cache_dir}/torch"
        logger.info(f"📁 Using custom cache directory: {cache_dir}")
    
    if args.temp_dir:
        temp_dir = os.path.expanduser(args.temp_dir)
        os.makedirs(temp_dir, exist_ok=True)
        os.environ["TMPDIR"] = temp_dir
        tempfile.tempdir = temp_dir
        logger.info(f"📁 Using custom temp directory: {temp_dir}")

    logger.info(f"🚀 Starting MultiAgent System API with RAG on {args.host}:{args.port}")
    logger.info(f"📋 Supported models: {list(MODEL_MAPPING.keys())}")
    logger.info(f"🔧 Strategies: collaborative, sequential, competitive, hierarchical")
    logger.info(f"🗄️  RAG: Enabled (Ollama + HuggingFace)")
    logger.info(f"🛑 Cancellation: ENABLED - Cancel button and Ctrl+C now work!")
    logger.info(f"💡 Press Ctrl+C to stop the server gracefully")
    
    try:
        # Run with lifespan to handle shutdown
        config = uvicorn.Config(
            app, 
            host=args.host, 
            port=args.port, 
            log_level="info",
            loop="asyncio"
        )
        server = uvicorn.Server(config)
        
        # Run server
        server.run()
        
    except KeyboardInterrupt:
        logger.info("🛑 Server stopped by user (Ctrl+C)")
        sys.exit(0)
    except Exception as e:
        logger.error(f"❌ Server error: {e}")
        sys.exit(1)