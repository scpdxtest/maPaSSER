from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import asyncio
import logging
import os
import time
import base64
from typing import Optional
import subprocess
import tempfile
import json
from datetime import datetime
import platform
import re
import argparse

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set environment variable to avoid tokenizer warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

app = FastAPI()

# Add CORS middleware
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Port configuration constants
DEFAULT_API_PORT = 8003
DEFAULT_MERMAID_PORT = 9000

# Global variables
llm_model = None
llm_tokenizer = None
llm_device = None
current_model_name = None  # Track currently loaded model
MERMAID_PORT = DEFAULT_MERMAID_PORT

# Global storage for diagrams
diagram_storage = {
    "original": {"code": "", "svg": "", "type": ""},
    "edited": {"code": "", "svg": "", "type": ""}
}

class DiagramGenerationRequest(BaseModel):
    scientific_text: str
    diagram_type: str = "mermaid"
    diagram_style: str = "flowchart"
    llm_model: str = "codegen-2b"  # Changed default to match frontend
    style_preferences: Optional[dict] = None

class DiagramEditRequest(BaseModel):
    current_diagram_code: str
    edit_instructions: str
    diagram_type: str = "mermaid"
    llm_model: str = "codegen-2b"  # Changed default to match frontend
    style_preferences: Optional[dict] = None

def parse_arguments():
    """Parse command line arguments for port configuration"""
    parser = argparse.ArgumentParser(description='Scientific Diagram Generation Server')
    parser.add_argument('--port', type=int, default=DEFAULT_API_PORT, 
                       help=f'API server port (default: {DEFAULT_API_PORT})')
    parser.add_argument('--mermaid-port', type=int, default=DEFAULT_MERMAID_PORT,
                       help=f'Mermaid CLI port (default: {DEFAULT_MERMAID_PORT})')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                       help='API server host (default: 0.0.0.0)')
    return parser.parse_args()

def get_optimal_device():
    """Get the best available device with MPS support for Apple Silicon"""
    # Check for Apple Silicon MPS support
    if torch.backends.mps.is_available():
        logger.info("MPS (Metal Performance Shaders) is available - using Apple Silicon GPU acceleration")
        return "mps"
    # Check for CUDA support
    elif torch.cuda.is_available():
        logger.info(f"CUDA is available - using GPU: {torch.cuda.get_device_name()}")
        return "cuda"
    # Fallback to CPU
    else:
        logger.info("Using CPU (no GPU acceleration available)")
        return "cpu"

def log_device_info():
    """Log detailed device information"""
    logger.info(f"System: {platform.system()} {platform.release()}")
    logger.info(f"Architecture: {platform.machine()}")
    logger.info(f"PyTorch version: {torch.__version__}")
    
    if torch.backends.mps.is_available():
        logger.info("✅ MPS (Apple Silicon GPU) is available and supported")
    elif torch.cuda.is_available():
        logger.info(f"✅ CUDA is available: {torch.cuda.get_device_name()}")
        logger.info(f"CUDA version: {torch.version.cuda}")
    else:
        logger.info("ℹ️ No GPU acceleration available, using CPU")

def clear_mps_cache():
    """Clear MPS cache to prevent memory issues"""
    if llm_device == "mps":
        try:
            if hasattr(torch.mps, 'empty_cache'):
                torch.mps.empty_cache()
                logger.debug("Cleared MPS cache")
        except Exception as e:
            logger.warning(f"Failed to clear MPS cache: {e}")

def get_model_name_from_key(model_key):
    """Map UI model keys to actual model names"""
    model_mapping = {
        "codegen-2b": "Salesforce/codegen-2B-nl",
        "codegen-350m": "Salesforce/codegen-350M-nl", 
        "deepseek-coder": "deepseek-ai/deepseek-coder-6.7b-instruct",
        "diaglo-gpt": "microsoft/DialoGPT-medium",
        "distilgpt2": "distilgpt2",
        "wizardcoder": "WizardLM/WizardCoder-Python-7B-V1.0",
        "sci-coder": "Salesforce/codegen-2B-nl"
    }
    # FIXED: Default to 2B model to match frontend expectation
    return model_mapping.get(model_key, "Salesforce/codegen-2B-nl")

def reload_llm_model(model_name):
    """Force reload a specific model"""
    global llm_model, llm_tokenizer, llm_device, current_model_name
    
    logger.info(f"🔄 FORCE RELOADING model: {model_name}")
    
    # Clear existing model
    llm_model = None
    llm_tokenizer = None
    llm_device = None
    current_model_name = None
    
    # Clear cache
    clear_mps_cache()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Load new model
    return load_llm_model(model_name)

def load_llm_model(model_name="Salesforce/codegen-2B-nl"):  # Changed default to 2B
    """Load CodeGen or other LLM for diagram generation with MPS support"""
    global llm_model, llm_tokenizer, llm_device, current_model_name
    
    # FIXED: Check if we need to reload a different model
    if llm_model is not None and current_model_name == model_name:
        logger.info(f"Model {model_name} already loaded, reusing...")
        return llm_model, llm_tokenizer
    elif llm_model is not None and current_model_name != model_name:
        logger.info(f"Different model requested ({model_name} vs {current_model_name}), reloading...")
        llm_model = None
        llm_tokenizer = None
        llm_device = None
        current_model_name = None
        clear_mps_cache()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    logger.info(f"🔄 Loading LLM model: {model_name}")
    log_device_info()
    
    # Get optimal device with MPS support
    device = get_optimal_device()
    llm_device = device
    
    # For debugging - log exactly what's being requested
    logger.info(f"📋 Model request details:")
    logger.info(f"   - Requested model: {model_name}")
    logger.info(f"   - Target device: {device}")

    # FIXED: Try the requested model first, only fallback if it fails
    # Don't automatically downgrade based on device
    models_to_try = [model_name]
    
    # Add fallbacks only if the primary model fails
    fallback_models = [
        "Salesforce/codegen-350M-nl",       # 350MB - Fallback
        "microsoft/DialoGPT-medium",        # 345MB - Conversational
        "distilgpt2",                       # 353MB - General purpose
        "gpt2",                             # 548MB - Final fallback
    ]
    
    # Only add fallbacks if they're different from requested model
    for fallback in fallback_models:
        if fallback != model_name:
            models_to_try.append(fallback)
    
    logger.info(f"Model loading order: {models_to_try}")
    
    for model_to_try in models_to_try:
        try:
            logger.info(f"🔄 Attempting to load: {model_to_try}")
            
            # Special handling for CodeGen models
            if "codegen" in model_to_try.lower():
                llm_tokenizer = AutoTokenizer.from_pretrained(
                    model_to_try, 
                    trust_remote_code=True,
                    padding_side='left'  # Important for CodeGen
                )
                # Add pad token if missing
                if llm_tokenizer.pad_token is None:
                    llm_tokenizer.pad_token = llm_tokenizer.eos_token
                
                # Determine optimal data type based on device
                if device == "mps":
                    # MPS works best with float32 for now (Apple Silicon optimization)
                    torch_dtype = torch.float32
                    logger.info("Using float32 for MPS compatibility")
                elif device == "cuda":
                    torch_dtype = torch.float16
                    logger.info("Using float16 for CUDA optimization")
                else:
                    torch_dtype = torch.float32
                    logger.info("Using float32 for CPU")
                
                llm_model = AutoModelForCausalLM.from_pretrained(
                    model_to_try,
                    torch_dtype=torch_dtype,
                    device_map=None,  # Don't use auto device mapping for manual control
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                    pad_token_id=llm_tokenizer.eos_token_id
                )
            else:
                # Standard loading for other models
                llm_tokenizer = AutoTokenizer.from_pretrained(model_to_try, trust_remote_code=True)
                if llm_tokenizer.pad_token is None:
                    llm_tokenizer.pad_token = llm_tokenizer.eos_token
                
                # Determine optimal data type based on device
                if device == "mps":
                    torch_dtype = torch.float32
                elif device == "cuda":
                    torch_dtype = torch.float16
                else:
                    torch_dtype = torch.float32
                
                llm_model = AutoModelForCausalLM.from_pretrained(
                    model_to_try,
                    torch_dtype=torch_dtype,
                    device_map=None,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True
                )
            
            # Move model to device with error handling
            try:
                llm_model.to(device)
                current_model_name = model_to_try  # Store the loaded model name
                logger.info(f"✅ Successfully loaded: {model_to_try} on {device}")
                
                # Log model info
                param_count = sum(p.numel() for p in llm_model.parameters()) / 1e6
                logger.info(f"Model parameters: ~{param_count:.1f}M")
                
                if device == "mps":
                    logger.info("🚀 Using Apple Silicon GPU acceleration via MPS")
                elif device == "cuda":
                    logger.info(f"🚀 Using NVIDIA GPU: {torch.cuda.get_device_name()}")
                else:
                    logger.info("💻 Using CPU processing")
                
                return llm_model, llm_tokenizer
                
            except Exception as device_error:
                logger.error(f"Failed to move model to {device}: {device_error}")
                if device == "mps":
                    logger.info("Falling back to CPU due to MPS error")
                    device = "cpu"
                    llm_device = device
                    llm_model.to(device)
                    current_model_name = model_to_try
                    logger.info(f"✅ Successfully loaded: {model_to_try} on {device}")
                    return llm_model, llm_tokenizer
                else:
                    raise device_error
            
        except Exception as e:
            logger.error(f"❌ Failed to load {model_to_try}: {e}")
            # Clear any partial loading
            llm_model = None
            llm_tokenizer = None
            current_model_name = None
            clear_mps_cache()
            continue
    
    raise Exception("Failed to load any compatible model")

def clean_generated_code(generated_text, diagram_type):
    """Clean and validate generated diagram code"""
    if not generated_text or len(generated_text.strip()) < 5:
        logger.warning("Generated text is empty or too short")
        return None
    
    # Remove common artifacts and HTML/CSS/JS content
    cleaned = generated_text
    
    # Remove HTML tags, CSS, and JavaScript
    cleaned = re.sub(r'<[^>]+>', '', cleaned)  # Remove HTML tags
    cleaned = re.sub(r'{\s*[^}]*\s*}', '', cleaned)  # Remove CSS blocks
    cleaned = re.sub(r'/\*.*?\*/', '', cleaned, flags=re.DOTALL)  # Remove CSS comments
    cleaned = re.sub(r'<script.*?</script>', '', cleaned, flags=re.DOTALL)  # Remove scripts
    cleaned = re.sub(r'function\s+\w+.*?{.*?}', '', cleaned, flags=re.DOTALL)  # Remove functions
    cleaned = re.sub(r'\$\([^)]+\).*?;', '', cleaned)  # Remove jQuery
    cleaned = re.sub(r'var\s+\w+.*?;', '', cleaned)  # Remove variable declarations
    cleaned = re.sub(r'console\.log.*?;', '', cleaned)  # Remove console.log
    
    # Remove template syntax
    cleaned = re.sub(r'{%.*?%}', '', cleaned)
    cleaned = re.sub(r'{{.*?}}', '', cleaned)
    
    # Remove markdown code blocks
    cleaned = re.sub(r'```\w*\n?', '', cleaned)
    cleaned = re.sub(r'```', '', cleaned)
    
    # Clean up whitespace and extra characters
    cleaned = re.sub(r'\n\s*\n', '\n', cleaned)  # Multiple newlines to single
    cleaned = re.sub(r'^\s+', '', cleaned, flags=re.MULTILINE)  # Remove leading whitespace
    cleaned = cleaned.strip()
    
    # Enhanced validation for Mermaid
    if diagram_type == "mermaid":
        # Check if it starts with a valid Mermaid directive
        valid_starts = ['flowchart', 'graph', 'sequenceDiagram', 'classDiagram', 'stateDiagram', 'erDiagram', 'journey', 'gantt']
        if not any(cleaned.lower().startswith(start.lower()) for start in valid_starts):
            logger.warning(f"Generated Mermaid code doesn't start with valid directive. Starts with: {cleaned[:50]}")
            return None
        
        # Check for basic Mermaid syntax
        if len(cleaned) < 20 or '-->' not in cleaned and '->' not in cleaned and 'participant' not in cleaned:
            logger.warning(f"Generated Mermaid code lacks basic syntax elements. Code: {cleaned[:100]}")
            return None
    
    # PlantUML validation
    elif diagram_type == "plantuml":
        if not cleaned.lower().startswith('@startuml'):
            logger.warning(f"Generated PlantUML code doesn't start with @startuml. Starts with: {cleaned[:50]}")
            return None
        
        # Check for basic PlantUML syntax
        if len(cleaned) < 30 or ('@enduml' not in cleaned.lower() and len(cleaned) < 100):
            logger.warning(f"Generated PlantUML code lacks proper syntax. Code: {cleaned[:100]}")
            return None
    
    # GraphViz validation
    elif diagram_type == "graphviz":
        if not cleaned.lower().startswith('digraph') and not cleaned.lower().startswith('graph'):
            logger.warning(f"Generated GraphViz code doesn't start with digraph/graph")
            return None
    
    # If the cleaned result is too short, return None
    if len(cleaned) < 20:
        logger.warning(f"Generated code too short after cleaning: {len(cleaned)} chars")
        return None
    
    logger.info(f"Code validation passed. Length: {len(cleaned)}, starts with: {cleaned[:30]}")
    return cleaned

def generate_diagram_code(scientific_text, diagram_type, diagram_style, style_prefs=None, model_key="codegen-2b"):
    """Generate diagram code using LLM with improved prompts for CodeGen and device optimization"""
    
    # Clear MPS cache before generation
    clear_mps_cache()
    
    model_name = get_model_name_from_key(model_key)
    logger.info(f"🎯 Generating with model key: {model_key} -> {model_name}")
    
    # FIXED: Force load the specific requested model
    model, tokenizer = load_llm_model(model_name)
    
    # IMPROVED: Much more specific and constrained prompts
    if diagram_type == "mermaid":
        if diagram_style == "flowchart":
            prompt = f"""Create a Mermaid flowchart from this text.

TEXT: {scientific_text[:800]}

Generate ONLY valid Mermaid flowchart syntax. Start with "flowchart TD" and use --> for connections:

flowchart TD
    A["""
        elif diagram_style == "sequence":
            prompt = f"""Create a Mermaid sequence diagram from this text.

TEXT: {scientific_text[:800]}

Generate ONLY valid Mermaid sequence diagram syntax:

sequenceDiagram
    participant A as """
        elif diagram_style == "class":
            prompt = f"""Create a Mermaid class diagram from this text.

TEXT: {scientific_text[:800]}

Generate ONLY valid Mermaid class diagram syntax:

classDiagram
    class A {{"""
        else:  # default to flowchart
            prompt = f"""Create a Mermaid flowchart from this text.

TEXT: {scientific_text[:800]}

Generate ONLY valid Mermaid flowchart syntax. Start with "flowchart TD" and use --> for connections:

flowchart TD
    A["""
    elif diagram_type == "graphviz":
        prompt = f"""Create a GraphViz diagram from this text.

TEXT: {scientific_text[:800]}

Generate ONLY valid DOT notation:

digraph G {{
    rankdir=TB;
    node [shape=box];
    """
    else:
        prompt = f"""Create a {diagram_type} diagram from this text: {scientific_text[:500]}

Generate only the diagram code:"""
    
    # Shorter tokenization to prevent truncation issues
    inputs = tokenizer(
        prompt, 
        return_tensors="pt", 
        truncation=True, 
        max_length=512,  # Reduced to prevent context overflow
        padding=True
    )
    
    # Move inputs to device
    if llm_device != "cpu":
        try:
            inputs = {k: v.to(llm_device) for k, v in inputs.items()}
        except Exception as e:
            logger.warning(f"Failed to move inputs to {llm_device}: {e}")
    
    # FIXED: More conservative generation parameters
    generation_kwargs = {
        "max_new_tokens": 100,      # Reduced for more focused output
        "temperature": 0.1,         # Very low for more deterministic output
        "do_sample": True,
        "top_p": 0.7,              # More focused
        "top_k": 20,               # More focused
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "repetition_penalty": 1.2,  # Stronger repetition penalty
        "no_repeat_ngram_size": 2,
    }
    
    # Device-specific optimizations
    if llm_device == "mps":
        logger.info("Using MPS-optimized generation parameters")
        generation_kwargs.update({
            "max_new_tokens": 80,
            "temperature": 0.05,
            "top_p": 0.6,
            "top_k": 15,
        })
    elif llm_device == "cuda":
        logger.info("Using CUDA-optimized generation parameters")
        generation_kwargs["max_new_tokens"] = 120
    
    logger.info(f"Generating with model: {current_model_name} on {llm_device}")
    logger.info(f"📊 Request: {diagram_type} - {diagram_style}")
    logger.info(f"📝 Text preview: {scientific_text[:100]}...")
    logger.info(f"🔧 Generation params: max_tokens={generation_kwargs['max_new_tokens']}, temp={generation_kwargs['temperature']}")
        
    with torch.no_grad():
        try:
            logger.info(f"Starting generation on {llm_device}")
            outputs = model.generate(**inputs, **generation_kwargs)
            logger.info("Generation completed successfully")
        except Exception as gen_error:
            logger.error(f"Generation failed on {llm_device}: {gen_error}")
            # Try with even more conservative parameters
            fallback_kwargs = {
                "max_new_tokens": 30,
                "temperature": 0.01,
                "do_sample": False,  # Greedy decoding
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token_id": tokenizer.eos_token_id,
            }
            try:
                logger.info("Retrying with minimal parameters...")
                outputs = model.generate(**inputs, **fallback_kwargs)
                logger.info("Fallback generation successful")
            except Exception as fallback_error:
                logger.error(f"Fallback generation also failed: {fallback_error}")
                return generate_fallback_diagram(scientific_text, diagram_type, diagram_style)
    
    # Clear cache after generation
    clear_mps_cache()
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    logger.info(f"Raw generated response length: {len(response)}")
    logger.info(f"Raw response: {response[:300]}...")
    
    # Extract the generated part (remove the prompt)
    generated_part = response[len(prompt):].strip()
    logger.info(f"Generated part (after prompt removal): {generated_part[:200]}...")
    
    # Clean the generated code
    cleaned_code = clean_generated_code(generated_part, diagram_type)
    
    if cleaned_code is None:
        logger.warning("Generated code failed validation, using fallback")
        return generate_fallback_diagram(scientific_text, diagram_type, diagram_style)
    
    logger.info(f"Final cleaned code length: {len(cleaned_code)}")
    logger.info(f"Final code: {cleaned_code[:150]}...")
    return cleaned_code

def generate_fallback_diagram(scientific_text, diagram_type, diagram_style):
    """Generate a simple fallback diagram if LLM fails"""
    logger.info(f"Generating fallback diagram for {diagram_type} - {diagram_style}")
    
    # Create a more relevant fallback based on the text content
    text_lower = scientific_text.lower()
    
    if diagram_type == "mermaid" and diagram_style == "flowchart":
        # Check if text is about blockchain/evaluation framework
        if any(word in text_lower for word in ['blockchain', 'evaluation', 'framework', 'hste', 'typology']):
            return """flowchart TD
    A[Design Axes] --> B[Authority Definition]
    A --> C[Accountability Measures] 
    A --> D[Legitimacy Framework]
    
    B --> E[HSTE Model Integration]
    C --> E
    D --> E
    
    E --> F[Pillar A: Technical Performance]
    E --> G[Pillar B: Governance Quality]
    E --> H[Pillar C: Institutional Fit]
    E --> I[Pillar D: Sustainability & Risk]
    
    F --> J[Decision Models]
    G --> J
    H --> J
    I --> J
    
    J --> K[Scenario Planning]
    J --> L[Delphi Analysis]
    J --> M[Cost-Benefit Analysis] 
    J --> N[Risk-Benefit Analysis]
    
    K --> O[Reproducible Evidence Protocol]
    L --> O
    M --> O
    N --> O"""
        else:
            # Generic fallback
            return """flowchart TD
    A[Start: Input Data] --> B[Data Processing]
    B --> C[Analysis Phase]
    C --> D[Model Training]
    D --> E[Evaluation]
    E --> F[Results Output]"""
    
    elif diagram_type == "mermaid" and diagram_style == "class":
        if any(word in text_lower for word in ['blockchain', 'evaluation', 'framework']):
            return """classDiagram
    class DesignAxes {
        +authority
        +accountability
        +legitimacy
        +defineAxes()
    }
    
    class HSTEModel {
        +pillarA_technical
        +pillarB_governance
        +pillarC_institutional
        +pillarD_sustainability
        +integrate()
    }
    
    class DecisionModels {
        +scenarioPlanning
        +delphiAnalysis
        +costBenefitAnalysis
        +riskBenefitAnalysis
        +applyModels()
    }
    
    class EvidenceProtocol {
        +reproducibleResults
        +auditTrail
        +generateEvidence()
    }
    
    DesignAxes --> HSTEModel
    HSTEModel --> DecisionModels
    DecisionModels --> EvidenceProtocol"""
        else:
            return """classDiagram
    class DataProcessor {
        +inputData
        +processedData
        +process()
    }
    
    class Analyzer {
        +analysisResult
        +analyze()
    }
    
    class ResultGenerator {
        +finalResult
        +generate()
    }
    
    DataProcessor --> Analyzer
    Analyzer --> ResultGenerator"""
    
    elif diagram_type == "mermaid" and diagram_style == "sequence":
        if any(word in text_lower for word in ['blockchain', 'evaluation', 'framework']):
            return """sequenceDiagram
    participant Evaluator as Framework Evaluator
    participant Typology as Design Typology
    participant HSTE as HSTE Model
    participant Decision as Decision Models
    participant Evidence as Evidence Protocol
    
    Evaluator->>Typology: Define design axes
    Typology->>HSTE: Map to four pillars
    HSTE->>Decision: Apply rule-gates
    Decision->>Decision: Weight trade-offs
    Decision->>Evidence: Generate artifacts
    Evidence-->>Evaluator: Reproducible results"""
        else:
            return """sequenceDiagram
    participant User as Researcher
    participant System as Analysis System
    participant Data as Data Store
    participant Model as ML Model
    
    User->>System: Submit input data
    System->>Data: Store raw data
    System->>System: Preprocess data
    System->>Model: Train model
    Model-->>System: Return metrics
    System-->>User: Analysis results"""
    
    elif diagram_type == "graphviz":
        if any(word in text_lower for word in ['blockchain', 'evaluation', 'framework']):
            return """digraph G {
    rankdir=TB;
    node [shape=box, style=filled, fillcolor="#e3f2fd", fontcolor="#0d47a1"];
    edge [color="#1976d2", penwidth=2];
    
    "Design Axes" -> "Authority";
    "Design Axes" -> "Accountability";  
    "Design Axes" -> "Legitimacy";
    
    "Authority" -> "HSTE Model";
    "Accountability" -> "HSTE Model";
    "Legitimacy" -> "HSTE Model";
    
    "HSTE Model" -> "Technical Performance";
    "HSTE Model" -> "Governance Quality";
    "HSTE Model" -> "Institutional Fit";
    "HSTE Model" -> "Sustainability Risk";
    
    "Technical Performance" -> "Decision Models";
    "Governance Quality" -> "Decision Models";
    "Institutional Fit" -> "Decision Models";
    "Sustainability Risk" -> "Decision Models";
    
    "Decision Models" -> "Scenario Planning";
    "Decision Models" -> "Delphi Analysis";
    "Decision Models" -> "Cost-Benefit Analysis";
    "Decision Models" -> "Risk-Benefit Analysis";
    
    "Scenario Planning" -> "Evidence Protocol";
    "Delphi Analysis" -> "Evidence Protocol";
    "Cost-Benefit Analysis" -> "Evidence Protocol";
    "Risk-Benefit Analysis" -> "Evidence Protocol";
    
    "Design Axes" [fillcolor="#c8e6c9"];
    "Evidence Protocol" [fillcolor="#ffcdd2"];
}"""
        else:
            return """digraph G {
    rankdir=TB;
    node [shape=box, style=filled, fillcolor="#e3f2fd", fontcolor="#0d47a1"];
    edge [color="#1976d2", penwidth=2];
    
    "Input Data" -> "Data Processing";
    "Data Processing" -> "Feature Extraction";
    "Feature Extraction" -> "Model Training";
    "Model Training" -> "Evaluation";
    "Evaluation" -> "Results";
    
    "Input Data" [fillcolor="#c8e6c9"];
    "Results" [fillcolor="#ffcdd2"];
}"""
    
    elif diagram_type == "plantuml":
        return """@startuml
!theme plain
skinparam backgroundColor white
skinparam defaultFontSize 12

actor Evaluator as evaluator
participant "Design Typology" as typology
participant "HSTE Model" as hste
participant "Decision Models" as decision
database "Evidence Protocol" as evidence

evaluator -> typology: Apply design axes
typology -> hste: Map to pillars
hste -> decision: Apply rule-gates
decision -> decision: Weight trade-offs
decision -> evidence: Generate artifacts
evidence --> evaluator: Reproducible results
@enduml"""
    
    # Default fallback for any unhandled cases
    return """flowchart TD
    A[Input] --> B[Process]
    B --> C[Output]"""

def render_diagram(code, diagram_type, mermaid_port=None):
    """Render diagram code to SVG"""
    if mermaid_port is None:
        mermaid_port = MERMAID_PORT
        
    try:
        if diagram_type == "mermaid":
            return render_mermaid(code, mermaid_port)
        elif diagram_type == "graphviz":
            return render_graphviz(code)
        elif diagram_type == "plantuml":
            return render_plantuml(code)
        else:
            return render_mermaid(code, mermaid_port)  # Default to mermaid
    except Exception as e:
        logger.error(f"Failed to render {diagram_type} diagram: {e}")
        return create_fallback_svg(code, diagram_type)

def create_fallback_svg(code, diagram_type):
    """Create a simple SVG when rendering fails"""
    device_info = f"Device: {llm_device.upper()}" if llm_device else "Device: Unknown"
    model_info = f"Model: {current_model_name}" if current_model_name else "Model: Unknown"
    return f"""<svg width="500" height="400" xmlns="http://www.w3.org/2000/svg">
    <rect width="100%" height="100%" fill="#f8f9fa" stroke="#dee2e6"/>
    <text x="250" y="80" text-anchor="middle" font-family="Arial" font-size="18" fill="#495057">
        {diagram_type.upper()} Diagram Generated
    </text>
    <text x="250" y="120" text-anchor="middle" font-family="Arial" font-size="14" fill="#6c757d">
        Rendering temporarily unavailable
    </text>
    <text x="250" y="160" text-anchor="middle" font-family="monospace" font-size="12" fill="#6c757d">
        Code Length: {len(code)} characters
    </text>
    <text x="250" y="180" text-anchor="middle" font-family="Arial" font-size="12" fill="#6c757d">
        {device_info}
    </text>
    <text x="250" y="200" text-anchor="middle" font-family="Arial" font-size="10" fill="#6c757d">
        {model_info}
    </text>
    <text x="250" y="240" text-anchor="middle" font-family="Arial" font-size="10" fill="#6c757d">
        Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    </text>
    <rect x="50" y="280" width="400" height="80" fill="#fff" stroke="#dee2e6"/>
    <text x="70" y="300" font-family="monospace" font-size="10" fill="#495057">Code Preview:</text>
    <text x="70" y="320" font-family="monospace" font-size="9" fill="#6c757d">
        {code[:60]}{'...' if len(code) > 60 else ''}
    </text>
    <text x="70" y="340" font-family="monospace" font-size="9" fill="#6c757d">
        {code[60:120] if len(code) > 60 else ''}{'...' if len(code) > 120 else ''}
    </text>
</svg>"""

def render_mermaid(code, mermaid_port=DEFAULT_MERMAID_PORT):
    """Simple online mermaid rendering for remote servers"""
    try:
        import requests
        
        # Clean and encode
        clean_code = code.strip()
        if clean_code.startswith('```'):
            clean_code = clean_code[3:].strip()
        if clean_code.startswith('mermaid'):
            clean_code = clean_code[7:].strip()
        if clean_code.endswith('```'):
            clean_code = clean_code[:-3].strip()
        
        encoded_code = base64.b64encode(clean_code.encode('utf-8')).decode('ascii')
        url = f"https://mermaid.ink/svg/{encoded_code}"
        
        logger.info("🌐 Using mermaid.ink online service for remote server...")
        response = requests.get(url, timeout=30)
        
        if response.status_code == 200 and response.content:
            logger.info("✅ Online rendering successful")
            return response.text
        else:
            raise Exception(f"Service returned status {response.status_code}")
            
    except Exception as e:
        logger.warning(f"Online rendering failed: {e}")
        return create_enhanced_mermaid_svg(code)

def render_graphviz(code):
    """Render GraphViz DOT to SVG"""
    try:
        import graphviz
        source = graphviz.Source(code)
        return source.pipe(format='svg', encoding='utf-8')
    except ImportError:
        logger.warning("GraphViz not installed, using fallback")
        return create_fallback_svg(code, "graphviz")
    except Exception as e:
        logger.error(f"GraphViz rendering error: {e}")
        return create_fallback_svg(code, "graphviz")

def render_plantuml(code):
    """Render PlantUML to SVG using online service"""
    try:
        import requests
        import zlib
        
        # Clean the code
        clean_code = code.strip()
        if not clean_code.startswith('@startuml'):
            clean_code = f"@startuml\n{clean_code}"
        if not clean_code.endswith('@enduml'):
            clean_code = f"{clean_code}\n@enduml"
        
        # Encode for PlantUML server
        compressed = zlib.compress(clean_code.encode('utf-8'))
        encoded = base64.b64encode(compressed).decode('ascii')
        
        # Use PlantUML server
        url = f"http://www.plantuml.com/plantuml/svg/{encoded}"
        
        logger.info("🌐 Using plantuml.com online service...")
        response = requests.get(url, timeout=30)
        
        if response.status_code == 200 and response.content:
            logger.info("✅ PlantUML online rendering successful")
            return response.text
        else:
            logger.warning(f"PlantUML service returned status {response.status_code}")
            raise Exception(f"PlantUML service failed with status {response.status_code}")
            
    except Exception as e:
        logger.error(f"PlantUML online rendering failed: {e}")
        # Try alternative Kroki service
        try:
            return render_plantuml_kroki(code)
        except Exception as e2:
            logger.error(f"Kroki PlantUML rendering also failed: {e2}")
            return create_fallback_svg(code, "plantuml")

def render_plantuml_kroki(code):
    """Alternative PlantUML renderer using Kroki service"""
    try:
        import requests
        import zlib
        
        # Clean the code
        clean_code = code.strip()
        if not clean_code.startswith('@startuml'):
            clean_code = f"@startuml\n{clean_code}"
        if not clean_code.endswith('@enduml'):
            clean_code = f"{clean_code}\n@enduml"
        
        # Compress and encode for Kroki
        compressed = zlib.compress(clean_code.encode('utf-8'))
        encoded = base64.urlsafe_b64encode(compressed).decode('ascii')
        
        url = f"https://kroki.io/plantuml/svg/{encoded}"
        
        logger.info("🌐 Using Kroki PlantUML service...")
        response = requests.get(url, timeout=30)
        
        if response.status_code == 200:
            logger.info("✅ Kroki PlantUML rendering successful")
            return response.text
        else:
            raise Exception(f"Kroki service returned status {response.status_code}")
            
    except Exception as e:
        logger.error(f"Kroki PlantUML rendering failed: {e}")
        raise

# WebSocket endpoints for diagram generation
@app.websocket("/generate-diagram")
async def generate_diagram_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    try:
        data = await websocket.receive_json()
        
        # FIXED: Get actual model name from UI selection with proper fallback
        model_key = data.get("llm_model", "codegen-2b")  # Changed default to match frontend
        model_name = get_model_name_from_key(model_key)
        device_name = f"{llm_device.upper()}" if llm_device else "Unknown"
        
        # Log the model selection for debugging
        logger.info(f"🎯 Frontend requested model key: {model_key}")
        logger.info(f"🎯 Mapped to model name: {model_name}")
        logger.info(f"🎯 Currently loaded model: {current_model_name}")
        
        await websocket.send_json({"type": "progress", "progress": 10, "message": f"Loading {model_name} on {device_name}..."})
        
        # FIXED: Force load the specific requested model if different from current
        try:
            if model_name != current_model_name:
                logger.info(f"🔄 Force loading requested model: {model_name}")
                model, tokenizer = reload_llm_model(model_name)
                logger.info(f"✅ Model reloaded successfully: {current_model_name}")
            else:
                logger.info(f"✅ Requested model already loaded: {current_model_name}")
                model, tokenizer = load_llm_model(model_name)
        except Exception as model_error:
            logger.error(f"❌ Failed to load requested model {model_name}: {model_error}")
            await websocket.send_json({"type": "error", "message": f"Failed to load model {model_name}: {str(model_error)}"})
            return
        
        # Generate diagram code with timeout
        await websocket.send_json({"type": "progress", "progress": 30, "message": "Generating diagram code..."})
        
        try:
            # Add timeout for diagram generation
            diagram_code = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(
                    None, 
                    generate_diagram_code,
                    data["scientific_text"],
                    data["diagram_type"],
                    data["diagram_style"],
                    data.get("style_preferences", {}),
                    model_key  # Pass the model_key here
                ),
                timeout=120.0  # 2 minute timeout
            )

            await websocket.send_json({"type": "progress", "progress": 70, "message": "Code generated successfully! Rendering diagram..."})
            
        except asyncio.TimeoutError:
            logger.error("Diagram generation timed out")
            await websocket.send_json({"type": "error", "message": "Diagram generation timed out. Try with simpler text or a smaller model."})
            return
        except Exception as gen_error:
            logger.error(f"Diagram generation failed: {gen_error}")
            await websocket.send_json({"type": "progress", "progress": 50, "message": "Generation failed, using fallback..."})
            # Use fallback generation
            diagram_code = generate_fallback_diagram(
                data["scientific_text"], 
                data["diagram_type"], 
                data["diagram_style"]
            )
        
        # Render diagram with configured mermaid port
        svg_content = await asyncio.get_event_loop().run_in_executor(
            None, 
            render_diagram,
            diagram_code,
            data["diagram_type"],
            MERMAID_PORT  # Pass the configured port
        )
        
        # Store results
        diagram_storage["original"]["code"] = diagram_code
        diagram_storage["original"]["svg"] = base64.b64encode(svg_content.encode()).decode()
        diagram_storage["original"]["type"] = data["diagram_type"]
        
        await websocket.send_json({"type": "progress", "progress": 100, "message": "Complete"})
        await websocket.send_json({"type": "complete"})
        
    except Exception as e:
        logger.error(f"Generation error: {e}")
        await websocket.send_json({"type": "error", "message": str(e)})
    finally:
        await websocket.close()

@app.websocket("/edit-diagram")
async def edit_diagram_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    try:
        data = await websocket.receive_json()
        
        await websocket.send_json({"type": "progress", "progress": 20, "message": "Processing edit instructions..."})
        
        # Get model info - FIXED: Default to match frontend
        model_key = data.get("llm_model", "codegen-2b")  # Changed default
        model_name = get_model_name_from_key(model_key)
        
        # Create edit prompt
        edit_prompt = f"""Modify this {data['diagram_type']} diagram code:

Current code:
{data['current_diagram_code']}

Instructions: {data['edit_instructions']}

Generate the modified {data['diagram_type']} code:

"""
        
        # FIXED: Force load the requested model for editing too
        try:
            if model_name != current_model_name:
                logger.info(f"🔄 Loading model for editing: {model_name}")
                model, tokenizer = reload_llm_model(model_name)
            else:
                model, tokenizer = load_llm_model(model_name)
        except Exception as model_error:
            logger.error(f"❌ Failed to load model for editing: {model_error}")
            # Use fallback editing
            await websocket.send_json({"type": "progress", "progress": 80, "message": "Using fallback editing..."})
            edited_code = apply_simple_edits(data['current_diagram_code'], data['edit_instructions'])
            svg_content = render_diagram(edited_code, data["diagram_type"], MERMAID_PORT)
            diagram_storage["edited"]["code"] = edited_code
            diagram_storage["edited"]["svg"] = base64.b64encode(svg_content.encode()).decode()
            diagram_storage["edited"]["type"] = data["diagram_type"]
            await websocket.send_json({"type": "complete"})
            return
        
        inputs = tokenizer(edit_prompt, return_tensors="pt", truncation=True, max_length=1024, padding=True)
        if llm_device != "cpu":
            try:
                inputs = {k: v.to(llm_device) for k, v in inputs.items()}
            except Exception as e:
                logger.warning(f"Failed to move inputs to {llm_device}: {e}")
        
        await websocket.send_json({"type": "progress", "progress": 60, "message": "Generating edited code..."})
        
        # Device-specific generation parameters
        generation_kwargs = {
            "max_new_tokens": 150,
            "temperature": 0.2,
            "do_sample": True,
            "top_p": 0.8,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "repetition_penalty": 1.1,
        }
        
        if llm_device == "mps":
            generation_kwargs["max_new_tokens"] = 120
        elif llm_device == "cuda":
            generation_kwargs["max_new_tokens"] = 200
        
        with torch.no_grad():
            try:
                outputs = model.generate(**inputs, **generation_kwargs)
            except Exception as gen_error:
                logger.error(f"Edit generation failed on {llm_device}: {gen_error}")
                # Use fallback editing
                await websocket.send_json({"type": "progress", "progress": 80, "message": "Using fallback editing..."})
                edited_code = apply_simple_edits(data['current_diagram_code'], data['edit_instructions'])
                svg_content = render_diagram(edited_code, data["diagram_type"], MERMAID_PORT)
                diagram_storage["edited"]["code"] = edited_code
                diagram_storage["edited"]["svg"] = base64.b64encode(svg_content.encode()).decode()
                diagram_storage["edited"]["type"] = data["diagram_type"]
                await websocket.send_json({"type": "complete"})
                return
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract edited code
        generated_part = response[len(edit_prompt):].strip()
        edited_code = clean_generated_code(generated_part, data['diagram_type'])
        
        if edited_code is None:
            # Use simple fallback editing
            edited_code = apply_simple_edits(data['current_diagram_code'], data['edit_instructions'])
        
        await websocket.send_json({"type": "progress", "progress": 90, "message": "Rendering edited diagram..."})
        
        # Render edited diagram with configured mermaid port
        svg_content = await asyncio.get_event_loop().run_in_executor(
            None, 
            render_diagram,
            edited_code,
            data["diagram_type"],
            MERMAID_PORT  # Pass the configured port
        )
        
        # Store edited results
        diagram_storage["edited"]["code"] = edited_code
        diagram_storage["edited"]["svg"] = base64.b64encode(svg_content.encode()).decode()
        diagram_storage["edited"]["type"] = data["diagram_type"]
        
        await websocket.send_json({"type": "complete"})
        
    except Exception as e:
        logger.error(f"Edit error: {e}")
        await websocket.send_json({"type": "error", "message": str(e)})
    finally:
        await websocket.close()

def apply_simple_edits(code, instructions):
    """Apply simple text-based edits to diagram code"""
    edited_code = code
    instructions_lower = instructions.lower()
    
    # Color changes
    if "blue" in instructions_lower:
        edited_code = edited_code.replace("#3498db", "#2196F3")
        edited_code = edited_code.replace("blue", "#2196F3")
    if "green" in instructions_lower:
        edited_code = edited_code.replace("#2ecc71", "#4CAF50")
        edited_code = edited_code.replace("green", "#4CAF50")
    if "red" in instructions_lower:
        edited_code = edited_code.replace("#3498db", "#F44336")
        edited_code = edited_code.replace("blue", "#F44336")
    
    # Add more nodes if requested
    if "add" in instructions_lower or "more" in instructions_lower:
        if "flowchart TD" in edited_code:
            edited_code = edited_code.replace("D[Output]", "D[Analysis]\n    D --> E[Output]")
    
    return edited_code

# REST endpoints
@app.post("/get-diagram")
async def get_diagram(data: dict = None):
    """Return stored diagram data"""
    try:
        is_edited = data and data.get("is_edited", False) if data else False
        diagram_key = "edited" if is_edited else "original"
        
        if diagram_storage[diagram_key]["code"]:
            return {
                "diagram_code": diagram_storage[diagram_key]["code"],
                "rendered_image": diagram_storage[diagram_key]["svg"],
                "diagram_type": diagram_storage[diagram_key]["type"]
            }
        else:
            return {
                "diagram_code": "# No diagram generated yet",
                "rendered_image": "",
                "diagram_type": "mermaid"
            }
    except Exception as e:
        logger.error(f"Get diagram error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate-diagram-direct")
async def generate_diagram_direct(request: DiagramGenerationRequest):
    """Direct diagram generation without WebSocket"""
    try:
        logger.info(f"Direct generation request: {request.diagram_type} - {request.diagram_style}")
        logger.info(f"Requested model: {request.llm_model}")
        
        # Generate code
        diagram_code = generate_diagram_code(
            request.scientific_text,
            request.diagram_type,
            request.diagram_style,
            request.style_preferences,
            request.llm_model
        )
        
        # Render SVG with configured mermaid port
        svg_content = render_diagram(diagram_code, request.diagram_type, MERMAID_PORT)
        
        # Store results
        diagram_storage["original"]["code"] = diagram_code
        diagram_storage["original"]["svg"] = base64.b64encode(svg_content.encode()).decode()
        diagram_storage["original"]["type"] = request.diagram_type
        
        return {
            "diagram_code": diagram_code,
            "rendered_image": diagram_storage["original"]["svg"],
            "diagram_type": request.diagram_type
        }
        
    except Exception as e:
        logger.error(f"Direct generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/edit-diagram-direct")
async def edit_diagram_direct(request: DiagramEditRequest):
    """Direct diagram editing without WebSocket"""
    try:
        # Create edit prompt
        edit_prompt = f"""Modify this {request.diagram_type} diagram code:

Current code:
{request.current_diagram_code}

Instructions: {request.edit_instructions}

Generate the modified code:

"""
        
        model_name = get_model_name_from_key(request.llm_model)
        
        # FIXED: Force load the requested model for direct editing too
        try:
            if model_name != current_model_name:
                logger.info(f"🔄 Loading model for direct editing: {model_name}")
                model, tokenizer = reload_llm_model(model_name)
            else:
                model, tokenizer = load_llm_model(model_name)
        except Exception as model_error:
            logger.error(f"❌ Failed to load model for direct editing: {model_error}")
            # Use fallback editing
            edited_code = apply_simple_edits(request.current_diagram_code, request.edit_instructions)
            svg_content = render_diagram(edited_code, request.diagram_type, MERMAID_PORT)
            diagram_storage["edited"]["code"] = edited_code
            diagram_storage["edited"]["svg"] = base64.b64encode(svg_content.encode()).decode()
            diagram_storage["edited"]["type"] = request.diagram_type
            return {
                "diagram_code": edited_code,
                "rendered_image": diagram_storage["edited"]["svg"],
                "diagram_type": request.diagram_type
            }
        
        inputs = tokenizer(edit_prompt, return_tensors="pt", truncation=True, max_length=1024, padding=True)
        if llm_device != "cpu":
            try:
                inputs = {k: v.to(llm_device) for k, v in inputs.items()}
            except Exception as e:
                logger.warning(f"Failed to move inputs to {llm_device}: {e}")
        
        # Device-specific generation parameters
        generation_kwargs = {
            "max_new_tokens": 150,
            "temperature": 0.2,
            "do_sample": True,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        
        if llm_device == "mps":
            generation_kwargs["max_new_tokens"] = 120
        elif llm_device == "cuda":
            generation_kwargs["max_new_tokens"] = 200
        
        with torch.no_grad():
            try:
                outputs = model.generate(**inputs, **generation_kwargs)
            except Exception as gen_error:
                logger.error(f"Direct edit generation failed on {llm_device}: {gen_error}")
                # Use fallback editing
                edited_code = apply_simple_edits(request.current_diagram_code, request.edit_instructions)
                svg_content = render_diagram(edited_code, request.diagram_type, MERMAID_PORT)
                diagram_storage["edited"]["code"] = edited_code
                diagram_storage["edited"]["svg"] = base64.b64encode(svg_content.encode()).decode()
                diagram_storage["edited"]["type"] = request.diagram_type
                return {
                    "diagram_code": edited_code,
                    "rendered_image": diagram_storage["edited"]["svg"],
                    "diagram_type": request.diagram_type
                }
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract and clean code
        generated_part = response[len(edit_prompt):].strip()
        edited_code = clean_generated_code(generated_part, request.diagram_type)
        
        if edited_code is None:
            edited_code = apply_simple_edits(request.current_diagram_code, request.edit_instructions)
        
        # Render with configured mermaid port
        svg_content = render_diagram(edited_code, request.diagram_type, MERMAID_PORT)
        
        # Store
        diagram_storage["edited"]["code"] = edited_code
        diagram_storage["edited"]["svg"] = base64.b64encode(svg_content.encode()).decode()
        diagram_storage["edited"]["type"] = request.diagram_type
        
        return {
            "diagram_code": edited_code,
            "rendered_image": diagram_storage["edited"]["svg"],
            "diagram_type": request.diagram_type
        }
        
    except Exception as e:
        logger.error(f"Direct edit error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint with device information"""
    return {
        "status": "healthy",
        "model_loaded": llm_model is not None,
        "current_model": current_model_name,
        "model_device": llm_device,
        "device_info": {
            "mps_available": torch.backends.mps.is_available(),
            "cuda_available": torch.cuda.is_available(),
            "system": platform.system(),
            "architecture": platform.machine()
        },
        "pytorch_version": torch.__version__,
        "api_port": DEFAULT_API_PORT,
        "mermaid_port": MERMAID_PORT,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/")
async def root():
    """Root endpoint with API information"""
    device_emoji = "🚀" if llm_device in ["mps", "cuda"] else "💻"
    device_name = llm_device.upper() if llm_device else "Unknown"
    
    return {
        "message": f"Scientific Diagram Generation API with MPS Support {device_emoji}",
        "llm_model_loaded": llm_model is not None,
        "current_model": current_model_name,
        "device": f"{device_name} ({device_emoji})",
        "device_capabilities": {
            "mps_available": torch.backends.mps.is_available(),
            "cuda_available": torch.cuda.is_available(),
            "current_device": llm_device
        },
        "supported_types": ["mermaid", "graphviz", "plantuml"],
        "supported_models": ["codegen-350m", "codegen-2b", "deepseek-coder", "distilgpt2"],
        "endpoints": [
            "/generate-diagram", 
            "/edit-diagram", 
            "/get-diagram",
            "/generate-diagram-direct",
            "/edit-diagram-direct",
            "/health"
        ],
        "system_info": {
            "platform": platform.system(),
            "architecture": platform.machine(),
            "pytorch_version": torch.__version__
        },
        "port_configuration": {
            "api_port": DEFAULT_API_PORT,
            "mermaid_port": MERMAID_PORT
        },
        "version": "2.5.0-Port-Management",
        "timestamp": datetime.now().isoformat()
    }

def check_mermaid_dependencies():
    """Check and install mermaid-cli dependencies"""
    try:
        # Check if mermaid-cli is available
        result = subprocess.run(['which', 'mmdc'], capture_output=True, text=True)
        if result.returncode != 0:
            logger.warning("mermaid-cli not found. Install with: npm install -g @mermaid-js/mermaid-cli")
            return False
        
        # Check if playwright browsers are installed
        result = subprocess.run(['mmdc', '--version'], capture_output=True, text=True)
        if result.returncode != 0 and 'playwright install' in result.stderr:
            logger.warning("Playwright browsers not installed. Installing...")
            try:
                subprocess.run(['playwright', 'install'], check=True)
                logger.info("Playwright browsers installed successfully")
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to install playwright browsers: {e}")
                logger.info("Please run manually: playwright install")
                return False
        
        return True
    except Exception as e:
        logger.error(f"Error checking mermaid dependencies: {e}")
        return False


if __name__ == "__main__":
    import uvicorn
    
    # Parse command line arguments
    args = parse_arguments()
    
    # Set global mermaid port
    MERMAID_PORT = args.mermaid_port
    
    logger.info("Starting Scientific Diagram Generation Server with MPS Support...")
    logger.info(f"🌐 API Server will run on {args.host}:{args.port}")
    logger.info(f"🎨 Mermaid CLI will use port {MERMAID_PORT}")
    
    # Check mermaid dependencies
    logger.info("Checking mermaid dependencies...")
    mermaid_available = check_mermaid_dependencies()
    if not mermaid_available:
        logger.warning("⚠️ Mermaid rendering may not work properly. Fallback SVG will be used.")
    else:
        logger.info("✅ Mermaid dependencies are ready")
    
    logger.info("Detecting optimal device...")
    
    # Log device capabilities at startup
    log_device_info() 

    try:
        # FIXED: Pre-load the 2B model by default (matching frontend expectation)
        default_model = "Salesforce/codegen-2B-nl"  # Changed from 350M to 2B
        load_llm_model(default_model)
        device_info = f"on {llm_device.upper()}" if llm_device else "on Unknown device"
        logger.info(f"🎉 Model pre-loaded successfully {device_info}!")
        logger.info(f"🎯 Loaded model: {current_model_name}")
        
        if llm_device == "mps":
            logger.info("🍎 Apple Silicon GPU acceleration is active!")
        elif llm_device == "cuda":
            logger.info("🔥 NVIDIA GPU acceleration is active!")
        else:
            logger.info("💻 Running on CPU - consider upgrading for better performance")
            
    except Exception as e:
        logger.error(f"Failed to pre-load model: {e}")
        logger.info("Model will be loaded on first request")
    
    # Start server with configurable port
    logger.info(f"🚀 Starting server on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)