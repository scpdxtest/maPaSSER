import requests           # For HTTP requests to Ollama
import json               # For parsing LLM responses
import networkx as nx     # For creating and managing the graph data structure
import ipycytoscape       # For interactive in-notebook graph visualization
import ipywidgets         # For interactive elements
import pandas as pd       # For displaying data in tables
import os                 # For accessing environment variables (safer for API keys)
import math               # For basic math operations
import re                 # For basic text cleaning (regular expressions)
import warnings           # To suppress potential deprecation warnings

# Configure settings for better display and fewer warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
pd.set_option('display.max_rows', 100) # Show more rows in pandas tables
pd.set_option('display.max_colwidth', 150) # Show more text width in pandas tables

print("Libraries imported successfully.")

# --- Define LLM Model --- 
# Choose an Ollama-compatible model
llm_model_name = "llama3.1"  # or "mistral", "gemma", etc.
print(f"Using Ollama model: {llm_model_name}")

# --- Define LLM Call Parameters ---
llm_temperature = 0.0  # Keep same temperature for deterministic output
llm_max_tokens = 20000  # Note: Not all Ollama models respect this exactly

# --- Retrieve Credentials --- 
api_key = os.getenv("OPENAI_API_KEY")
base_url = os.getenv("OPENAI_API_BASE") # Will be None if not set (e.g., for standard OpenAI)

# --- FOR TESTING ONLY (Less Secure - Replace with Environment Variables) --- 
# Uncomment and set these lines ONLY if you cannot set environment variables easily.
# api_key = "YOUR_API_KEY_HERE"  # <--- PASTE KEY HERE FOR TESTING ONLY
# base_url = "YOUR_API_BASE_URL_HERE" # <--- PASTE BASE URL HERE (if needed)
# Example for Nebius:
# base_url="https://api.studio.nebius.com/v1/"
# api_key="YOUR_NEBIUS_KEY"

print(f"Retrieved API Key: {'Set' if api_key else 'Not Set'}")
print(f"Retrieved Base URL: {base_url if base_url else 'Not Set (will use default OpenAI)'}")

# --- Ollama Configuration --- 
ollama_base_url = "http://localhost:11434"  # Default Ollama URL
# You can set a custom URL via environment variable
if os.getenv("OLLAMA_BASE_URL"):
    ollama_base_url = os.getenv("OLLAMA_BASE_URL")

print(f"Using Ollama at: {ollama_base_url}")

# Check Ollama connection
try:
    models_response = requests.get(f"{ollama_base_url}/api/tags")
    if models_response.status_code == 200:
        available_models = [model['name'] for model in models_response.json()['models']]
        print(f"Connected to Ollama. Available models: {', '.join(available_models[:5])}")
        if len(available_models) > 5:
            print(f"...and {len(available_models) - 5} more")
    else:
        print(f"Warning: Connected to Ollama but couldn't fetch models. Status: {models_response.status_code}")
except Exception as e:
    print(f"Error connecting to Ollama: {e}")
    print("Check if Ollama is running on your machine.")
    raise SystemExit("Ollama connection failed.")

# --- Validate Key and Initialize Client --- 
try:
    # 2. Make the API Call to Ollama
    print("2. Sending request to Ollama...")
    response = requests.post(
        f"{ollama_base_url}/api/generate",
        json={
            "model": llm_model_name,
            # "prompt": extraction_system_prompt + "\n\n" + user_prompt,
            "temperature": llm_temperature,
            "max_tokens": llm_max_tokens,
            "format": "json"  # Request JSON if model supports it
        }
    )
    
    if response.status_code != 200:
        raise Exception(f"Ollama API returned status code {response.status_code}")
    
    response_data = response.json()
    print("   Ollama response received.", response.json())
    
    # 3. Extract Raw Response Content
    print("3. Extracting raw response content...")
    llm_output = response_data.get("response", "").strip()
    print("--- Raw LLM Output (Chunk {chunk_num}) ---")
    print(llm_output)
    print("-" * 20)

except Exception as e:
    error_message = str(e)
    print(f"   ERROR during API call: {error_message}")
    failed_chunks.append({'chunk_number': chunk_num, 'error': f'API/Processing Error: {error_message}', 'response': ''})

# --- Define LLM Call Parameters ---
llm_temperature = 0.0 # Lower temperature for more deterministic, factual output. 0.0 is best for extraction.
llm_max_tokens = 20000 # Max tokens for the LLM response (adjust based on model limits)

print(f"LLM Temperature set to: {llm_temperature}")
print(f"LLM Max Tokens set to: {llm_max_tokens}")

def query_ollama(prompt, model=llm_model_name, temperature=0, max_tokens=20000):
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": False,  # Force complete response
                "format": "json"
            }
        )
        response.raise_for_status()
        return response.json().get("response", "")
    except Exception as e:
        print(f"Error querying Ollama: {e}")
        return None

# Use the same prompts from your existing code
# full_prompt = "extraction_system_prompt" + "\n\n" + user_prompt
result = query_ollama("")
print('Test result: ', result)

unstructured_text = """
Marie Curie, born Maria Skłodowska in Warsaw, Poland, was a pioneering physicist and chemist.
She conducted groundbreaking research on radioactivity. Together with her husband, Pierre Curie,
she discovered the elements polonium and radium. Marie Curie was the first woman to win a Nobel Prize,
the first person and only woman to win the Nobel Prize twice, and the only person to win the Nobel Prize
in two different scientific fields. She won the Nobel Prize in Physics in 1903 with Pierre Curie
and Henri Becquerel. Later, she won the Nobel Prize in Chemistry in 1911 for her work on radium and
polonium. During World War I, she developed mobile radiography units, known as 'petites Curies',
to provide X-ray services to field hospitals. Marie Curie died in 1934 from aplastic anemia, likely
caused by her long-term exposure to radiation.

Marie was born on November 7, 1867, to a family of teachers who valued education. She received her
early schooling in Warsaw but moved to Paris in 1891 to continue her studies at the Sorbonne, where
she earned degrees in physics and mathematics. She met Pierre Curie, a professor of physics, in 1894, 
and they married in 1895, beginning a productive scientific partnership. Following Pierre's tragic 
death in a street accident in 1906, Marie took over his teaching position, becoming the first female 
professor at the Sorbonne.

The Curies' work on radioactivity was conducted in challenging conditions, in a poorly equipped shed 
with no proper ventilation, as they processed tons of pitchblende ore to isolate radium. Marie Curie
established the Curie Institute in Paris, which became a major center for medical research. She had
two daughters: Irène, who later won a Nobel Prize in Chemistry with her husband, and Eve, who became
a writer. Marie's notebooks are still radioactive today and are kept in lead-lined boxes. Her legacy
includes not only her scientific discoveries but also her role in breaking gender barriers in academia
and science.
"""

print("--- Input Text Loaded ---")
print(unstructured_text)
print("-" * 25)
# Basic stats visualization
char_count = len(unstructured_text)
word_count = len(unstructured_text.split())
print(f"Total characters: {char_count}")
print(f"Approximate word count: {word_count}")
print("-" * 25)

# --- Chunking Configuration ---
chunk_size = 150  # Number of words per chunk (adjust as needed)
overlap = 30     # Number of words to overlap (must be < chunk_size)

print(f"Chunk Size set to: {chunk_size} words")
print(f"Overlap set to: {overlap} words")

# --- Basic Validation ---
if overlap >= chunk_size and chunk_size > 0:
    print(f"Error: Overlap ({overlap}) must be smaller than chunk size ({chunk_size}).")
    raise SystemExit("Chunking configuration error.")
else:
    print("Chunking configuration is valid.")

words = unstructured_text.split()
total_words = len(words)

print(f"Text split into {total_words} words.")
# Visualize the first 20 words
print(f"First 20 words: {words[:20]}")

chunks = []
start_index = 0
chunk_number = 1

print(f"Starting chunking process...")

while start_index < total_words:
    end_index = min(start_index + chunk_size, total_words)
    chunk_text = " ".join(words[start_index:end_index])
    chunks.append({"text": chunk_text, "chunk_number": chunk_number})
    
    # print(f"  Created chunk {chunk_number}: words {start_index} to {end_index-1}") # Uncomment for detailed log
    
    # Calculate the start of the next chunk
    next_start_index = start_index + chunk_size - overlap
    
    # Ensure progress is made
    if next_start_index <= start_index:
        if end_index == total_words:
             break # Already processed the last part
        next_start_index = start_index + 1 
         
    start_index = next_start_index
    chunk_number += 1
    
    # Safety break (optional)
    if chunk_number > total_words: # Simple safety
        print("Warning: Chunking loop exceeded total word count, breaking.")
        break

print(f"\nText successfully split into {len(chunks)} chunks.")

print("--- Chunk Details ---")
if chunks:
    # Create a DataFrame for better visualization
    chunks_df = pd.DataFrame(chunks)
    chunks_df['word_count'] = chunks_df['text'].apply(lambda x: len(x.split()))
    display(chunks_df[['chunk_number', 'word_count', 'text']])
else:
    print("No chunks were created (text might be shorter than chunk size).")
print("-" * 25)

# --- System Prompt: Sets the context/role for the LLM --- 
extraction_system_prompt = """
You are an AI expert specialized in knowledge graph extraction. 
Your task is to identify and extract ALL factual Subject-Predicate-Object (SPO) triples from the given text.
Focus on accuracy and adhere strictly to the JSON output format requested in the user prompt.
Extract core entities and the most direct relationship.
"""

# --- User Prompt Template: Contains specific instructions and the text --- 
extraction_user_prompt_template = """
Please extract ALL Subject-Predicate-Object (S-P-O) triples from the text below.

**VERY IMPORTANT RULES:**
1.  **Output Format:** Respond ONLY with a single, valid JSON array. Each element MUST be an object with keys "subject", "predicate", "object".
2.  **JSON Only:** Do NOT include any text before or after the JSON array (e.g., no 'Here is the JSON:' or explanations). Do NOT use markdown ```json ... ``` tags.
3.  **Concise Predicates:** Keep the 'predicate' value concise (1-3 words, ideally 1-2). Use verbs or short verb phrases (e.g., 'discovered', 'was born in', 'won').
4.  **Lowercase:** ALL values for 'subject', 'predicate', and 'object' MUST be lowercase.
5.  **Pronoun Resolution:** Replace pronouns (she, he, it, her, etc.) with the specific lowercase entity name they refer to based on the text context (e.g., 'marie curie').
6.  **Specificity:** Capture specific details (e.g., 'nobel prize in physics' instead of just 'nobel prize' if specified).
7.  **Completeness:** Extract all distinct factual relationships mentioned.

**Text to Process:**
```text
{text_chunk}
```

**Required JSON Output Format Example:**
[
  {{ "subject": "marie curie", "predicate": "discovered", "object": "radium" }},
  {{ "subject": "marie curie", "predicate": "won", "object": "nobel prize in physics" }}
]

**Your JSON Output MUST start with '[' and end with ']'**
"""

print("--- System Prompt ---")
print(extraction_system_prompt)
print("\n" + "-" * 25 + "\n")

print("--- User Prompt Template (Structure) ---")
# Show structure, replacing the placeholder for clarity
print(extraction_user_prompt_template.replace("{text_chunk}", "[... text chunk goes here ...]"))
print("\n" + "-" * 25 + "\n")

# Show an example of the *actual* prompt that will be sent for the first chunk
print("--- Example Filled User Prompt (for Chunk 1) ---")
if chunks:
    example_filled_prompt = extraction_user_prompt_template.format(text_chunk=chunks[0]['text'])
    # Displaying a limited portion for brevity
    print(example_filled_prompt[:600] + "\n[... rest of chunk text ...]\n" + example_filled_prompt[-200:])
else:
    print("No chunks available to create an example filled prompt.")
print("\n" + "-" * 25)

# Initialize lists to store results and failures
all_extracted_triples = []
failed_chunks = []

print(f"Starting triple extraction from {len(chunks)} chunks using model '{llm_model_name}'...")
# We will process chunks one by one in the following cells.

import time  # Make sure to add this import

def query_ollama_complete(prompt, model=llm_model_name, temperature=0, max_tokens=32000):
    """Call Ollama with better streaming to get ALL triples - no CLI fallback."""
    print(f"Sending text to model {model} (improved approach)...")
    
    # Enhance the prompt with stronger instructions for completeness
    enhanced_prompt = prompt + "\n\nCRITICAL INSTRUCTION: You MUST extract ALL possible triples from the text. Return at least 10-15 triples in a complete JSON array. DO NOT stop early."
    
    # Strategy 1: Improved streaming approach with better chunk handling
    try:
        print("Using improved streaming approach...")
        full_response = ""
        complete_json = False
        
        # Add context parameter to continue from previous response if needed
        stream_response = requests.post(
            f"{ollama_base_url}/api/generate",
            json={
                "model": model,
                "prompt": enhanced_prompt,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": True
            },
            stream=True,
            timeout=180  # 3 minute timeout
        )
        
        chunk_count = 0
        json_array_started = False
        bracket_count = 0
        last_chunk_time = time.time()
        
        # Process the stream with better tracking of JSON structure
        for chunk in stream_response.iter_lines():
            if not chunk:
                if time.time() - last_chunk_time > 20:  # 20 second timeout
                    print("No data received for 20 seconds, checking if response is complete...")
                    # Check if we have a complete JSON array
                    if json_array_started and "]" in full_response:
                        print("JSON array appears complete despite no 'done' signal.")
                        break
                    elif len(full_response) > 500:  # Arbitrary threshold
                        print("Response seems substantial. Proceeding with parsing.")
                        break
                continue
                
            chunk_count += 1
            last_chunk_time = time.time()
            
            try:
                chunk_data = json.loads(chunk.decode('utf-8'))
                
                if "response" in chunk_data:
                    chunk_text = chunk_data["response"]
                    full_response += chunk_text
                    
                    # Track JSON array structure
                    if "[" in chunk_text and not json_array_started:
                        json_array_started = True
                        print("JSON array started")
                    
                    # Count brackets to detect potential completion
                    if json_array_started:
                        for char in chunk_text:
                            if char == '[': bracket_count += 1
                            elif char == ']': bracket_count -= 1
                        
                        # If brackets are balanced and we've started the array, we might be done
                        if bracket_count <= 0 and len(full_response) > 200:
                            print("JSON structure appears complete based on brackets")
                            complete_json = True
                
                if "done" in chunk_data and chunk_data["done"] == True:
                    print("Received done=True signal")
                    break
                
                # Progress updates
                if chunk_count % 5 == 0:
                    print(f"Received {chunk_count} chunks, length: {len(full_response)}")
                
                # Early exit if we detect a complete JSON array with substantial content
                if complete_json and len(full_response) > 500:
                    print("Detected complete JSON structure, finishing stream")
                    break
                    
            except json.JSONDecodeError:
                print(f"Warning: Failed to parse chunk")
        
        print(f"Streaming complete. Received {chunk_count} chunks, {len(full_response)} characters")
        
        # If we got substantial content, try to parse it
        if len(full_response) > 200:
            extracted_response = extract_json_from_text(full_response)
            if extracted_response and len(extracted_response) > 100:
                return extracted_response
        
        # If streaming didn't give us good results, try alternative approach
        print("Streaming response may not be complete. Trying alternative approach...")
    
    except Exception as e:
        print(f"Streaming approach failed: {e}")
        full_response = ""  # Reset in case we need it later
    
    # Strategy 2: Non-streaming approach with multiple model-specific optimizations
    try:
        print("\nTrying non-streaming approach with format=json...")
        
        # Further enhance the prompt for this approach
        format_prompt = enhanced_prompt + "\n\nYOUR RESPONSE MUST BE A VALID JSON ARRAY WITH AT LEAST 10 TRIPLES. Format: [{\"subject\": \"...\", \"predicate\": \"...\", \"object\": \"...\"}, ...]. NO explanations, ONLY JSON."
        
        response = requests.post(
            f"{ollama_base_url}/api/generate",
            json={
                "model": model,
                "prompt": format_prompt,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": False
                # "format": "json"  # Request JSON format
            },
            timeout=300  # 5 minute timeout
        )
        
        if response.status_code == 200:
            result = response.json().get("response", "")
            print(f"Non-streaming response received: {len(result)} characters")
            
            if result and len(result) > 100:
                extracted = extract_json_from_text(result)
                if extracted and "[" in extracted:
                    return extracted
        
        print("Standard non-streaming approach didn't return valid results")
        
        # Strategy 3: Final attempt with raw option
        print("\nTrying final approach with raw=true...")
        response = requests.post(
            f"{ollama_base_url}/api/generate",
            json={
                "model": model,
                "prompt": format_prompt,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": False,
                "raw": True
            },
            timeout=300  # 5 minute timeout
        )
        
        if response.status_code == 200:
            result = response.json().get("response", "")
            print(f"Raw mode response received: {len(result)} characters")
            return extract_json_from_text(result)
            
    except Exception as e:
        print(f"Alternative approaches failed: {e}")
    
    # Return whatever we have as best effort
    print("All approaches completed. Returning best available result.")
    return extract_json_from_text(full_response) if full_response else "[]"

def extract_json_from_text(text):
    """Improved function to extract valid JSON from text with fallbacks."""
    if not text or len(text) < 10:
        return "[]"
    
    # Method 1: Find complete array with regex
    array_match = re.search(r'\[\s*{.*}\s*\]', text, re.DOTALL)
    if array_match:
        try:
            extracted = array_match.group(0)
            json.loads(extracted)  # Validate it's actually valid JSON
            print(f"Successfully extracted JSON array ({len(extracted)} chars)")
            return extracted
        except json.JSONDecodeError:
            print("Found array-like text but it's not valid JSON")
    
    # Method 2: Find individual JSON objects and build array
    json_objects = []
    object_pattern = r'{[^{}]*"subject"[^{}]*"predicate"[^{}]*"object"[^{}]*}'
    for match in re.finditer(object_pattern, text, re.DOTALL):
        try:
            obj_text = match.group(0)
            # Clean up common issues
            obj_text = obj_text.replace("'", '"').replace('\\"', '"')
            obj = json.loads(obj_text)
            if "subject" in obj and "predicate" in obj and "object" in obj:
                json_objects.append(obj)
        except json.JSONDecodeError:
            continue
    
    if json_objects:
        print(f"Constructed array from {len(json_objects)} individual objects")
        return json.dumps(json_objects)
    
    # Method 3: Handle partial arrays by finding the portion between [ and ]
    try:
        bracket_start = text.find('[')
        if bracket_start >= 0:
            bracket_end = text.rfind(']')
            if bracket_end > bracket_start:
                array_text = text[bracket_start:bracket_end+1]
                # Try to fix common JSON errors
                array_text = array_text.replace("'", '"').replace('\\"', '"')
                # Try to parse it
                json.loads(array_text)
                print(f"Extracted partial JSON array ({len(array_text)} chars)")
                return array_text
    except json.JSONDecodeError:
        print("Found brackets but content isn't valid JSON")
    
    # No valid JSON found, return empty array
    print("Could not extract valid JSON, returning empty array")
    return "[]"

# --- Process ALL Chunks - Full Cycle Implementation ---
# This replaces the single-chunk demonstration with a loop that processes all chunks

# Initialize lists to store results and failures if not already done
if 'all_extracted_triples' not in locals():
    all_extracted_triples = []
if 'failed_chunks' not in locals():
    failed_chunks = []

print(f"Starting triple extraction from ALL {len(chunks)} chunks using model '{llm_model_name}'...")

import time  # Ensure time is imported

# Process each chunk
for chunk_index in range(len(chunks)):
    chunk_info = chunks[chunk_index]
    chunk_text = chunk_info['text']
    chunk_num = chunk_info['chunk_number']
    
    print(f"\n--- Processing Chunk {chunk_num}/{len(chunks)} --- ")
    
    # 1. Format the User Prompt
    print("1. Formatting User Prompt...")
    
    # Define the system and user prompts
    system_prompt = """
    You are an AI expert specialized in knowledge graph extraction. 
    Your task is to identify and extract ALL factual Subject-Predicate-Object (SPO) triples from the given text. 
    Focus on accuracy and adhere strictly to the JSON output format requested in the user prompt. 
    Extract core entities and the most direct relationship."""
    
    user_prompt = """
    Please extract ALL Subject-Predicate-Object (S-P-O) triples from the text below.

    **VERY IMPORTANT RULES:**
    1.  **Output Format:** Respond ONLY with a single, valid JSON array. Each element MUST be an object with keys "subject", "predicate", "object".
    2.  **JSON Only:** Do NOT include any text before or after the JSON array (e.g., no 'Here is the JSON:' or explanations). Do NOT use markdown ```json ... ``` tags.
    3.  **Concise Predicates:** Keep the 'predicate' value concise (1-3 words, ideally 1-2). Use verbs or short verb phrases (e.g., 'discovered', 'was born in', 'won').
    4.  **Lowercase:** ALL values for 'subject', 'predicate', and 'object' MUST be lowercase.
    5.  **Pronoun Resolution:** Replace pronouns (she, he, it, her, etc.) with the specific lowercase entity name they refer to based on the text context (e.g., 'marie curie').
    6.  **Specificity:** Capture specific details (e.g., 'nobel prize in physics' instead of just 'nobel prize' if specified).
    7.  **Completeness:** Extract all distinct factual relationships mentioned."""
    
    text_to_process = """
    **Text to Process:**
    ```
    """
    
    instruction_prompt = """
    **Required JSON Output Format Example:**
    [
      { "subject": "marie curie", "predicate": "discovered", "object": "radium" },
      { "subject": "marie curie", "predicate": "won", "object": "nobel prize in physics" }
    ]

    **Your JSON Output MUST start with '[' and end with ']':**
    """
    
    # Combine all parts
    combined_prompt = system_prompt + "\n" + user_prompt + "\n" + text_to_process + chunk_text + "\n```\n" + instruction_prompt
    combined_prompt += "\n\nEXTREMLY IMPORTANT: Extract ALL triples from the text! Return at least 10-15 triples if present. Be comprehensive!"
    
    # For debugging, uncomment this line:
    # print(f"Combined prompt (first 200 chars): {combined_prompt[:200]}...")
    
    llm_output = None
    error_message = None
    
    try:
        # 2. Make the API Call to Ollama
        print("2. Sending request to Ollama...")
        response = query_ollama_complete(combined_prompt, model=llm_model_name, temperature=0, max_tokens=32000)
        print(f"Response received, length: {len(response) if response else 0} characters")
        
        # 3. Extract Response Content
        print("3. Extracting response content...")
        llm_output = response
        print(f"--- Raw LLM Output Preview (Chunk {chunk_num}) ---")
        print(llm_output[:200] + "..." if llm_output and len(llm_output) > 200 else llm_output)
        print("-" * 20)
        
    except Exception as e:
        error_message = str(e)
        print(f"   ERROR during API call: {error_message}")
        failed_chunks.append({'chunk_number': chunk_num, 'error': f'API/Processing Error: {error_message}', 'response': ''})
    
    # 4. Parse JSON (if API call succeeded)
    parsed_json = None
    parsing_error = None
    if llm_output is not None:
        print("4. Attempting to parse JSON from response...")
        try:
            # Strategy 1: Direct parsing (ideal)
            try:
                parsed_data = json.loads(llm_output)
            except json.JSONDecodeError as e:
                # Handle cases where there are multiple JSON objects or extra data
                print(f"   JSONDecodeError: {e}. Attempting to split and parse...")
                potential_jsons = llm_output.split("\n")
                parsed_data = []
                for potential_json in potential_jsons:
                    try:
                        parsed_data.append(json.loads(potential_json))
                    except json.JSONDecodeError:
                        print(f"      Skipping invalid JSON segment: {potential_json[:100]}...")
                # Flatten the list if it contains multiple JSON objects
                if len(parsed_data) == 1:
                    parsed_data = parsed_data[0]

            # Handle if response_format={'type':'json_object'} returns a dict containing the list
            if isinstance(parsed_data, dict):
                print("   Detected dictionary response, attempting to extract list...")
                list_values = [v for v in parsed_data.values() if isinstance(v, list)]
                print(f"      Found {len(list_values)} list(s) in dictionary.")
                if len(list_values) == 1:
                    parsed_json = list_values[0]
                    print("      Successfully extracted list from dictionary.")
                else:
                    raise ValueError("JSON object received, but doesn't contain a single list of triples.")
            elif isinstance(parsed_data, list):
                parsed_json = parsed_data
                print("   Successfully parsed JSON list directly.")
            else:
                 raise ValueError("Parsed JSON is not a list or expected dictionary wrapper.")

        except json.JSONDecodeError as json_err:
            parsing_error = f"JSONDecodeError: {json_err}. Trying regex fallback..."
            print(f"   {parsing_error}")
            # Strategy 2: Regex fallback for arrays potentially wrapped in text/markdown
            match = re.search(r'^\s*(\[.*?\])\s*$', llm_output, re.DOTALL)
            if match:
                json_string_extracted = match.group(1)
                print("      Regex found potential JSON array structure.")
                try:
                    parsed_json = json.loads(json_string_extracted)
                    print("      Successfully parsed JSON from regex extraction.")
                    parsing_error = None # Clear previous error
                except json.JSONDecodeError as nested_err:
                    parsing_error = f"JSONDecodeError after regex: {nested_err}"
                    print(f"      ERROR: Regex content is not valid JSON: {nested_err}")
            else:
                 parsing_error = "JSONDecodeError and Regex fallback failed."
                 print("      ERROR: Regex could not find JSON array structure.")
                 
        except ValueError as val_err:
             parsing_error = f"ValueError: {val_err}" # Catches issues with unexpected structure
             print(f"   ERROR: {parsing_error}")

        # --- Show Parsed Result (or error) ---
        if parsed_json is not None:
            print(f"--- Parsed JSON Data (Chunk {chunk_num}) ---")
            print(f"Found {len(parsed_json)} triples")
            # Only print the first 3 triples to avoid cluttering the output
            if parsed_json and len(parsed_json) > 0:
                print(json.dumps(parsed_json[:3], indent=2) + f"\n... and {len(parsed_json)-3} more triples")
            print("-" * 20)
        else:
            print(f"--- JSON Parsing FAILED (Chunk {chunk_num}) --- ")
            print(f"   Final Parsing Error: {parsing_error}")
            print("-" * 20)
            failed_chunks.append({'chunk_number': chunk_num, 'error': f'Parsing Failed: {parsing_error}', 'response': llm_output})

    # 5. Validate and Store Triples (if parsing succeeded)
    if parsed_json is not None:
        print("5. Validating structure and extracting triples...")
        valid_triples_in_chunk = []
        invalid_entries = []
        if isinstance(parsed_json, list):
            for item in parsed_json:
                if isinstance(item, dict) and all(k in item for k in ['subject', 'predicate', 'object']):
                    # Basic check: ensure values are strings (can be refined)
                    if all(isinstance(item[k], str) for k in ['subject', 'predicate', 'object']):
                        item['chunk'] = chunk_num # Add source chunk info
                        valid_triples_in_chunk.append(item)
                    else:
                        invalid_entries.append({'item': item, 'reason': 'Non-string value'}) 
                else:
                    invalid_entries.append({'item': item, 'reason': 'Incorrect structure/keys'})
        else:
            print("   ERROR: Parsed data is not a list, cannot extract triples.")
            invalid_entries.append({'item': parsed_json, 'reason': 'Not a list'})
            # Also add to failed chunks if the overall structure was wrong
            if not any(fc['chunk_number'] == chunk_num for fc in failed_chunks):
                 failed_chunks.append({'chunk_number': chunk_num, 'error': 'Parsed data not a list', 'response': llm_output})
        
        # --- Show Validation Results --- 
        print(f"   Found {len(valid_triples_in_chunk)} valid triples in this chunk.")
        if invalid_entries:
             print(f"   Skipped {len(invalid_entries)} invalid entries.")
             # print(f"   Invalid entries details: {invalid_entries}") # Uncomment for debugging
             
        # --- Display Valid Triples from this Chunk --- 
        if valid_triples_in_chunk:
             print(f"--- Valid Triples Extracted (Chunk {chunk_num}) ---")
             # Display using pandas - but limit to 10 rows to avoid large outputs
             display_df = pd.DataFrame(valid_triples_in_chunk)
             if len(display_df) > 10:
                 print(f"Showing 10 of {len(display_df)} triples:")
                 display(display_df.head(10))
             else:
                 display(display_df)
             print("-" * 20)
             # Add to the main list
             all_extracted_triples.extend(valid_triples_in_chunk)
        else:
             print(f"--- No valid triples extracted from this chunk. ---")
             print("-" * 20)

    # --- Update Running Total (Visual Feedback) ---
    print(f"--- Running Total Triples Extracted: {len(all_extracted_triples)} --- ")
    print(f"--- Failed Chunks So Far: {len(failed_chunks)} --- ")
    
    # Optional: Add a brief pause between chunks to avoid rate limiting
    if chunk_index < len(chunks) - 1:
        print(f"Pausing briefly before processing next chunk...")
        time.sleep(1)  # 1 second pause

print("\nFinished processing ALL chunks.")
print(f"Total triples extracted: {len(all_extracted_triples)}")
print(f"Total chunks failed: {len(failed_chunks)}")

# --- Summary of Extraction (Reflecting state after the single chunk demo) ---
print(f"\n--- Overall Extraction Summary ---")
print(f"Total chunks defined: {len(chunks)}")
processed_chunks = len(chunks) - len(failed_chunks) # Approximation if loop isn't run fully
print(f"Chunks processed (attempted): {processed_chunks + len(failed_chunks)}") # Chunks we looped through
print(f"Total valid triples extracted across all processed chunks: {len(all_extracted_triples)}")
print(f"Number of chunks that failed API call or parsing: {len(failed_chunks)}")

if failed_chunks:
    print("\nDetails of Failed Chunks:")
    for failure in failed_chunks:
        print(f"  Chunk {failure['chunk_number']}: Error: {failure['error']}")
        # print(f"    Response (start): {failure.get('response', '')[:100]}...") # Uncomment for more detail
print("-" * 25)

# Display all extracted triples using Pandas
print("\n--- All Extracted Triples (Before Normalization) ---")
if all_extracted_triples:
    all_triples_df = pd.DataFrame(all_extracted_triples)
    display(all_triples_df)
else:
    print("No triples were successfully extracted.")
print("-" * 25)

# Initialize lists and tracking variables
normalized_triples = []
seen_triples = set() # Tracks (subject, predicate, object) tuples
original_count = len(all_extracted_triples)
empty_removed_count = 0
duplicates_removed_count = 0

print(f"Starting normalization and de-duplication of {original_count} triples...")

print("Processing triples for normalization (showing first 5 examples):")
example_limit = 5
processed_count = 0

for i, triple in enumerate(all_extracted_triples):
    show_example = (i < example_limit)
    if show_example:
        print(f"\n--- Example {i+1} ---")
        print(f"Original Triple (Chunk {triple.get('chunk', '?')}): {triple}")
        
    subject_raw = triple.get('subject')
    predicate_raw = triple.get('predicate')
    object_raw = triple.get('object')
    chunk_num = triple.get('chunk', 'unknown')
    
    triple_valid = False
    normalized_sub, normalized_pred, normalized_obj = None, None, None

    if isinstance(subject_raw, str) and isinstance(predicate_raw, str) and isinstance(object_raw, str):
        # 1. Normalize
        normalized_sub = subject_raw.strip().lower()
        normalized_pred = re.sub(r'\s+', ' ', predicate_raw.strip().lower()).strip()
        normalized_obj = object_raw.strip().lower()
        if show_example:
            print(f"Normalized: SUB='{normalized_sub}', PRED='{normalized_pred}', OBJ='{normalized_obj}'")

        # 2. Filter Empty
        if normalized_sub and normalized_pred and normalized_obj:
            triple_identifier = (normalized_sub, normalized_pred, normalized_obj)
            
            # 3. De-duplicate
            if triple_identifier not in seen_triples:
                normalized_triples.append({
                    'subject': normalized_sub,
                    'predicate': normalized_pred,
                    'object': normalized_obj,
                    'source_chunk': chunk_num
                })
                seen_triples.add(triple_identifier)
                triple_valid = True
                if show_example:
                    print("Status: Kept (New Unique Triple)")
            else:
                duplicates_removed_count += 1
                if show_example:
                    print("Status: Discarded (Duplicate)")
        else:
            empty_removed_count += 1
            if show_example:
                print("Status: Discarded (Empty component after normalization)")
    else:
        empty_removed_count += 1 # Count non-string/missing as needing removal
        if show_example:
             print("Status: Discarded (Non-string or missing component)")
    processed_count += 1

print(f"\n... Finished processing {processed_count} triples.")

# --- Summary of Normalization --- 
print(f"\n--- Normalization & De-duplication Summary ---")
print(f"Original extracted triple count: {original_count}")
print(f"Triples removed (empty/invalid components): {empty_removed_count}")
print(f"Duplicate triples removed: {duplicates_removed_count}")
final_count = len(normalized_triples)
print(f"Final unique, normalized triple count: {final_count}")
print("-" * 25)

# Display a sample of normalized triples using Pandas
print("\n--- Final Normalized Triples ---")
if normalized_triples:
    normalized_df = pd.DataFrame(normalized_triples)
    display(normalized_df)
else:
    print("No valid triples remain after normalization.")
print("-" * 25)

# Create an empty directed graph
knowledge_graph = nx.DiGraph()

print("Initialized an empty NetworkX DiGraph.")
# Visualize the initial empty graph state
print("--- Initial Graph Info ---")
try:
    # Try the newer method first
    print(nx.info(knowledge_graph))
except AttributeError:
    # Fallback for different NetworkX versions
    print(f"Type: {type(knowledge_graph).__name__}")
    print(f"Number of nodes: {knowledge_graph.number_of_nodes()}")
    print(f"Number of edges: {knowledge_graph.number_of_edges()}")
print("-" * 25)

print("Adding triples to the NetworkX graph...")

added_edges_count = 0
update_interval = 5 # How often to print graph info update

if not normalized_triples:
    print("Warning: No normalized triples to add to the graph.")
else:
    for i, triple in enumerate(normalized_triples):
        subject_node = triple['subject']
        object_node = triple['object']
        predicate_label = triple['predicate']
        
        # Nodes are added automatically when adding edges, but explicit calls are fine too
        # knowledge_graph.add_node(subject_node) 
        # knowledge_graph.add_node(object_node)
        
        # Add the directed edge with the predicate as a 'label' attribute
        knowledge_graph.add_edge(subject_node, object_node, label=predicate_label)
        added_edges_count += 1
        
        # --- Visualize Graph Growth --- 
        if (i + 1) % update_interval == 0 or (i + 1) == len(normalized_triples):
            print(f"\n--- Graph Info after adding Triple #{i+1} --- ({subject_node} -> {object_node})")
            try:
                # Try the newer method first
                print(nx.info(knowledge_graph))
            except AttributeError:
                # Fallback for different NetworkX versions
                print(f"Type: {type(knowledge_graph).__name__}")
                print(f"Number of nodes: {knowledge_graph.number_of_nodes()}")
                print(f"Number of edges: {knowledge_graph.number_of_edges()}")
            # For very large graphs, printing info too often can be slow. Adjust interval.

print(f"\nFinished adding triples. Processed {added_edges_count} edges.")

# --- Final Graph Statistics --- 
num_nodes = knowledge_graph.number_of_nodes()
num_edges = knowledge_graph.number_of_edges()

print(f"\n--- Final NetworkX Graph Summary ---")
print(f"Total unique nodes (entities): {num_nodes}")
print(f"Total unique edges (relationships): {num_edges}")

if num_edges != added_edges_count and isinstance(knowledge_graph, nx.DiGraph):
     print(f"Note: Added {added_edges_count} edges, but graph has {num_edges}. DiGraph overwrites edges with same source/target. Use MultiDiGraph if multiple edges needed.")

if num_nodes > 0:
    try:
       density = nx.density(knowledge_graph)
       print(f"Graph density: {density:.4f}")
       if nx.is_weakly_connected(knowledge_graph):
           print("The graph is weakly connected (all nodes reachable ignoring direction).")
       else:
           num_components = nx.number_weakly_connected_components(knowledge_graph)
           print(f"The graph has {num_components} weakly connected components.")
    except Exception as e:
        print(f"Could not calculate some graph metrics: {e}") # Handle potential errors on empty/small graphs
else:
    print("Graph is empty, cannot calculate metrics.")
print("-" * 25)

# --- Sample Nodes --- 
print("\n--- Sample Nodes (First 10) ---")
if num_nodes > 0:
    nodes_sample = list(knowledge_graph.nodes())[:10]
    display(pd.DataFrame(nodes_sample, columns=['Node Sample']))
else:
    print("Graph has no nodes.")

# --- Sample Edges --- 
print("\n--- Sample Edges (First 10 with Labels) ---")
if num_edges > 0:
    edges_sample = []
    for u, v, data in list(knowledge_graph.edges(data=True))[:10]:
        edges_sample.append({'Source': u, 'Target': v, 'Label': data.get('label', 'N/A')})
    display(pd.DataFrame(edges_sample))
else:
    print("Graph has no edges.")
print("-" * 25)

print("Preparing interactive visualization...")

# --- Check Graph Validity for Visualization --- 
can_visualize = False
if 'knowledge_graph' not in locals() or not isinstance(knowledge_graph, nx.Graph):
    print("Error: 'knowledge_graph' not found or is not a NetworkX graph.")
elif knowledge_graph.number_of_nodes() == 0:
    print("NetworkX Graph is empty. Cannot visualize.")
else:
    print(f"Graph seems valid for visualization ({knowledge_graph.number_of_nodes()} nodes, {knowledge_graph.number_of_edges()} edges).")
    can_visualize = True

cytoscape_nodes = []
cytoscape_edges = []

if can_visualize:
    print("Converting nodes...")
    # Calculate degrees for node sizing
    node_degrees = dict(knowledge_graph.degree())
    max_degree = max(node_degrees.values()) if node_degrees else 1
    
    for node_id in knowledge_graph.nodes():
        degree = node_degrees.get(node_id, 0)
        # Simple scaling for node size (adjust logic as needed)
        node_size = 15 + (degree / max_degree) * 50 if max_degree > 0 else 15
        
        cytoscape_nodes.append({
            'data': {
                'id': str(node_id), # ID must be string
                'label': str(node_id).replace(' ', '\n'), # Display label (wrap spaces)
                'degree': degree,
                'size': node_size,
                'tooltip_text': f"Entity: {str(node_id)}\nDegree: {degree}" # Tooltip on hover
            }
        })
    print(f"Converted {len(cytoscape_nodes)} nodes.")
    
    print("Converting edges...")
    edge_count = 0
    for u, v, data in knowledge_graph.edges(data=True):
        edge_id = f"edge_{edge_count}" # Unique edge ID
        predicate_label = data.get('label', '')
        cytoscape_edges.append({
            'data': {
                'id': edge_id,
                'source': str(u),
                'target': str(v),
                'label': predicate_label, # Label on edge
                'tooltip_text': f"Relationship: {predicate_label}" # Tooltip on hover
            }
        })
        edge_count += 1
    print(f"Converted {len(cytoscape_edges)} edges.")
    
    # Combine into the final structure
    cytoscape_graph_data = {'nodes': cytoscape_nodes, 'edges': cytoscape_edges}
    
    # Visualize the converted structure (first few nodes/edges)
    print("\n--- Sample Cytoscape Node Data (First 2) ---")
    print(json.dumps(cytoscape_graph_data['nodes'][:2], indent=2))
    print("\n--- Sample Cytoscape Edge Data (First 2) ---")
    print(json.dumps(cytoscape_graph_data['edges'][:2], indent=2))
    print("-" * 25)
else:
     print("Skipping data conversion as graph is not valid for visualization.")
     cytoscape_graph_data = {'nodes': [], 'edges': []}

if can_visualize:
    print("Creating ipycytoscape widget...")
    cyto_widget = ipycytoscape.CytoscapeWidget()
    print("Widget created.")
    
    print("Loading graph data into widget...")
    cyto_widget.graph.add_graph_from_json(cytoscape_graph_data, directed=True)
    print("Data loaded.")
else:
    print("Skipping widget creation.")
    cyto_widget = None

if cyto_widget:
    print("Defining enhanced colorful and interactive visual style...")
    # More vibrant and colorful styling with a modern color scheme
    visual_style = [
        {
            'selector': 'node',
            'style': {
                'label': 'data(label)',
                'width': 'data(size)',
                'height': 'data(size)',
                'background-color': '#3498db',  # Bright blue
                'background-opacity': 0.9,
                'color': '#ffffff',             # White text
                'font-size': '12px',
                'font-weight': 'bold',
                'text-valign': 'center',
                'text-halign': 'center',
                'text-wrap': 'wrap',
                'text-max-width': '100px',
                'text-outline-width': 2,
                'text-outline-color': '#2980b9',  # Matching outline
                'text-outline-opacity': 0.7,
                'border-width': 3,
                'border-color': '#1abc9c',      # Turquoise border
                'border-opacity': 0.9,
                'shape': 'ellipse',
                'transition-property': 'background-color, border-color, border-width, width, height',
                'transition-duration': '0.3s',
                'tooltip-text': 'data(tooltip_text)'
            }
        },
        {
            'selector': 'node:selected',
            'style': {
                'background-color': '#e74c3c',  # Pomegranate red
                'border-width': 4,
                'border-color': '#c0392b',
                'text-outline-color': '#e74c3c',
                'width': 'data(size) * 1.2',    # Enlarge selected nodes
                'height': 'data(size) * 1.2'
            }
        },
        {
            'selector': 'node:hover',
            'style': {
                'background-color': '#9b59b6',  # Purple on hover
                'border-width': 4,
                'border-color': '#8e44ad',
                'cursor': 'pointer',
                'z-index': 999
            }
        },
        {
            'selector': 'edge',
            'style': {
                'label': 'data(label)',
                'width': 2.5,
                'curve-style': 'bezier',
                'line-color': '#2ecc71',         # Green
                'line-opacity': 0.8,
                'target-arrow-color': '#27ae60',
                'target-arrow-shape': 'triangle',
                'arrow-scale': 1.5,
                'font-size': '10px',
                'font-weight': 'normal',
                'color': '#2c3e50',
                'text-background-opacity': 0.9,
                'text-background-color': '#ecf0f1',
                'text-background-shape': 'roundrectangle',
                'text-background-padding': '3px',
                'text-rotation': 'autorotate',
                'edge-text-rotation': 'autorotate',
                'transition-property': 'line-color, width, target-arrow-color',
                'transition-duration': '0.3s',
                'tooltip-text': 'data(tooltip_text)'
            }
        },
        {
            'selector': 'edge:selected',
            'style': {
                'line-color': '#f39c12',         # Yellow-orange
                'target-arrow-color': '#d35400',
                'width': 4,
                'text-background-color': '#f1c40f',
                'color': '#ffffff',               # White text
                'z-index': 998
            }
        },
        {
            'selector': 'edge:hover',
            'style': {
                'line-color': '#e67e22',         # Orange on hover
                'width': 3.5,
                'cursor': 'pointer',
                'target-arrow-color': '#d35400',
                'z-index': 997
            }
        },
        {
            'selector': '.center-node',
            'style': {
                'background-color': '#16a085',    # Teal
                'background-opacity': 1,
                'border-width': 4,
                'border-color': '#1abc9c',        # Turquoise border
                'border-opacity': 1
            }
        }
    ]
    
    print("Setting enhanced visual style on widget...")
    cyto_widget.set_style(visual_style)
    
    # Apply a better animated layout
    cyto_widget.set_layout(name='cose', 
                          nodeRepulsion=5000, 
                          nodeOverlap=40, 
                          idealEdgeLength=120, 
                          edgeElasticity=200, 
                          nestingFactor=6, 
                          gravity=90, 
                          numIter=2500,
                          animate=True,
                          animationDuration=1000,
                          initialTemp=300,
                          coolingFactor=0.95)
    
    # Add a special class to main nodes (Marie Curie)
    if len(cyto_widget.graph.nodes) > 0:
        main_nodes = [node.data['id'] for node in cyto_widget.graph.nodes 
                     if node.data.get('degree', 0) > 10]
        
        # Create gradient styles for center nodes
        for i, node_id in enumerate(main_nodes):
            # Use vibrant colors for center nodes
            center_style = {
                'selector': f'node[id = "{node_id}"]',
                'style': {
                    'background-color': '#9b59b6',   # Purple
                    'background-opacity': 0.95,
                    'border-width': 4,
                    'border-color': '#8e44ad',      # Darker purple border
                    'border-opacity': 1,
                    'text-outline-width': 3,
                    'text-outline-color': '#8e44ad',
                    'font-size': '14px'
                }
            }
            visual_style.append(center_style)
        
        # Update the style with the new additions
        cyto_widget.set_style(visual_style)
    
    print("Enhanced colorful and interactive style applied successfully.")
else:
    print("Skipping style definition.")

if cyto_widget:
    print("Setting layout algorithm ('cose')...")
    # cose (Compound Spring Embedder) is often good for exploring connections
    cyto_widget.set_layout(name='cose', 
                           animate=True, 
                           # Adjust parameters for better spacing/layout
                           nodeRepulsion=4000, # Increase repulsion 
                           nodeOverlap=40,    # Increase overlap avoidance
                           idealEdgeLength=120, # Slightly longer ideal edges
                           edgeElasticity=150, 
                           nestingFactor=5, 
                           gravity=100,        # Increase gravity slightly
                           numIter=1500,      # More iterations
                           initialTemp=200,
                           coolingFactor=0.95,
                           minTemp=1.0)
    print("Layout set. The graph will arrange itself when displayed.")
else:
     print("Skipping layout setting.")

# For VS Code Jupyter extension
import ipywidgets
print(f"ipywidgets version: {ipywidgets.__version__}")
import ipycytoscape
print(f"ipycytoscape version: {ipycytoscape.__version__}")

# Try rendering a simple widget to test if widgets work at all
from ipywidgets import Button
button = Button(description="Test Button")
display(button)

if cyto_widget:
    print("Displaying interactive graph widget below...")
    print("Interact: Zoom (scroll), Pan (drag background), Move Nodes (drag nodes), Hover for details.")
    
    # More explicit display approach
    from IPython.display import display
    display(cyto_widget)
    
    # Try alternative display method if the above doesn't work
    import IPython
    IPython.display.display_html(IPython.display.HTML(
        f"<div>Widget should appear above. If not, try restarting kernel.</div>"
    ))
else:
    print("No widget to display.")

# Add a clear separator
print("\n" + "-" * 25 + "\nEnd of Visualization Step." + "\n" + "-" * 25)

# Fallback visualization using NetworkX's drawing capabilities
if cyto_widget is None or True:  # Force fallback for testing
    print("Attempting fallback static visualization...")
    
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 10))
    
    # Create node color list
    node_colors = ['#3498db' for _ in range(knowledge_graph.number_of_nodes())]
    
    # Highlight high-degree nodes
    node_degrees = dict(knowledge_graph.degree())
    for i, (node, degree) in enumerate(node_degrees.items()):
        if degree > 10:  # High degree nodes
            node_colors[i] = '#9b59b6'  # Purple
    
    # Create edge labels
    edge_labels = {(u, v): d.get('label', '') for u, v, d in knowledge_graph.edges(data=True)}
    
    # Draw the graph
    pos = nx.spring_layout(knowledge_graph, seed=42)  # Consistent layout
    nx.draw_networkx_nodes(knowledge_graph, pos, node_size=700, node_color=node_colors, alpha=0.8)
    nx.draw_networkx_edges(knowledge_graph, pos, width=1.5, alpha=0.7, arrowsize=20, edge_color='#2ecc71')
    nx.draw_networkx_labels(knowledge_graph, pos, font_size=10, font_weight='bold')
    nx.draw_networkx_edge_labels(knowledge_graph, pos, edge_labels=edge_labels, font_size=8)
    
    plt.title("Knowledge Graph (Static Fallback Visualization)")
    plt.axis('off')
    plt.tight_layout()
    plt.show()


# --- Generate Training Examples from Knowledge Graph --- NEW PART UNTESTED ---
def generate_training_examples(knowledge_graph, template_variety=3):
    """Convert knowledge graph triples to natural language training examples"""
    training_examples = []
    
    # Templates for converting triples to natural language
    templates = [
        # Question -> Answer templates
        {"q": "What is the relationship between {subject} and {object}?", 
         "a": "The relationship is that {subject} {predicate} {object}."},
        {"q": "Tell me about {subject}.", 
         "a": "{subject} {predicate} {object}."},
        {"q": "What do you know about {object}?", 
         "a": "{subject} {predicate} {object}."},
        
        # Statement -> Completion templates
        {"input": "{subject} {predicate}", 
         "output": "{object}"},
        {"input": "The {predicate} of {subject} is", 
         "output": "{object}"},
    ]
    
    # Process each triple in the graph
    for u, v, data in knowledge_graph.edges(data=True):
        subject = u
        object = v
        predicate = data.get('label', '')
        
        # Skip invalid triples
        if not (subject and predicate and object):
            continue
            
        # Generate variety of examples using different templates
        for i in range(min(template_variety, len(templates))):
            template = templates[i]
            
            if "q" in template:  # QA pair
                question = template["q"].format(subject=subject, predicate=predicate, object=object)
                answer = template["a"].format(subject=subject, predicate=predicate, object=object)
                training_examples.append({
                    "question": question,
                    "answer": answer,
                    "type": "qa"
                })
            else:  # Text completion
                input_text = template["input"].format(subject=subject, predicate=predicate, object=object)
                output_text = template["output"].format(subject=subject, predicate=predicate, object=object)
                training_examples.append({
                    "input": input_text,
                    "output": output_text,
                    "type": "completion"
                })
    
    return training_examples

def prepare_finetuning_dataset(training_examples, format="alpaca"):
    """Convert training examples to popular finetuning formats"""
    
    if format == "alpaca":
        # Alpaca format
        return [
            {
                "instruction": ex["question"] if ex["type"] == "qa" else f"Complete the following: {ex['input']}",
                "input": "",
                "output": ex["answer"] if ex["type"] == "qa" else ex["output"]
            }
            for ex in training_examples
        ]
    
    elif format == "llama":
        # Llama chat format
        return [
            {
                "messages": [
                    {"role": "user", "content": ex["question"] if ex["type"] == "qa" else ex["input"]},
                    {"role": "assistant", "content": ex["answer"] if ex["type"] == "qa" else ex["output"]}
                ]
            }
            for ex in training_examples
        ]
    
    # Add other formats as needed
    
    return training_examples

import json

def export_dataset(dataset, file_path):
    """Export dataset to JSONL file for finetuning"""
    with open(file_path, 'w', encoding='utf-8') as f:
        for example in dataset:
            f.write(json.dumps(example) + '\n')
    print(f"Dataset exported to {file_path}")

# Example usage
training_examples = generate_training_examples(knowledge_graph)
dataset = prepare_finetuning_dataset(training_examples, format="alpaca")
export_dataset(dataset, "knowledge_graph_finetuning.jsonl")


