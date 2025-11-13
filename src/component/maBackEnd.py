import pyntelope
import json
import os
import nltk
import torch
import numpy as np
import hashlib
import logging
import sys
from logging.handlers import RotatingFileHandler
from nltk.translate.meteor_score import single_meteor_score
from rouge import Rouge 
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.lm import Laplace
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.lm import Lidstone
from transformers import BertTokenizer, BertModel
from scipy.spatial.distance import cityblock  # Manhattan Distance
from scipy.spatial.distance import jaccard
from scipy.stats import spearmanr, pearsonr

# import chromadb

from flask import Flask, request, jsonify
from flask_cors import CORS

# ============================================================
# LOGGING CONFIGURATION
# ============================================================
# Get the directory where the script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_FILE = os.path.join(SCRIPT_DIR, 'maBackEnd.log')

# Create a custom logger
logger = logging.getLogger('maBackEnd')
logger.setLevel(logging.INFO)

# Create handlers
# File handler with rotation (max 10MB per file, keep 5 backup files)
file_handler = RotatingFileHandler(
    LOG_FILE,
    maxBytes=10*1024*1024,  # 10MB
    backupCount=5,
    encoding='utf-8'
)
file_handler.setLevel(logging.INFO)

# Console handler (to also see logs in terminal/systemd journal)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)

# Create formatter
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Add handlers to logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# Redirect print statements to logger
class LoggerWriter:
    def __init__(self, logger_func):
        self.logger_func = logger_func
        self.buffer = ''
    
    def write(self, message):
        if message != '\n':
            self.buffer += message
            if '\n' in self.buffer:
                lines = self.buffer.split('\n')
                for line in lines[:-1]:
                    if line.strip():
                        self.logger_func(line.strip())
                self.buffer = lines[-1]
        else:
            if self.buffer.strip():
                self.logger_func(self.buffer.strip())
                self.buffer = ''
    
    def flush(self):
        if self.buffer.strip():
            self.logger_func(self.buffer.strip())
            self.buffer = ''

# Redirect stdout and stderr to logger
sys.stdout = LoggerWriter(logger.info)
sys.stderr = LoggerWriter(logger.error)

logger.info("="*60)
logger.info("maBackEnd service starting...")
logger.info(f"Log file: {LOG_FILE}")
logger.info(f"Script directory: {SCRIPT_DIR}")
logger.info("="*60)

# Patch SQLite3 to use pysqlite3 if available (for ChromaDB compatibility)
try:
    import pysqlite3.dbapi2 as sqlite3_new
    import sys
    sys.modules['sqlite3'] = sqlite3_new
    sys.modules['sqlite3.dbapi2'] = sqlite3_new
    print(f"Using pysqlite3 with SQLite3 version: {sqlite3_new.sqlite_version}")
except ImportError:
    print("pysqlite3 not available, using system sqlite3")

app = Flask(__name__)
CORS(app)

@app.route('/chromacollections', methods=['GET'])
def get_chroma_collections():
    """
    Get all ChromaDB collection names from the specified URL
    Query parameter: url - ChromaDB server URL (e.g., http://127.0.0.1:8000)
    """
    try:
        # Get the ChromaDB URL from query parameters
        chroma_url = request.args.get('url')
        if not chroma_url:
            return jsonify({'error': 'URL parameter is required'}), 400
        
        print(f"Fetching collections from ChromaDB at: {chroma_url}")
        
        # Try different approaches to connect to ChromaDB
        
        # Method 1: Try direct HTTP API call first (avoids chromadb library issues)
        try:

            # chroma_client = chromadb.HttpClient(host='localhost', port=8000)
            # coll = chroma_client.list_collections()

            import requests
            api_endpoints = [
                f"{chroma_url}/api/v1/collections",
                f"{chroma_url}/api/v2/collections",
                f"{chroma_url}/api/v2/tenants/default_tenant/databases/default_database/collections",
                f"{chroma_url}/collections",
                f"{chroma_url}/api/collections"
            ]
            
            for endpoint in api_endpoints:
                try:
                    response = requests.get(endpoint, timeout=10)
                    print(f"Trying endpoint: {endpoint} - Status: {response.status_code}")
                    if response.status_code == 200:
                        data = response.json()
                        print(f"Successfully fetched from HTTP API: {endpoint}")
                        print(f"Response data: {data}")
                        
                        # Parse different response formats
                        collections = []
                        if isinstance(data, list):
                            collections = data
                        elif isinstance(data, dict) and 'collections' in data:
                            collections = data['collections']
                        elif isinstance(data, dict) and 'result' in data:
                            collections = data['result']
                        elif isinstance(data, dict) and 'data' in data:
                            collections = data['data']
                        
                        collection_names = []
                        for collection in collections:
                            if isinstance(collection, dict):
                                name = collection.get('name', collection.get('id', str(collection)))
                                id = collection.get('id', collection.get('id', str(collection)))
                                collection_names.append({'name': name, 'id': id})
                            else:
                                collection_names.append({'name': str(collection), 'id': str(collection)})
                        
                        print(f"Parsed {len(collection_names)} collections: {[c['name'] for c in collection_names]}")
                        
                        return jsonify({
                            'success': True,
                            'collections': collection_names,
                            'count': len(collection_names),
                            'method': 'HTTP_API',
                            'endpoint_used': endpoint,
                            'resp': response.json()
                        }), 200
                except requests.RequestException as e:
                    print(f"HTTP API attempt failed for {endpoint}: {e}")
                    continue
                    
        except ImportError:
            print("requests library not available")
                
        # If all methods fail
        return jsonify({
            'error': 'All connection methods failed',
            'tried_methods': ['HTTP_API', 'CHROMADB_CLIENT'],
            'solutions': [
                {
                    'method': 'Use local database path',
                    'description': 'Use a local file path like: ./chroma_db',
                    'example': 'Try URL: ./chroma_db or /path/to/chroma/data'
                },
                {
                    'method': 'Start ChromaDB HTTP server',
                    'command': 'chroma run --host localhost --port 8000',
                    'description': 'Start ChromaDB as HTTP server, then use: http://localhost:8000'
                },
                {
                    'method': 'Use Docker',
                    'command': 'docker run -p 8000:8000 chromadb/chroma',
                    'description': 'Run ChromaDB in Docker, then use: http://localhost:8000'
                }
            ],
            'suggestion': 'Try using a local database path first, or start ChromaDB server'
        }), 500
        
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        return jsonify({'error': f'Unexpected error: {str(e)}'}), 500

@app.route('/getnames', methods=['GET'])
def get_test_names():
    """
    Get unique test names from the unique_test_names collection
    Optional query parameter: prefix - filter testids starting with this prefix
    """
    try:
        prefix = request.args.get('prefix', '')
        
        # Connect to MongoDB
        from pymongo import MongoClient
        mongo_client = MongoClient('mongodb://your-mongodb-server-ip:port/')
        db = mongo_client['myDB']
        collection_name = 'unique_test_names'
        
        # Check if collection exists
        if collection_name not in db.list_collection_names():
            mongo_client.close()
            return jsonify({'error': 'Collection does not exist.'}), 404
        
        unique_tests_collection = db[collection_name]
        
        # Build query filter
        query_filter = {}
        if prefix:
            query_filter['testid'] = {'$regex': f'^{prefix}', '$options': 'i'}
        
        # Retrieve unique testids, sorted by last_updated (most recent first)
        documents = list(
            unique_tests_collection
            .find(query_filter, {'_id': 0, 'testid': 1})
            .sort('last_updated', -1)
        )
        
        # Close the MongoDB connection
        mongo_client.close()
        
        print(f"✅ Found {len(documents)} unique test names{' with prefix: ' + prefix if prefix else ''}")
        
        # Return the documents as JSON
        return jsonify(documents), 200
        
    except Exception as e:
        print(f"❌ Error retrieving test names: {str(e)}")
        return jsonify({'error': f'Error: {str(e)}'}), 500

@app.route('/metrics', methods=['POST'])
def metrics():
    data = request.json
    if not data:
        return jsonify({'error': 'JSON data is missing.'}), 400

    reference = data.get('reference')
    if not reference:
        return jsonify({'error': 'Reference parameter is missing.'}), 400
    candidate = data.get('candidate')
    if not candidate:
        return jsonify({'error': 'Candidate parameter is missing.'}), 400
    userID = data.get('userID')
    if not userID:
        return jsonify({'error': 'userID parameter is missing.'}), 400
    testID = data.get('testID')
    if not testID:
        return jsonify({'error': 'testID parameter is missing.'}), 400
    description = data.get('description')
    if not description:
        return jsonify({'error': 'Description parameter is missing.'}), 400
    
    # NEW: Get optional additional metrics array
    additional_metrics = data.get('additional_metrics', None)
    
    print("reference ---> ", reference)
    print("candidate ---> ", candidate)
    print("userID ---> ", userID)
    print("testID ---> ", testID)
    print("description ---> ", description)
    if additional_metrics:
        print("additional_metrics ---> ", additional_metrics)
        print(f"additional_metrics count: {len(additional_metrics)}")

    metrics_result = calc_metrics(reference, candidate, userID, testID, description, additional_metrics)
    return jsonify({
        'message': 'Metrics calculated successfully.',
        'metrics': metrics_result['metrics'],
        'checksum': metrics_result['checksum']
    })

def calculate_checksum(data_array):
    """
    Calculate SHA-256 checksum of the data array
    Converts all values to strings and concatenates them for hashing
    
    Args:
        data_array: List of numerical values
        
    Returns:
        str: SHA-256 hash in hexadecimal format
    """
    # Convert all values to strings with consistent formatting for reproducibility
    # Use repr() to ensure floating point precision is maintained
    data_string = ''.join([repr(value) for value in data_array])
    
    # Calculate SHA-256 hash
    checksum = hashlib.sha256(data_string.encode('utf-8')).hexdigest()
    
    return checksum

def calc_metrics(reference: str, candidate: str, userID: str, testID: str, description: str, additional_metrics: list = None) -> dict:

    meteor_score = single_meteor_score(reference.split(), candidate.split())
    print("METEOR", meteor_score)

    hypothesis = reference
    ref = candidate

    rouge = Rouge()
    rouge_scores = rouge.get_scores(hypothesis, ref)

    print("ROUGE", rouge_scores)
    
   # Reference and candidate sentences should be tokenized
    reference_blue = reference.split()
    candidate_blue = candidate.split()

    # Create a smoothing function
    smoothie = SmoothingFunction().method4

    # Calculate BLEU score with smoothing
    blue_score = sentence_bleu([reference_blue], candidate_blue, smoothing_function=smoothie)

    print("BLEU", blue_score)

    # Example text paragraph
    text_paragraph = reference
    # Tokenize the text into sentences
    tokenized_text = [list(map(str.lower, word_tokenize(sent))) for sent in nltk.sent_tokenize(text_paragraph)]

    # Train-test split (in a real scenario, you should use separate training and testing sets)
    train_data, vocab = padded_everygram_pipeline(2, tokenized_text)  # Example uses a bigram model

    # Train an n-gram model with Laplace smoothing (add-one smoothing)
    model = Laplace(2)  # Using a bigram model for this example
    model.fit(train_data, vocab)

    # Function to calculate perplexity
    def calculate_perplexity(model, text):
        test_data, _ = padded_everygram_pipeline(2, [word_tokenize(text.lower())])
        return model.perplexity(next(test_data))

    # Example text to evaluate
    test_text = candidate

    # Calculate perplexity
    laplace_perplexity = calculate_perplexity(model, test_text)
    print(f"Laplace Perplexity: {laplace_perplexity}")

    # Sample training text
    training_text = reference
    # Tokenizing the training text into sentences and then into words
    tokenized_text = [list(map(str.lower, word_tokenize(sent))) for sent in sent_tokenize(training_text)]

    # Preparing the training data for a trigram model
    n = 3  # Order of the n-gram model
    train_data, padded_sents = padded_everygram_pipeline(n, tokenized_text)

    # Creating and training the language model with Lidstone smoothing
    # gamma is the Lidstone smoothing parameter, typically a small fraction
    gamma = 0.1
    model = Lidstone(order=n, gamma=gamma)
    model.fit(train_data, padded_sents)

    # Sample text paragraph to calculate perplexity
    test_text = candidate
    # Tokenizing the test text
    tokenized_test_text = [list(map(str.lower, word_tokenize(sent))) for sent in sent_tokenize(test_text)]
    # Preparing the test data
    test_data, _ = padded_everygram_pipeline(n, tokenized_test_text)

    # Calculating and printing the perplexity of the test text
    # Perplexity is calculated as the exponentiated negative average log-likelihood of the test set
    lidstone_perplexity = model.perplexity(next(test_data))
    print(f"Lidstone Perplexity of the test text: {lidstone_perplexity}")

    # Function to encode text to BERT embeddings
    def get_bert_embedding(text, tokenizer, model):
        # Tokenize and convert to tensor
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        # Get embeddings
        with torch.no_grad():
            outputs = model(**inputs)
        # Use the [CLS] token embedding for representing the sentence
        return outputs.last_hidden_state[:, 0, :].numpy()

    # Initialize the tokenizer and model for BERT
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')

    # Sample text paragraphs
    text_1 = reference
    text_2 = candidate

    # Tokenize the text paragraphs
    inputs_1 = tokenizer(text_1, return_tensors='pt', padding=True, truncation=True, max_length=512)
    inputs_2 = tokenizer(text_2, return_tensors='pt', padding=True, truncation=True, max_length=512)

    # Generate embeddings
    with torch.no_grad():
        outputs_1 = model(**inputs_1)
        outputs_2 = model(**inputs_2)

    # Get the embeddings for the [CLS] token (used as the aggregate representation for classification tasks)
    embeddings_1 = outputs_1.last_hidden_state[:, 0, :]
    embeddings_2 = outputs_2.last_hidden_state[:, 0, :]

    # Calculate cosine similarity between the two embeddings
    cosine_similarity = torch.nn.functional.cosine_similarity(embeddings_1, embeddings_2)

    # Function to encode text to BERT embeddings
    def get_bert_embedding_manhatan(text, tokenizer, model):
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        # Aggregate embedding representation
        return outputs.last_hidden_state.mean(dim=1).numpy()

    embedding1M = get_bert_embedding_manhatan(text_1, tokenizer, model)
    embedding2M = get_bert_embedding_manhatan(text_2, tokenizer, model)

    # Calculate Pearson Correlation Coefficient
    pearson_corr, _ = pearsonr(embedding1M.flatten(), embedding2M.flatten())

    print(f"Cosine similarity: {cosine_similarity.item()}")
    print(f"Pearson Correlation Coefficient: {pearson_corr}")

    from collections import Counter

    def f1_score(prediction, truth):
        prediction_tokens = prediction.strip().lower().split()
        truth_tokens = truth.strip().lower().split()
        common_tokens = Counter(prediction_tokens) & Counter(truth_tokens)
        num_same = sum(common_tokens.values())

        if num_same == 0:
            return 0

        precision = 1.0 * num_same / len(prediction_tokens)
        recall = 1.0 * num_same / len(truth_tokens)
        f1 = (2 * precision * recall) / (precision + recall)

        return f1

    # Example
    prediction = candidate
    truth = reference

    f1 = f1_score(prediction, truth)

    print(f"F1 Score: {f1:.3f}")

    def compute_bertscore_alternative(predictions, references):
        # You already have BERT model and tokenizer loaded in your code
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertModel.from_pretrained('bert-base-uncased')
        
        # Get embeddings for predictions and references
        pred_embeddings = []
        ref_embeddings = []
        
        # Process each prediction and reference
        for pred, ref in zip([predictions], [references]):
            # Get embeddings
            pred_inputs = tokenizer(pred, return_tensors='pt', padding=True, truncation=True, max_length=512)
            ref_inputs = tokenizer(ref, return_tensors='pt', padding=True, truncation=True, max_length=512)
            
            with torch.no_grad():
                pred_outputs = model(**pred_inputs)
                ref_outputs = model(**ref_inputs)
            
            # Get token-level embeddings
            pred_emb = pred_outputs.last_hidden_state
            ref_emb = ref_outputs.last_hidden_state
            
            # Calculate F1 using cosine similarity
            similarities = torch.nn.functional.cosine_similarity(
                pred_emb.unsqueeze(2), 
                ref_emb.unsqueeze(1), 
                dim=3
            )
            
            # Calculate precision, recall, F1
            precision = similarities.max(dim=2)[0].mean().item()
            recall = similarities.max(dim=1)[0].mean().item()
            f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
            
            return {'precision': precision, 'recall': recall, 'f1': f1}
        
    bert1_score = compute_bertscore_alternative(reference, candidate)
    print("BERT Score", bert1_score)

    def evaluate_bert_rt_score(reference, candidate, tokenizer, model):
        """
        Evaluate text quality using existing BERT model with regression targets
        to approximate Nubia-like scores
        """
        # You already have the tokenizer and model loaded
        
        # Create a combined input for quality assessment
        combined_text = f"Reference: {reference} Candidate: {candidate}"
        
        # Tokenize and get embeddings
        inputs = tokenizer(combined_text, return_tensors='pt', padding=True, truncation=True, max_length=512)
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Get the embeddings for the [CLS] token
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        
        # Simple scoring based on vector norms and similarities
        # We'll use different projections of the embedding to simulate different aspects
        
        # Get base similarity
        ref_inputs = tokenizer(reference, return_tensors='pt', padding=True, truncation=True, max_length=512)
        cand_inputs = tokenizer(candidate, return_tensors='pt', padding=True, truncation=True, max_length=512)
        
        with torch.no_grad():
            ref_outputs = model(**ref_inputs)
            cand_outputs = model(**cand_inputs)
        
        ref_emb = ref_outputs.last_hidden_state[:, 0, :]
        cand_emb = cand_outputs.last_hidden_state[:, 0, :]
        
        # Calculate base similarity
        sim = torch.nn.functional.cosine_similarity(ref_emb, cand_emb).item()
        
        # Calculate different "aspect" scores using vector projections
        # These are simplified approximations of Nubia's scores
        
        # Coherence: how well the text flows (use normalized embedding components)
        coherence = (torch.norm(cls_embedding[:, :100]).item() / 10.0) * sim
        coherence = min(max(coherence, 0.0), 5.0)  # Scale to 0-5
        
        # Consistency: semantic alignment (direct similarity)
        consistency = sim * 5.0  # Scale similarity to 0-5
        
        # Fluency: language quality (use different embedding components)
        fluency = (torch.norm(cls_embedding[:, 100:300]).item() / 15.0) * sim
        fluency = min(max(fluency, 0.0), 5.0)  # Scale to 0-5
        
        # Relevance: how relevant the response is to the reference
        relevance = sim * 5.0  # Scale similarity to 0-5
        
        # Average score
        avg_score = (coherence + consistency + fluency + relevance) / 4.0
        
        return {
            'coherence': coherence,
            'consistency': consistency,
            'fluency': fluency,
            'relevance': relevance,
            'avg_score': avg_score
        }

    # Use the function with your existing BERT model
    bert_rt_score = evaluate_bert_rt_score(reference, candidate, tokenizer, model)
    print("BERT-RT Score", bert_rt_score)

    # Build base results array (24 base NLP metrics)
    res = [
        meteor_score, blue_score, 
        rouge_scores[0]['rouge-1']['r'], rouge_scores[0]['rouge-1']['p'], rouge_scores[0]['rouge-1']['f'],
        rouge_scores[0]['rouge-2']['r'], rouge_scores[0]['rouge-2']['p'], rouge_scores[0]['rouge-2']['f'],
        rouge_scores[0]['rouge-l']['r'], rouge_scores[0]['rouge-l']['p'], rouge_scores[0]['rouge-l']['f'],
        laplace_perplexity, lidstone_perplexity, 
        cosine_similarity.item(),
        pearson_corr, f1, 
        bert1_score['precision'], bert1_score['recall'], bert1_score['f1'], 
        bert_rt_score['coherence'], bert_rt_score['consistency'], bert_rt_score['fluency'], 
        bert_rt_score['relevance'], bert_rt_score['avg_score']
    ]

    base_metrics_count = len(res)
    print(f"📊 Base NLP metrics count: {base_metrics_count}")

    # Append additional metrics if provided
    if additional_metrics and isinstance(additional_metrics, list):
        # Validate that additional metrics are numeric
        validated_metrics = []
        for i, metric in enumerate(additional_metrics):
            try:
                validated_metrics.append(float(metric))
            except (ValueError, TypeError):
                print(f"⚠️ Warning: Additional metric at index {i} is not numeric: {metric}, using 0")
                validated_metrics.append(0.0)
        
        res.extend(validated_metrics)
        print(f"✅ Appended {len(validated_metrics)} additional metrics")
        print(f"📋 Total metrics: {len(res)} (Base: {base_metrics_count} + Additional: {len(validated_metrics)})")
    else:
        print(f"ℹ️ No additional metrics provided, using {base_metrics_count} base metrics only")
    
    # Calculate checksum of the COMPLETE results array
    checksum = calculate_checksum(res)
    print(f"🔐 Results checksum (SHA-256): {checksum}")
    print(f"📦 Final res array length: {len(res)}")

    # At the end, create a structured dictionary instead of a list
    metrics_dict = {
        'meteor_score': meteor_score,
        'bleu_score': blue_score,
        'rouge_1': {
            'recall': rouge_scores[0]['rouge-1']['r'],
            'precision': rouge_scores[0]['rouge-1']['p'],
            'f1': rouge_scores[0]['rouge-1']['f']
        },
        'rouge_2': {
            'recall': rouge_scores[0]['rouge-2']['r'],
            'precision': rouge_scores[0]['rouge-2']['p'],
            'f1': rouge_scores[0]['rouge-2']['f']
        },
        'rouge_l': {
            'recall': rouge_scores[0]['rouge-l']['r'],
            'precision': rouge_scores[0]['rouge-l']['p'],
            'f1': rouge_scores[0]['rouge-l']['f']
        },
        'perplexity': {
            'laplace': laplace_perplexity,
            'lidstone': lidstone_perplexity
        },
        'bert_similarity': {
            'cosine_similarity': cosine_similarity.item(),
            'pearson_correlation': pearson_corr
        },
        'f1_score': f1,
        'bert_score': {
            'precision': bert1_score['precision'],
            'recall': bert1_score['recall'],
            'f1': bert1_score['f1']
        },
        'bert_rt_score': {
            'coherence': bert_rt_score['coherence'],
            'consistency': bert_rt_score['consistency'],
            'fluency': bert_rt_score['fluency'],
            'relevance': bert_rt_score['relevance'],
            'avg_score': bert_rt_score['avg_score']
        }
    }

    # Add additional metrics to the dictionary if provided
    if additional_metrics and isinstance(additional_metrics, list):
        metrics_dict['additional_metrics'] = additional_metrics
        metrics_dict['additional_metrics_count'] = len(additional_metrics)

    from pymongo import MongoClient
    import datetime

    # MongoDB connection setup
    mongo_client = MongoClient('mongodb://your-mongodb-server-ip:port/')
    db = mongo_client['myDB']
    
    current_datetime = datetime.datetime.utcnow()
    
    # ========================================
    # COLLECTION 1: unique_test_names
    # Store ONLY unique testIDs (for the test list)
    # ========================================
    unique_tests_collection = db['unique_test_names']
    
    # Check if collection exists, create it if it doesn't
    if 'unique_test_names' not in db.list_collection_names():
        print(f"Creating new collection: unique_test_names")
        db.create_collection('unique_test_names')
        # Create unique index on testid to prevent duplicates
        db['unique_test_names'].create_index([("testid", 1)], unique=True)
        print(f"Collection unique_test_names created successfully")
    
    # Insert or update ONLY the testID in unique_test_names (upsert)
    try:
        unique_tests_collection.update_one(
            {"testid": testID},  # Filter
            {
                "$set": {
                    "testid": testID,
                    "last_updated": current_datetime
                },
                "$setOnInsert": {
                    "created_at": current_datetime,
                    "creator": userID  # Store the first creator
                }
            },
            upsert=True  # Insert if doesn't exist, update if exists
        )
        print(f"✅ TestID '{testID}' ensured in unique_test_names collection")
    except Exception as e:
        print(f"❌ Error updating unique_test_names: {e}")
    
    # ========================================
    # COLLECTION 2: test_scores
    # Store ALL score records (every test run)
    # ========================================
    scores_collection = db['test_scores']
    
    # Check if collection exists, create it if it doesn't
    if 'test_scores' not in db.list_collection_names():
        print(f"Creating new collection: test_scores")
        db.create_collection('test_scores')
        # Create indexes for better query performance
        db['test_scores'].create_index([("testid", 1)])
        db['test_scores'].create_index([("creator", 1)])
        db['test_scores'].create_index([("created_at", -1)])
        db['test_scores'].create_index([("checksum", 1)])
        print(f"Collection test_scores created successfully")
    
    # Create the score document with ALL data
    score_document = {
        "testid": testID,
        "creator": userID,
        "created_at": current_datetime,
        "results": res,  # Complete results array (base + additional metrics)
        "checksum": checksum,
        "description": description,
        "reference": reference,
        "candidate": candidate
    }
    
    # Add additional metrics info to the document if present
    if additional_metrics and isinstance(additional_metrics, list):
        score_document["additional_metrics_count"] = len(additional_metrics)
        score_document["total_metrics_count"] = len(res)
        score_document["has_multiagent_metrics"] = True
    else:
        score_document["has_multiagent_metrics"] = False
    
    # ALWAYS insert the new score record (no duplicate check)
    try:
        result = scores_collection.insert_one(score_document)
        print(f"✅ Score record inserted with ID: {result.inserted_id}")
        print(f"📊 TestID: {testID}")
        print(f"📋 Metrics count: {len(res)} (Base: 24, Additional: {len(additional_metrics) if additional_metrics else 0})")
        print(f"🔐 Checksum: {checksum}")
    except Exception as e:
        print(f"❌ Error inserting score document: {e}")
    
    # Close MongoDB connection
    mongo_client.close()
    
    print("Create Transaction for blockchain...")
    data=[
        pyntelope.Data(
            name="creator",
            value=pyntelope.types.Name(userID), 
        ),
        pyntelope.Data(
            name="testid",
            value=pyntelope.types.Name(testID), 
        ),
        pyntelope.Data(
            name="description",
            value=pyntelope.types.String(description),
        ),
        pyntelope.Data(
            name="results",
            value=pyntelope.types.Array.from_dict(res, type_=pyntelope.types.Float64),
        ),
    ]

    auth = pyntelope.Authorization(actor="llmtest", permission="active")

    action = pyntelope.Action(
        account="llmtest", # this is the contract account
        name="addtest", # this is the action name
        data=data,
        authorization=[auth],
    )

    raw_transaction = pyntelope.Transaction(actions=[action])

    print("Link transaction to the network")
    net = pyntelope.Net(host = 'http://blockchain2.uni-plovdiv.net:8033')  
    # notice that pyntelope returns a new object instead of change in place
    linked_transaction = raw_transaction.link(net=net)


    print("Sign transaction")
    key = "5HyZQrptLQnoTdjtwfMkPtgH18inm1vkSee8HBKEZHydhB79Tst"
    signed_transaction = linked_transaction.sign(key=key)

    print("Send")
    resp = signed_transaction.send()

    print("Printing the response")
    resp_fmt = json.dumps(resp, indent=4)
    print(f"Response:\n{resp_fmt}")

    # Return both metrics and checksum
    return {
        'metrics': metrics_dict,
        'checksum': checksum
    }

@app.route('/gettestdata', methods=['GET'])
def get_test_data():
    """
    Get test score records from MongoDB with optional filtering
    Query parameters:
    - testid: exact match for testid
    - testid_prefix: get all tests where testid starts with this prefix
    - userid: filter by user
    - limit: number of results to return (default: 1000)
    - sort: field to sort by (default: created_at)
    - order: asc or desc (default: desc)
    """
    try:
        # Get query parameters
        testid = request.args.get('testid')
        testid_prefix = request.args.get('testid_prefix')
        userid = request.args.get('userid')
        limit = int(request.args.get('limit', 1000))
        sort_field = request.args.get('sort', 'created_at')
        sort_order = request.args.get('order', 'desc')
        
        # Connect to MongoDB
        from pymongo import MongoClient
        mongo_client = MongoClient('mongodb://your-mongodb-server-ip:port/')
        db = mongo_client['myDB']
        collection_name = 'test_scores'  # Changed to test_scores collection
        
        # Check if collection exists
        if collection_name not in db.list_collection_names():
            mongo_client.close()
            return jsonify({
                'success': False,
                'error': 'Collection does not exist.',
                'hint': 'Run some tests first to create the test_scores collection'
            }), 404
        
        scores_collection = db[collection_name]
        
        # Build query filter
        query_filter = {}
        
        if testid:
            # Exact match - get ALL records for this testid
            query_filter['testid'] = testid
            print(f"🔍 Fetching ALL score records for testid: {testid}")
        elif testid_prefix:
            # Prefix match using regex - get ALL records matching prefix
            query_filter['testid'] = {'$regex': f'^{testid_prefix}', '$options': 'i'}
            print(f"🔍 Fetching ALL score records with prefix: {testid_prefix}")
        
        if userid:
            query_filter['creator'] = userid
        
        # Build sort order
        sort_direction = -1 if sort_order.lower() == 'desc' else 1
        
        print(f"📊 MongoDB Query: {query_filter}")
        print(f"📊 Sort: {sort_field} {sort_order}")
        print(f"📊 Limit: {limit}")
        
        # Execute query - get ALL matching records
        documents = list(
            scores_collection
            .find(query_filter, {'_id': 0})
            .sort(sort_field, sort_direction)
            .limit(limit)
        )
        
        # Close connection
        mongo_client.close()
        
        print(f"✅ Found {len(documents)} score records")
        if documents:
            print(f"📋 Sample record metrics count: {len(documents[0].get('results', []))}")
            if documents[0].get('checksum'):
                print(f"🔐 Records include checksums")
        
        return jsonify({
            'success': True,
            'count': len(documents),
            'query': query_filter,
            'data': documents
        }), 200
        
    except Exception as e:
        print(f"❌ Error retrieving test data: {str(e)}")
        return jsonify({'error': f'Error: {str(e)}'}), 500

if __name__ == '__main__':
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='maBackEnd - LLM Metrics Backend Service')
    parser.add_argument('--port', type=int, default=8402, help='Port to run the server on (default: 8402)')
    parser.add_argument('--host', type=str, default='195.230.127.227', help='Host to bind to (default: 195.230.127.227)')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode')
    args = parser.parse_args()
    
    logger.info(f"Starting Flask server on {args.host}:{args.port}")
    logger.info(f"Debug mode: {args.debug}")
    
    try:
        app.run(debug=args.debug, host=args.host, port=args.port)
    except KeyboardInterrupt:
        logger.info("Server stopped by user (Ctrl+C)")
    except Exception as e:
        logger.error(f"Server error: {str(e)}", exc_info=True)
        sys.exit(1)
    finally:
        logger.info("maBackEnd service stopped")
        logger.info("="*60)