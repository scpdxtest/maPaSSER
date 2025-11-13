# PaSSER Examples and Use Cases

This document provides practical examples and use cases for using PaSSER in your research and development workflows.

## Table of Contents

1. [Basic RAG Testing](#basic-rag-testing)
2. [Multi-Agent Evaluation](#multi-agent-evaluation)
3. [Model Comparison](#model-comparison)
4. [Batch Processing](#batch-processing)
5. [Image Generation](#image-generation)
6. [Blockchain Integration](#blockchain-integration)

## Basic RAG Testing

### Example 1: Testing a Scientific Knowledge Base

**Scenario**: You have a collection of scientific papers and want to evaluate how well your RAG system answers questions.

**Steps**:

1. **Prepare your documents**:
   ```javascript
   // In Database from PDF interface
   - Upload: "quantum_computing_basics.pdf"
   - Database name: "quantum_kb"
   - Chunk size: 500
   - Chunk overlap: 50
   ```

2. **Create test questions** (`quantum_test.json`):
   ```json
   [
     {
       "question": "What is quantum superposition?",
       "expected_answer": "Quantum superposition is the principle that quantum systems can exist in multiple states simultaneously until measured."
     },
     {
       "question": "Explain quantum entanglement.",
       "expected_answer": "Quantum entanglement is a phenomenon where particles become correlated in such a way that the quantum state of one particle cannot be described independently of the others."
     }
   ]
   ```

3. **Run the test**:
   - Navigate to RAG Testing
   - Upload `quantum_test.json`
   - Select model: `mistral:7b`
   - Set k=5 for retrieval
   - Click "Start Test"

4. **Analyze results**:
   ```
   Results saved to: maTests/filtered_quantum_test_mistral.csv
   
   Average BLEU: 0.45
   Average ROUGE-L: 0.62
   Average Cosine Similarity: 0.78
   ```

### Example 2: Optimizing Retrieval Parameters

**Objective**: Find the optimal number of retrieved documents (k) for your use case.

```python
# Run multiple tests with different k values
k_values = [3, 5, 10, 15, 20]
results = []

for k in k_values:
    # Configure test with current k value
    # Run test
    # Collect metrics
    results.append({
        'k': k,
        'bleu': bleu_score,
        'rouge': rouge_score,
        'response_time': avg_time
    })

# Analyze trade-offs
# Plot k vs. metrics to find sweet spot
```

## Multi-Agent Evaluation

### Example 3: Complex Question Answering with Agents

**Scenario**: Evaluate a multi-agent system for answering complex queries requiring multiple steps.

**Test Configuration**:

```json
{
  "test_name": "multi_step_reasoning",
  "agents": [
    {
      "name": "retriever",
      "model": "llama2:7b",
      "role": "Document retrieval and summarization"
    },
    {
      "name": "reasoner",
      "model": "mistral:7b",
      "role": "Logical reasoning and synthesis"
    },
    {
      "name": "validator",
      "model": "orca2:7b",
      "role": "Answer validation and refinement"
    }
  ],
  "questions": [
    {
      "id": 1,
      "query": "Compare the energy efficiency of solar panels and wind turbines, considering manufacturing costs and lifetime output.",
      "expected_steps": ["retrieve solar data", "retrieve wind data", "compare metrics", "synthesize answer"]
    }
  ]
}
```

**Results Interpretation**:

```
Agent Pipeline Performance:
├── Retriever: 2.3s, tokens: 450
├── Reasoner: 3.1s, tokens: 320
└── Validator: 1.8s, tokens: 180

Total Time: 7.2s
Total Tokens: 950
Final BLEU: 0.58
Pipeline Efficiency: High
```

## Model Comparison

### Example 4: Benchmarking Multiple Models

**Objective**: Compare performance across different LLM models on the same task.

```python
# models_comparison.py
models = [
    'llama2:7b',
    'llama3:8b',
    'mistral:7b',
    'orca2:7b',
    'granite:8b'
]

test_config = {
    'temperature': 0.7,
    'top_p': 0.9,
    'k': 5,
    'test_file': 'maTests/unique_Passer_MA_100QA.json'
}

# Run tests for each model
for model in models:
    print(f"Testing {model}...")
    run_test(model, test_config)
    
# Generate comparison report
generate_comparison_chart(models)
```

**Sample Results**:

| Model | Avg BLEU | Avg ROUGE-L | Avg Time (s) | Tokens/s |
|-------|----------|-------------|--------------|----------|
| llama2:7b | 0.42 | 0.58 | 2.5 | 45 |
| llama3:8b | 0.48 | 0.63 | 2.8 | 52 |
| mistral:7b | 0.45 | 0.61 | 2.1 | 48 |
| orca2:7b | 0.40 | 0.56 | 2.3 | 43 |
| granite:8b | 0.50 | 0.65 | 3.2 | 38 |

**Analysis**:
- Granite:8b achieves highest quality but is slowest
- Mistral:7b offers best speed/quality trade-off
- Llama3:8b provides balanced performance

## Batch Processing

### Example 5: Large-Scale Evaluation

**Scenario**: Process 1000+ questions efficiently.

```python
# batch_test.py
import json
from concurrent.futures import ThreadPoolExecutor

def process_batch(questions, model, batch_size=10):
    results = []
    
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = []
        
        for i in range(0, len(questions), batch_size):
            batch = questions[i:i+batch_size]
            future = executor.submit(run_test_batch, batch, model)
            futures.append(future)
        
        for future in futures:
            results.extend(future.result())
    
    return results

# Load questions
with open('maTests/large_test_set.json', 'r') as f:
    questions = json.load(f)

# Process in batches
results = process_batch(questions, 'llama2:7b', batch_size=20)

# Save results
save_results('maTests/batch_results.csv', results)
```

## Image Generation

### Example 6: Generating Training Data Images

**Use Case**: Generate synthetic images for data augmentation.

```javascript
// Image generation configuration
const config = {
  model: "FLUX.1-Dev",
  prompts: [
    "A realistic photo of a red apple on a wooden table",
    "A modern office workspace with laptop and coffee",
    "A sunny park with children playing"
  ],
  parameters: {
    guidance_scale: 7.5,
    num_inference_steps: 50,
    width: 512,
    height: 512
  }
};

// Generate images
for (const prompt of config.prompts) {
  await generateImage(prompt, config.parameters);
}
```

### Example 7: Image Editing for Augmentation

```javascript
// Edit existing images
const editConfig = {
  baseImage: "original.png",
  prompt: "Change the background to a beach scene",
  maskArea: "background",
  strength: 0.8
};

await editImage(editConfig);
```

## Blockchain Integration

### Example 8: Storing Test Results on Blockchain

**Purpose**: Create immutable record of model evaluation for research reproducibility.

```javascript
// After running a test
const testResult = {
  test_id: "exp_001",
  model: "llama2:7b",
  dataset: "scientific_qa",
  metrics: {
    bleu: 0.45,
    rouge: 0.62,
    f1: 0.58
  },
  timestamp: Date.now(),
  config: {
    temperature: 0.7,
    k: 5
  }
};

// Store on blockchain
await storeResultOnBlockchain(testResult);

// Returns transaction ID for citation
// TX: a1b2c3d4e5f6...
```

### Example 9: Verifying Published Results

**Scenario**: Verify results from a published paper.

```bash
# Query blockchain for specific test
cleos -u http://your-blockchain-node-url:port get table llmtest llmtest tests \
  --key-type name \
  --index 2 \
  --lower "exp_001" \
  --upper "exp_001"

# Output shows original test configuration and results
# Verify against published data
```

## Advanced Use Cases

### Example 10: Hyperparameter Optimization

```python
# optimize_params.py
from itertools import product
import json

# Define parameter grid
param_grid = {
    'temperature': [0.5, 0.7, 0.9],
    'top_p': [0.8, 0.9, 0.95],
    'k': [3, 5, 10],
    'chunk_size': [300, 500, 700]
}

# Generate all combinations
combinations = list(product(*param_grid.values()))

best_score = 0
best_params = None

for params in combinations:
    config = dict(zip(param_grid.keys(), params))
    
    # Run test with current parameters
    results = run_test(config)
    
    # Calculate aggregate score
    score = (results['bleu'] + results['rouge']) / 2
    
    if score > best_score:
        best_score = score
        best_params = config

print(f"Best Parameters: {best_params}")
print(f"Best Score: {best_score}")
```

### Example 11: Cross-Lingual Evaluation

```python
# Evaluate RAG system across multiple languages
languages = ['en', 'es', 'fr', 'de']
models = {
    'en': 'llama2:7b',
    'es': 'llama2:7b-spanish',
    'fr': 'mistral:7b',
    'de': 'llama2:7b-german'
}

results_by_language = {}

for lang in languages:
    test_file = f'tests/qa_dataset_{lang}.json'
    model = models[lang]
    
    results = run_test(model, test_file)
    results_by_language[lang] = results

# Compare cross-lingual performance
analyze_cross_lingual_results(results_by_language)
```

## Tips and Best Practices

### 1. Metric Selection

Choose metrics based on your use case:
- **Factual QA**: Focus on F1, Exact Match
- **Summarization**: ROUGE scores
- **Translation**: BLEU scores
- **Semantic similarity**: Cosine similarity, BERT score

### 2. Baseline Establishment

Always establish baselines:
```python
# Run baseline with simple retrieval
baseline = run_test(model='llama2:7b', k=3, temperature=0.0)

# Compare against improvements
improved = run_test(model='llama2:7b', k=5, temperature=0.7, 
                   with_reranking=True)

improvement = (improved['f1'] - baseline['f1']) / baseline['f1'] * 100
print(f"Improvement: {improvement:.2f}%")
```

### 3. Statistical Significance

Run multiple iterations:
```python
import numpy as np
from scipy import stats

# Run 5 iterations
results = [run_test(config) for _ in range(5)]
scores = [r['f1'] for r in results]

# Calculate confidence interval
mean = np.mean(scores)
std = np.std(scores)
ci = stats.t.interval(0.95, len(scores)-1, loc=mean, scale=stats.sem(scores))

print(f"F1: {mean:.3f} ± {std:.3f}")
print(f"95% CI: [{ci[0]:.3f}, {ci[1]:.3f}]")
```

## Troubleshooting Common Issues

### Out of Memory Errors

```python
# Reduce batch size
config['batch_size'] = 5  # Instead of 10

# Reduce context window
config['max_context_tokens'] = 2048  # Instead of 4096

# Use smaller model
config['model'] = 'llama2:7b'  # Instead of 'llama2:13b'
```

### Slow Performance

```python
# Enable caching
config['use_cache'] = True

# Reduce retrievals
config['k'] = 3  # Instead of 10

# Use faster embedding model
config['embedding_model'] = 'all-MiniLM-L6-v2'
```

## More Examples

Find more examples in:
- `/examples` directory (coming soon)
- Community contributions
- Research papers using PaSSER

## Contributing Examples

Have a great use case? Share it!
1. Create a new example file
2. Include code, data, and results
3. Submit a PR

Happy experimenting! 🔬
