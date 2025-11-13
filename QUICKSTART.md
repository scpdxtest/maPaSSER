# Quick Start Guide

Get PaSSER up and running in less than 10 minutes!

## Prerequisites Check

Before you begin, ensure you have:

```bash
# Check Node.js version (should be 14+)
node --version

# Check Python version (should be 3.8+)
python --version

# Check npm
npm --version

# Check pip
pip --version
```

## 5-Minute Setup

### 1. Clone and Install (2 minutes)

```bash
# Clone the repository
git clone https://github.com/yourusername/passer.git
cd passer

# Install frontend dependencies
npm install

# Install Python dependencies
pip install flask flask-cors pymongo nltk rouge scipy scikit-learn torch transformers chromadb
```

### 2. Start Ollama (1 minute)

```bash
# Install Ollama (if not already installed)
# Visit: https://ollama.ai/download

# Start Ollama
ollama serve

# In another terminal, pull a model
ollama pull llama2:7b
```

### 3. Start ChromaDB (1 minute)

```bash
# Install ChromaDB
pip install chromadb

# Start ChromaDB server
chroma run --host 127.0.0.1 --port 8000
```

### 4. Start PaSSER (1 minute)

```bash
# Start the frontend
npm start

# In another terminal, start the backend
cd src/component
python backEnd.py
```

## Your First RAG Test

### Step 1: Create a Knowledge Base

1. Navigate to **Database from Text** in the application
2. Enter some sample text:
   ```
   PaSSER is a platform for testing RAG systems.
   It supports multiple LLM models and provides comprehensive metrics.
   The platform integrates with blockchain for result verification.
   ```
3. Enter a database name: `my_first_db`
4. Click **Save to ChromaDB**

### Step 2: Chat with Your Knowledge Base

1. Navigate to **Chat from DB**
2. Select your database: `my_first_db`
3. Select model: `llama2:7b`
4. Ask a question: `What is PaSSER?`
5. View the response!

### Step 3: Run a Test

1. Navigate to **RAG Testing**
2. Upload a test file or use the sample questions
3. Configure:
   - Model: `llama2:7b`
   - Temperature: `0.7`
   - k (retrievals): `3`
4. Click **Start Test**
5. View metrics and results!

## Common Issues

### Ollama Connection Error

```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# If not running, start it
ollama serve
```

### ChromaDB Connection Error

```bash
# Check if ChromaDB is running
curl http://localhost:8000/api/v1/heartbeat

# If not running, start it
chroma run --host 127.0.0.1 --port 8000
```

### Port Already in Use

```bash
# For frontend (default: 3000)
PORT=3001 npm start

# For backend, edit the Python file and change the port
# app.run(port=8088, debug=True)  # Change 8088 to another port
```

## Next Steps

- 📖 Read the [full documentation](README.md)
- 🧪 Try [example workflows](docs/EXAMPLES.md)
- 🤝 Check out [contributing guidelines](CONTRIBUTING.md)
- 💬 Join our community discussions

## Quick Reference

### Default Endpoints

| Service | Default URL |
|---------|-------------|
| Frontend | http://localhost:3000 |
| Backend | http://localhost:8088 |
| Ollama | http://localhost:11434 |
| ChromaDB | http://localhost:8000 |

### Useful Commands

```bash
# Update Ollama models
ollama pull llama2:7b
ollama pull mistral:7b

# List installed models
ollama list

# Check ChromaDB collections
# Use the web interface or Python API

# Export test results
# Results are automatically saved in maTests/

# Clear browser cache if UI issues occur
# Chrome: Ctrl+Shift+Delete (Cmd+Shift+Delete on Mac)
```

## Need Help?

- 📝 Check the [FAQ](docs/FAQ.md)
- 🐛 Report issues on [GitHub](https://github.com/yourusername/passer/issues)
- 💬 Ask questions in [Discussions](https://github.com/yourusername/passer/discussions)

Happy Testing! 🚀
