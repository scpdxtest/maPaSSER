# PaSSER: Platform for Smart Testing and Evaluation of RAG Systems

[![Version](https://img.shields.io/badge/version-0.7.0-blue.svg)](https://github.com/yourusername/passer)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![React](https://img.shields.io/badge/React-18.2.0-61dafb.svg)](https://reactjs.org/)
[![Python](https://img.shields.io/badge/Python-3.8+-3776ab.svg)](https://www.python.org/)

> A comprehensive web-based platform for implementing, testing, and evaluating Retrieval-Augmented Generation (RAG) systems with Large Language Models (LLMs), blockchain integration, and advanced NLP metrics.

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Testing and Evaluation](#testing-and-evaluation)
- [Results and Metrics](#results-and-metrics)
- [Blockchain Integration](#blockchain-integration)
- [Citation](#citation)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## 🎯 Overview

**PaSSER** (Platform for Smart Testing and Evaluation of RAG Systems) is an advanced web application designed for researchers, developers, and data scientists working with Retrieval-Augmented Generation (RAG) models and Large Language Models (LLMs). The platform provides a comprehensive suite of tools for:

- **RAG System Development**: Build and test RAG pipelines with multiple LLM backends
- **Multi-Agent Orchestration**: Coordinate multiple AI agents for complex tasks
- **Performance Evaluation**: Comprehensive NLP metrics and scoring systems
- **Blockchain Integration**: Secure, immutable storage of test results and model performance data
- **Knowledge Base Management**: Vector database integration with ChromaDB and IPFS
- **Text-to-Image Generation**: AI image generation and editing capabilities

## ✨ Key Features

### 🤖 LLM Integration
- **Multiple Model Support**: Compatible with Mistral, Llama2, Llama3, Orca2, Granite, and other open-source LLMs
- **Ollama Backend**: Seamless integration with local and remote Ollama instances
- **OpenAI Integration**: Support for GPT models and embeddings
- **Custom Model Management**: Add and configure custom LLM endpoints

### 📚 RAG Capabilities
- **Document Processing**: Support for PDF, DOCX, TXT, and web content ingestion
- **Vector Database**: ChromaDB integration for efficient semantic search
- **IPFS Storage**: Decentralized storage for knowledge bases
- **Adaptive Retrieval**: Configurable chunk sizes and retrieval strategies

### 🧪 Testing & Evaluation
- **Comprehensive Metrics**: 
  - BLEU, ROUGE, METEOR scores
  - Cosine similarity, Jaccard index
  - Perplexity and F1 scores
  - Semantic similarity metrics
- **Batch Testing**: Run tests on multiple questions and models simultaneously
- **Multi-Agent Pipeline Testing**: Evaluate complex multi-agent workflows
- **Performance Tracking**: Monitor token usage, response times, and throughput

### 🔗 Blockchain Features
- **Antelope.io Integration**: Store test results on blockchain for immutability
- **Wharf Wallet Support**: Secure authentication with Anchor wallet
- **Result Verification**: Cryptographically verify test results and model performance
- **Transparent Audit Trail**: Complete history of experiments and evaluations

### 🎨 Image Generation
- **Text-to-Image**: Generate images from text prompts
- **Image Editing**: AI-powered image manipulation and enhancement
- **FLUX Model Support**: State-of-the-art diffusion models
- **Real-time Progress**: WebSocket-based progress tracking

### 📊 Data Visualization
- **Interactive Charts**: Visualize metrics with Chart.js
- **Comparative Analysis**: Compare performance across models and configurations
- **Export Capabilities**: Export results in CSV and JSON formats
- **Historical Trends**: Track performance over time

## 🏗️ System Architecture

```
PaSSER Platform
├── Frontend (React)
│   ├── RAG Chat Interface
│   ├── Multi-Agent Testing
│   ├── Image Generation
│   ├── Database Management
│   └── Results Visualization
│
├── Backend Services (Python/Flask)
│   ├── RAG Pipeline Server
│   ├── Multi-Agent Orchestration
│   ├── Metrics Calculation Engine
│   └── Image Generation APIs
│
├── Data Layer
│   ├── ChromaDB (Vector Database)
│   ├── MongoDB (Test Results)
│   ├── IPFS (Decentralized Storage)
│   └── Blockchain (Result Verification)
│
└── LLM Infrastructure
    ├── Ollama Instances
    ├── OpenAI API
    └── Custom Model Endpoints
```

## 🔧 Prerequisites

### System Requirements
- **Node.js**: v14.0 or higher
- **Python**: 3.8 or higher
- **MongoDB**: v4.4 or higher
- **Ollama**: Latest version (for local LLM hosting)
- **ChromaDB**: v0.4.0 or higher

### Optional Components
- **IPFS Node**: For decentralized storage
- **Blockchain Node**: Antelope.io compatible node
- **CUDA**: For GPU-accelerated image generation

## 📦 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/passer.git
cd passer
```

### 2. Install Frontend Dependencies

```bash
npm install
```

### 3. Install Python Dependencies

```bash
cd scripts
pip install -r requirements.txt
cd ..
```

### 4. Install Additional Python Packages

```bash
pip install flask flask-cors pymongo chromadb
pip install nltk rouge scipy scikit-learn
pip install torch transformers
```

### 5. Download NLTK Data

```python
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
```

## ⚙️ Configuration

### 1. Configure Endpoints

Edit `src/component/configuration.json` with your server details:

```json
{
  "passer": {
    "Ollama": [
      {
        "name": "OllamaEndPoint1",
        "url": "http://127.0.0.1:11434"
      },
      {
        "name": "OllamaEndPoint2",
        "url": "http://your-ollama-server-ip:port"
      }
    ],
    "Chroma": [
      {
        "name": "ChromaEndPoint1",
        "url": "http://127.0.0.1:8000"
      }
    ],
    "MultiAgent": [
      {
        "name": "Local MultiAgent",
        "url": "http://127.0.0.1:8004"
      }
    ],
    "BCEndPoints": [
      {
        "name": "BCEndPoint1",
        "url": "http://your-blockchain-node-url:port"
      }
    ],
    "BCid": "your-blockchain-transaction-id",
    "IPFSEndPoints": [
      {
        "name": "IPFSEndPoint1",
        "host": "your-ipfs-server-ip",
        "port": "5001"
      }
    ]
  }
}
```

### 2. Configure MongoDB

Update MongoDB connection strings in backend files:

```python
mongo_client = MongoClient('mongodb://your-mongodb-server-ip:port/')
```

### 3. Configure API Keys

For OpenAI integration, set your API key:

```javascript
const OPENAI_API_KEY = "your-openai-api-key-here";
```

Or use environment variables:

```bash
export OPENAI_API_KEY="your-api-key"
```

## 🚀 Usage

### Starting the Frontend

```bash
npm start
```

The application will be available at `http://localhost:3000`

### Starting Backend Services

#### RAG Backend Server
```bash
cd src/component
python backEnd.py
```

#### Multi-Agent Backend Server
```bash
python maBackEnd.py
```

#### Image Generation Server
```bash
python t2img3_server_api.py
```

### Starting ChromaDB

```bash
chroma run --host 127.0.0.1 --port 8000
```

### Starting Ollama

```bash
ollama serve
```

Pull required models:
```bash
ollama pull llama2:7b
ollama pull mistral:7b
ollama pull orca2:7b
```

## 🧪 Testing and Evaluation

### Running RAG Tests

1. Navigate to **RAG Testing** in the application
2. Select your test dataset from `maTests/unique_Passer_MA_100QA.json`
3. Configure test parameters:
   - Model selection
   - Temperature and top_p
   - Context window size
   - Number of retrieved documents (k)
4. Click **Start Test** to begin evaluation

### Running Multi-Agent Tests

1. Navigate to **Multi-Agent Testing**
2. Upload or select your question dataset
3. Configure the multi-agent pipeline
4. Monitor real-time progress and metrics
5. View results in the dashboard

### Batch Testing

For automated batch testing:

```bash
cd scripts
python addAllTestNames.py
```

## 📊 Results and Metrics

### Test Results Location

All test results are stored in the `maTests/` directory:

- **CSV Format**: `filtered_newTest[ModelName]_[Threshold].csv`
- **JSON Format**: `unique_Passer_MA_100QA.json`

### Available Metrics

PaSSER calculates the following metrics for each test:

| Metric | Description | Range |
|--------|-------------|-------|
| **BLEU** | Precision-based n-gram overlap | 0.0 - 1.0 |
| **ROUGE-1** | Unigram recall | 0.0 - 1.0 |
| **ROUGE-2** | Bigram recall | 0.0 - 1.0 |
| **ROUGE-L** | Longest common subsequence | 0.0 - 1.0 |
| **METEOR** | Alignment-based metric | 0.0 - 1.0 |
| **Cosine Similarity** | Vector space similarity | -1.0 - 1.0 |
| **Jaccard Index** | Set overlap coefficient | 0.0 - 1.0 |
| **F1 Score** | Harmonic mean of precision/recall | 0.0 - 1.0 |
| **Perplexity** | Language model uncertainty | Lower is better |
| **Token Count** | Response length metrics | Integer |
| **Response Time** | Generation latency | Seconds |

### Viewing Results

Results can be viewed through:

1. **Web Interface**: Navigate to "Show Test Results"
2. **MongoDB**: Query the `myDB.test_scores` collection
3. **Blockchain**: Verify results via blockchain explorer
4. **Export**: Download as CSV or JSON

## 🔐 Blockchain Integration

### Setup Blockchain Connection

1. Install Anchor Wallet browser extension
2. Configure blockchain endpoint in `configuration.json`
3. Connect wallet through the application

### Storing Results on Blockchain

Results are automatically stored on the blockchain when:
- A test completes successfully
- The blockchain endpoint is configured
- The user has sufficient resources

### Verifying Results

```bash
cleos -u http://your-blockchain-node-url:port get table llmtest llmtest tests
```

## 📖 Citation

If you use PaSSER in your research, please cite:

```bibtex
@article{passer2025,
  title={PaSSER: Platform for Smart Testing and Evaluation of RAG Systems},
  author={Your Name and Co-authors},
  journal={Journal Name},
  year={2025},
  volume={X},
  pages={XX-XX},
  doi={10.xxxx/xxxxx}
}
```

## 🤝 Contributing

We welcome contributions from the research community! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Development Guidelines

- Follow React and Python best practices
- Add tests for new features
- Update documentation as needed
- Ensure backward compatibility

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

This work was supported by:

- **Bulgarian Ministry of Education and Science** under the National Research Program "Smart crop production" approved by the Ministry Council No. 866/26.11.2020
- Contributors and maintainers of open-source LLM projects
- The ChromaDB, Ollama, and LangChain communities
- The Antelope.io blockchain ecosystem

## 📧 Contact

For questions, suggestions, or collaboration opportunities:

- **Project Repository**: [https://github.com/yourusername/passer](https://github.com/yourusername/passer)
- **Email**: your.email@institution.edu
- **Research Group**: [Your Research Group Website]

## 🗺️ Roadmap

### Current Version (0.7.0)
- ✅ RAG system implementation
- ✅ Multi-agent orchestration
- ✅ Blockchain integration
- ✅ Comprehensive metrics suite

### Upcoming Features (0.8.0)
- 🔄 Enhanced multi-modal support
- 🔄 Advanced visualization dashboards
- 🔄 Real-time collaboration features
- 🔄 API documentation and SDK

### Future Plans
- 📅 Fine-tuning interface for custom models
- 📅 Automated hyperparameter optimization
- 📅 Federated learning support
- 📅 Mobile application

---

<p align="center">
  Made with ❤️ for the AI Research Community
</p>

<p align="center">
  <a href="#-table-of-contents">⬆️ Back to Top</a>
</p>
