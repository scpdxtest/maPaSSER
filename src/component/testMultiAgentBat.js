import React, { useState, useEffect, useRef } from 'react';
import { InputSwitch } from 'primereact/inputswitch';
import { Dropdown } from 'primereact/dropdown';
import { InputText } from 'primereact/inputtext';
import { Button } from 'primereact/button';
import { Card } from 'primereact/card';
import { Accordion, AccordionTab } from 'primereact/accordion';
import { Chip } from 'primereact/chip';
import { Tag } from 'primereact/tag';
import { ProgressBar } from 'primereact/progressbar';
import { Message } from 'primereact/message';
import { Divider } from 'primereact/divider';
import { Badge } from 'primereact/badge';
import { Panel } from 'primereact/panel';
import { Tooltip } from 'primereact/tooltip';
import { ChromaClient } from "chromadb";

import 'primereact/resources/themes/lara-light-blue/theme.css';
import 'primereact/resources/primereact.min.css';
import 'primeicons/primeicons.css';
import './testMultiAgentBat.css';

const TestMultiAgentBat = () => {
    // State Management
    const [selectedModel, setSelectedModel] = useState('codegen-350m');
    const [customModelName, setCustomModelName] = useState('');
    const [useCustomModel, setUseCustomModel] = useState(false);
    const [agentStrategy, setAgentStrategy] = useState('collaborative');
    const [numAgents, setNumAgents] = useState(3);
    const [testQuery, setTestQuery] = useState('');
    const [context, setContext] = useState('');
    const [isProcessing, setIsProcessing] = useState(false);
    const [progress, setProgress] = useState(0);
    const [progressMessage, setProgressMessage] = useState('');
    const [results, setResults] = useState(null);
    const [error, setError] = useState(null);
    const [wsConnection, setWsConnection] = useState(null);
    const [availableModels, setAvailableModels] = useState([]);
    const [serverStatus, setServerStatus] = useState('checking');
    const [customModels, setCustomModels] = useState([]);

    // NEW: Context Size State
    const [contextSize, setContextSize] = useState(null); // null = auto, or custom value
    const [contextTokens, setContextTokens] = useState(0);
    const [queryTokens, setQueryTokens] = useState(0);

    // RAG State
    const [useRAG, setUseRAG] = useState(false);
    const [chromaURL, setChromaURL] = useState('http://127.0.0.1:8000');
    const [ollamaURL, setOllamaURL] = useState('http://127.0.0.1:11434');
    const [collectionName, setCollectionName] = useState('');
    const [embeddingModel, setEmbeddingModel] = useState('mistral');
    const [embeddingSource, setEmbeddingSource] = useState('ollama');
    const [topKDocs, setTopKDocs] = useState(7);
    const [availableCollections, setAvailableCollections] = useState([]);
    const [ragStatus, setRagStatus] = useState('idle');
    const [expandedRagSection, setExpandedRagSection] = useState(false);
    const [multiAgentAPI, setMultiAgentAPI] = useState('http://127.0.0.1:8004');

    const wsRef = useRef(null);
    const initRef = useRef(false);

    // NEW: Token estimation helper
    const estimateTokens = (text) => {
        // Simple estimation: ~4 characters per token on average
        // More accurate than word count for most LLMs
        return Math.ceil(text.length / 4);
    };

    // NEW: Auto-calculate recommended context size
    const getRecommendedContextSize = () => {
        const totalInputTokens = queryTokens + contextTokens;
        
        if (totalInputTokens < 2000) return 4096;    // Small: 4K
        if (totalInputTokens < 5000) return 8192;    // Medium: 8K
        if (totalInputTokens < 10000) return 16384;  // Large: 16K
        if (totalInputTokens < 20000) return 32768;  // XL: 32K
        if (totalInputTokens < 40000) return 65536;  // XXL: 64K
        return 128000; // Maximum: 128K
    };

    // NEW: Update token counts when query/context changes
    useEffect(() => {
        setContextTokens(estimateTokens(context));
    }, [context]);

    useEffect(() => {
        setQueryTokens(estimateTokens(testQuery));
    }, [testQuery]);

    // Recommended Models
    const recommendedModels = [
        { 
            key: 'mistral-7b', 
            name: 'Mistral 7B v0.3',
            fullName: 'mistralai/Mistral-7B-Instruct-v0.3',
            description: 'Excellent instruction following and reasoning',
            size: '7B parameters',
            recommended: true,
            category: 'Advanced'
        },
        { 
            key: 'llama-3.2-3b', 
            name: 'Llama 3.2 3B',
            fullName: 'meta-llama/Llama-3.2-3B-Instruct',
            description: 'Strong reasoning and multi-turn conversations',
            size: '3B parameters',
            recommended: true,
            category: 'Advanced'
        },
        { 
            key: 'granite-3.3-8b', 
            name: 'IBM Granite 3.3 8B',
            fullName: 'ibm-granite/granite-3.3-8b-instruct',
            description: 'Enterprise-grade with strong coding capabilities',
            size: '8B parameters',
            recommended: true,
            category: 'Advanced'
        },
        { 
            key: 'codegen-2b', 
            name: 'CodeGen 2B',
            fullName: 'Salesforce/codegen-2B-nl',
            description: 'Fast general-purpose model',
            size: '2B parameters',
            recommended: false,
            category: 'Standard'
        },
        { 
            key: 'codegen-350m', 
            name: 'CodeGen 350M (Fastest)',
            fullName: 'Salesforce/codegen-350M-nl',
            description: 'Lightweight and very fast',
            size: '350M parameters',
            recommended: false,
            category: 'Lightweight'
        },
        { 
            key: 'distilgpt2', 
            name: 'DistilGPT2',
            fullName: 'distilgpt2',
            description: 'Very lightweight baseline model',
            size: '82M parameters',
            recommended: false,
            category: 'Lightweight'
        },
    ];

    // Agent strategies
    const strategies = [
        { 
            key: 'collaborative', 
            name: 'Collaborative',
            description: 'All agents work together in parallel, combining diverse perspectives',
            icon: '🤝',
            bestFor: 'Complex queries requiring multiple viewpoints',
            agentRange: [2, 7]
        },
        { 
            key: 'sequential', 
            name: 'Sequential',
            description: 'Agents work one after another, building on previous responses',
            icon: '➡️',
            bestFor: 'Step-by-step problem solving',
            agentRange: [2, 5]
        },
        { 
            key: 'competitive', 
            name: 'Competitive',
            description: 'Agents compete to provide the best answer',
            icon: '🏆',
            bestFor: 'When accuracy is critical',
            agentRange: [3, 10]
        },
        { 
            key: 'hierarchical', 
            name: 'Hierarchical',
            description: 'Manager agent coordinates specialist agents',
            icon: '🎯',
            bestFor: 'Complex tasks requiring oversight',
            agentRange: [3, 8]
        },
    ];

    // Sample test queries
    const sampleQueries = [
        {
            name: 'Blockchain Governance',
            query: "Explain blockchain evaluation frameworks and their importance",
            context: "Design axes include Authority, Accountability, and Legitimacy. HSTE model covers Technical Performance, Governance Quality, Institutional Fit, and Sustainability Risk. Focus on observable levers that can be translated into evidence-based questions.",
            category: 'Technical',
            suggestRAG: true
        },
        {
            name: 'Consensus Mechanisms',
            query: "Compare Proof of Work, Proof of Stake, and Delegated Proof of Stake consensus mechanisms",
            context: "Consider trade-offs in security, scalability, decentralization, energy efficiency, and economic incentives. Analyze how each mechanism handles Byzantine fault tolerance.",
            category: 'Technical',
            suggestRAG: true
        },
        {
            name: 'Smart Contract Security',
            query: "What are the main security vulnerabilities in smart contracts and how can they be prevented?",
            context: "Focus on reentrancy attacks, integer overflow/underflow, access control issues, and best practices for secure smart contract development.",
            category: 'Security',
            suggestRAG: true
        },
        {
            name: 'DeFi Protocols',
            query: "Explain how automated market makers (AMMs) work in decentralized finance",
            context: "Discuss constant product formula, liquidity pools, impermanent loss, and the role of liquidity providers in DeFi ecosystems.",
            category: 'Finance',
            suggestRAG: false
        },
        {
            name: 'AI Ethics',
            query: "What are the key ethical considerations in deploying large language models?",
            context: "Consider bias, fairness, transparency, privacy, environmental impact, and responsible AI development practices.",
            category: 'Ethics',
            suggestRAG: false
        },
        {
            name: 'Quantum Computing',
            query: "How will quantum computing impact current cryptographic systems?",
            context: "Analyze the threat to RSA and elliptic curve cryptography, and discuss post-quantum cryptography solutions.",
            category: 'Technology',
            suggestRAG: false
        }
    ];

    // Embedding models
    const embeddingModels = [
        { 
            label: '🦙 Mistral (Ollama) - 4096 dims',
            value: 'mistral',
            dimensions: 4096,
            description: 'Ollama Mistral embedding model',
            source: 'ollama'
        },
        { 
            label: '🦙 Llama3 (Ollama) - 4096 dims',
            value: 'llama3',
            dimensions: 4096,
            description: 'Ollama Llama3 embedding model',
            source: 'ollama'
        },
        { 
            label: '🦙 mxbai-embed-large (Ollama) - 1024 dims',
            value: 'mxbai-embed-large',
            dimensions: 1024,
            description: 'MixBread AI large embedding model',
            source: 'ollama'
        },
        { 
            label: '🦙 nomic-embed-text (Ollama) - 768 dims',
            value: 'nomic-embed-text',
            dimensions: 768,
            description: 'Nomic text embedding model',
            source: 'ollama'
        },
        { 
            label: '🤗 all-MiniLM-L6-v2 (384 dims)',
            value: 'sentence-transformers/all-MiniLM-L6-v2',
            dimensions: 384,
            description: 'Fast and efficient HuggingFace model',
            source: 'huggingface'
        },
        { 
            label: '🤗 all-mpnet-base-v2 (768 dims)',
            value: 'sentence-transformers/all-mpnet-base-v2',
            dimensions: 768,
            description: 'Higher quality HuggingFace embeddings',
            source: 'huggingface'
        },
        { 
            label: '🤗 bge-large-en-v1.5 (1024 dims)',
            value: 'BAAI/bge-large-en-v1.5',
            dimensions: 1024,
            description: 'State-of-the-art quality',
            source: 'huggingface'
        }
    ];

    // Auto-detect embedding source
    useEffect(() => {
        const selectedModelInfo = embeddingModels.find(m => m.value === embeddingModel);
        if (selectedModelInfo) {
            setEmbeddingSource(selectedModelInfo.source);
        }
    }, [embeddingModel]);

    // Initialize from localStorage
    useEffect(() => {
        const initializeFromStorage = async () => {
            if (initRef.current) {
                console.log("Initialization already completed, skipping...");
                return;
            }
            
            initRef.current = true;
            console.log("Initializing MultiAgent from localStorage...");

            const storedMultiAgentAPI = localStorage.getItem("selectedMultiAgent");
            if (storedMultiAgentAPI) {
                setMultiAgentAPI(storedMultiAgentAPI);
                console.log("Loaded MultiAgent API from localStorage:", storedMultiAgentAPI);
            } else {
                const defaultAPI = 'http://127.0.0.1:8004';
                setMultiAgentAPI(defaultAPI);
                localStorage.setItem("selectedMultiAgent", defaultAPI);
                console.log("Set default MultiAgent API:", defaultAPI);
            }

            const storedChromaURL = localStorage.getItem("selectedChromaDB");
            if (storedChromaURL) {
                setChromaURL(storedChromaURL);
                console.log("Loaded ChromaDB URL from localStorage:", storedChromaURL);
            }

            const storedOllamaURL = localStorage.getItem("selectedOllama");
            if (storedOllamaURL) {
                setOllamaURL(storedOllamaURL);
                console.log("Loaded Ollama URL from localStorage:", storedOllamaURL);
            }

            const storedModel = localStorage.getItem("selectedLLMModel");
            if (storedModel) {
                const modelMapping = {
                    'mistral': 'mistral-7b',
                    'llama3.1': 'llama-3.1-8b',
                    'llama': 'llama-3.1-8b',
                    'granite': 'granite-3.3-8b',
                };
                const mappedModel = modelMapping[storedModel.toLowerCase()] || 'codegen-350m';
                setSelectedModel(mappedModel);
                console.log("Loaded model from localStorage:", storedModel, "->", mappedModel);
            }

            checkServerStatus();
            console.log("✅ MultiAgent initialization completed");
        };

        initializeFromStorage();

        return () => {
            if (wsRef.current) {
                wsRef.current.close();
            }
        };
    }, []);

    // Load custom models
    useEffect(() => {
        const saved = localStorage.getItem('customModels');
        if (saved) {
            try {
                setCustomModels(JSON.parse(saved));
            } catch (e) {
                console.error('Failed to load custom models:', e);
            }
        }
    }, []);

    // Fetch collections when RAG is enabled
    useEffect(() => {
        if (useRAG && expandedRagSection && chromaURL) {
            fetchCollections();
        }
    }, [useRAG, expandedRagSection, chromaURL]);

    const checkServerStatus = async () => {
        try {
            const apiURL = multiAgentAPI || 'http://127.0.0.1:8004';
            console.log("Checking server status at:", apiURL);
            
            const response = await fetch(`${apiURL}/health`, { timeout: 5000 });
            const data = await response.json();
            setServerStatus('online');
            
            console.log("✅ MultiAgent API server is online:", data);
            
            try {
                const modelsResponse = await fetch(`${apiURL}/models`);
                const models = await modelsResponse.json();
                setAvailableModels(models);
                console.log("Available models:", models);
            } catch (e) {
                console.error('Failed to fetch models:', e);
            }
        } catch (err) {
            console.error('Server check failed:', err);
            setServerStatus('offline');
            setError(`Cannot connect to MultiAgent API server at ${multiAgentAPI}. Please start the server or select a different endpoint in configuration.`);
        }
    };

    const fetchCollections = async () => {
        if (!chromaURL) {
            console.log("No ChromaDB URL configured");
            return;
        }

        try {
            setRagStatus('loading');
            console.log("Fetching collections from ChromaDB:", chromaURL);
            
            const client = new ChromaClient({ path: chromaURL });
            const collections = await client.listCollections();
            
            console.log("Collections received:", collections);
            
            const collectionNames = collections.map(col => {
                return typeof col === 'string' ? col : col.name;
            });
            
            setAvailableCollections(collectionNames);
            setRagStatus('success');
            
            console.log(`✅ Successfully fetched ${collectionNames.length} collections`);
            
        } catch (err) {
            console.error('Failed to fetch ChromaDB collections:', err);
            setRagStatus('error');
            setError(`Failed to fetch ChromaDB collections: ${err.message}. Check if ChromaDB is running at ${chromaURL}`);
        }
    };

    const handleChromaURLChange = (newURL) => {
        setChromaURL(newURL);
        localStorage.setItem("selectedChromaDB", newURL);
        console.log("ChromaDB URL saved to localStorage:", newURL);
        
        if (useRAG && newURL) {
            fetchCollections();
        }
    };

    const handleOllamaURLChange = (newURL) => {
        setOllamaURL(newURL);
        localStorage.setItem("selectedOllama", newURL);
        console.log("Ollama URL saved to localStorage:", newURL);
    };

    const addCustomModel = () => {
        if (!customModelName.trim()) {
            setError('Please enter a valid model name');
            return;
        }

        const exists = [...recommendedModels, ...customModels].some(
            m => m.fullName === customModelName || m.key === customModelName
        );

        if (exists) {
            setError('This model already exists');
            return;
        }

        const newModel = {
            key: customModelName.toLowerCase().replace(/[^a-z0-9-]/g, '-'),
            name: customModelName.split('/').pop() || customModelName,
            fullName: customModelName,
            description: 'Custom HuggingFace model',
            size: 'Unknown',
            recommended: false,
            category: 'Custom',
            isCustom: true
        };

        const updated = [...customModels, newModel];
        setCustomModels(updated);
        localStorage.setItem('customModels', JSON.stringify(updated));
        
        setSelectedModel(newModel.key);
        setCustomModelName('');
        setUseCustomModel(false);
        setError(null);
    };

    const removeCustomModel = (modelKey) => {
        const updated = customModels.filter(m => m.key !== modelKey);
        setCustomModels(updated);
        localStorage.setItem('customModels', JSON.stringify(updated));
        
        if (selectedModel === modelKey) {
            setSelectedModel('codegen-350m');
        }
    };

    const loadSampleQuery = (sample) => {
        setTestQuery(sample.query);
        setContext(sample.context);
        setError(null);
        
        if (sample.suggestRAG && !useRAG) {
            setTimeout(() => {
                if (window.confirm('This query would benefit from RAG (Knowledge Base). Enable RAG?')) {
                    setUseRAG(true);
                    setExpandedRagSection(true);
                }
            }, 500);
        }
    };

    const getSelectedStrategy = () => {
        return strategies.find(s => s.key === agentStrategy);
    };

    const validateAgentCount = () => {
        const strategy = getSelectedStrategy();
        if (strategy) {
            const [min, max] = strategy.agentRange;
            if (numAgents < min || numAgents > max) {
                return `${strategy.name} strategy recommends ${min}-${max} agents`;
            }
        }
        return null;
    };

    const handleTestMultiAgent = async () => {
        if (!testQuery.trim()) {
            setError('Please enter a test query');
            return;
        }

        if (serverStatus === 'offline') {
            setError(`Server is offline. Please start the MultiAgent API server at ${multiAgentAPI} or select a different endpoint.`);
            return;
        }

        if (useRAG && !collectionName) {
            setError('Please select a ChromaDB collection for RAG');
            return;
        }

        if (useRAG && (selectedModel === 'codegen-350m' || selectedModel === 'distilgpt2')) {
            if (!window.confirm('⚠️ Warning: Small models (350M/DistilGPT2) may produce incomplete RAG responses.\n\nRecommended models for RAG:\n• Mistral 7B\n• Llama 3.2 3B\n• Granite 3.3 8B\n\nContinue anyway?')) {
                return;
            }
        }

        const agentWarning = validateAgentCount();
        if (agentWarning && !window.confirm(`Warning: ${agentWarning}\n\nContinue anyway?`)) {
            return;
        }

        setIsProcessing(true);
        setProgress(0);
        setProgressMessage('Initializing MultiAgent system...');
        setError(null);
        setResults(null);

        try {
            let modelKey = selectedModel;
            
            const allModels = [...recommendedModels, ...customModels];
            const modelInfo = allModels.find(m => m.key === selectedModel);
            
            const wsURL = multiAgentAPI.replace('http://', 'ws://').replace('https://', 'wss://');
            const ws = new WebSocket(`${wsURL}/multiagent-query`);
            wsRef.current = ws;
            setWsConnection(ws);

            ws.onopen = () => {
                console.log(`MultiAgent WebSocket connected to ${wsURL}`);
                
                // NEW: Determine context size
                const effectiveContextSize = contextSize || getRecommendedContextSize();
                
                const requestData = {
                    query: testQuery,
                    context: context,
                    model: modelKey,
                    strategy: agentStrategy,
                    num_agents: numAgents,
                    context_size: effectiveContextSize, // NEW: Add context_size
                    // RAG parameters
                    use_rag: useRAG,
                    chroma_url: chromaURL,
                    collection_name: collectionName,
                    embedding_model: embeddingModel,
                    embedding_source: embeddingSource,
                    ollama_url: ollamaURL,
                    top_k_docs: topKDocs,
                    ...(modelInfo?.isCustom && { custom_model_name: modelInfo.fullName })
                };
                
                console.log('Sending request with context_size:', effectiveContextSize);
                ws.send(JSON.stringify(requestData));
            };

            ws.onmessage = (event) => {
                const data = JSON.parse(event.data);
                console.log('MultiAgent message:', data);

                if (data.type === 'progress') {
                    setProgress(data.progress);
                    setProgressMessage(data.message);
                } else if (data.type === 'keepalive') {
                    console.log('Keepalive received');
                } else if (data.type === 'cancelled') {
                    setProgress(0);
                    setProgressMessage('');
                    setError('Processing cancelled by user');
                    setIsProcessing(false);
                    ws.close();
                } else if (data.type === 'agent_response') {
                    console.log(`Agent ${data.agent_id} response:`, data.response);
                } else if (data.type === 'complete') {
                    setProgress(100);
                    setProgressMessage('MultiAgent processing complete!');
                    setResults(data.results);
                    setIsProcessing(false);
                    ws.close();
                } else if (data.type === 'error') {
                    setError(data.message);
                    setIsProcessing(false);
                    ws.close();
                }
            };

            ws.onerror = (error) => {
                console.error('MultiAgent WebSocket error:', error);
                setError(`WebSocket connection error to ${multiAgentAPI}. Please check if the MultiAgent server is running.`);
                setIsProcessing(false);
            };

            ws.onclose = () => {
                console.log('MultiAgent WebSocket closed');
                setWsConnection(null);
                wsRef.current = null;
            };

        } catch (err) {
            console.error('MultiAgent test error:', err);
            setError(`Failed to start MultiAgent test: ${err.message}`);
            setIsProcessing(false);
        }
    };

    const cancelProcessing = () => {
        if (wsRef.current) {
            console.log('🛑 Cancelling processing - closing WebSocket');
            wsRef.current.close();
            wsRef.current = null;
        }
        setIsProcessing(false);
        setProgress(0);
        setProgressMessage('Cancelling...');
        
        setTimeout(() => {
            setError('Processing cancelled. The server has been notified to stop.');
            setProgressMessage('');
        }, 500);
    };

    const getModelColor = (category) => {
        const colors = {
            'Advanced': '#8b5cf6',
            'Standard': '#3b82f6',
            'Lightweight': '#10b981',
            'Specialized': '#f59e0b',
            'Custom': '#ec4899'
        };
        return colors[category] || '#6b7280';
    };

    const allModels = [...recommendedModels, ...customModels];
    const selectedModelInfo = allModels.find(m => m.key === selectedModel);

    const modelOptions = [
        {
            label: '🌟 Recommended Models',
            items: recommendedModels.filter(m => m.recommended).map(m => ({
                label: `⭐ ${m.name} - ${m.size}`,
                value: m.key
            }))
        },
        {
            label: '📦 Standard Models',
            items: recommendedModels.filter(m => !m.recommended).map(m => ({
                label: `${m.name} - ${m.size}`,
                value: m.key
            }))
        },
        ...(customModels.length > 0 ? [{
            label: '🎨 Custom Models',
            items: customModels.map(m => ({
                label: m.name,
                value: m.key
            }))
        }] : [])
    ];

    const collectionOptions = availableCollections.map(col => ({
        label: col,
        value: col
    }));

    return (
        <div className="multiagent-test-container">
            {/* Header */}
            <Card className="multiagent-header-card">
                <div className="header-content">
                    <h1><i className="pi pi-users"></i> MultiAgent System Testing</h1>
                    <p className="multiagent-subtitle">
                        Test collaborative AI agents using HuggingFace LLMs with optional RAG support
                    </p>
                    <div className="server-status-bar">
                        <Chip 
                            label={`Server: ${serverStatus === 'online' ? 'Online' : serverStatus === 'offline' ? 'Offline' : 'Checking...'}`}
                            icon={`pi ${serverStatus === 'online' ? 'pi-check-circle' : serverStatus === 'offline' ? 'pi-times-circle' : 'pi-spin pi-spinner'}`}
                            className={`status-chip ${serverStatus}`}
                        />
                        <Chip 
                            label={`API: ${multiAgentAPI}`}
                            icon="pi pi-server"
                            className="api-chip"
                        />
                        <Button 
                            icon="pi pi-refresh" 
                            label="Refresh" 
                            onClick={checkServerStatus}
                            className="p-button-sm p-button-text"
                        />
                    </div>
                </div>
            </Card>

            {/* Configuration Section */}
            <Card title="⚙️ Configuration" className="config-card">
                {/* Model Selection */}
                <div className="p-field config-group">
                    <label className="config-label">
                        <i className="pi pi-microchip"></i> Language Model
                        {selectedModelInfo && (
                            <Tag 
                                value={selectedModelInfo.category} 
                                style={{ backgroundColor: getModelColor(selectedModelInfo.category), marginLeft: '10px' }}
                            />
                        )}
                    </label>
                    
                    <div className="model-selector-prime">
                        <Dropdown 
                            value={selectedModel}
                            options={modelOptions}
                            optionGroupLabel="label"
                            optionGroupChildren="items"
                            onChange={(e) => setSelectedModel(e.value)}
                            placeholder="Select a model"
                            disabled={isProcessing}
                            className="model-dropdown"
                            style={{ flex: 1 }}
                        />
                        <Button 
                            label={useCustomModel ? 'Cancel' : 'Add Custom'}
                            icon={useCustomModel ? 'pi pi-times' : 'pi pi-plus'}
                            onClick={() => setUseCustomModel(!useCustomModel)}
                            disabled={isProcessing}
                            className="p-button-outlined"
                        />
                    </div>

                    {selectedModelInfo && (
                        <Card className="model-info-card">
                            <div className="info-row">
                                <strong>Full Name:</strong>
                                <code>{selectedModelInfo.fullName}</code>
                            </div>
                            <div className="info-row">
                                <strong>Description:</strong>
                                <span>{selectedModelInfo.description}</span>
                            </div>
                            {selectedModelInfo.isCustom && (
                                <Button 
                                    label="Remove Custom Model"
                                    icon="pi pi-trash"
                                    onClick={() => removeCustomModel(selectedModelInfo.key)}
                                    className="p-button-danger p-button-sm"
                                    style={{ marginTop: '10px' }}
                                />
                            )}
                        </Card>
                    )}

                    {useCustomModel && (
                        <Card className="custom-model-card">
                            <div className="p-inputgroup">
                                <InputText 
                                    value={customModelName}
                                    onChange={(e) => setCustomModelName(e.target.value)}
                                    placeholder="e.g., facebook/opt-1.3b"
                                    style={{ fontFamily: 'monospace' }}
                                />
                                <Button 
                                    label="Add"
                                    icon="pi pi-check"
                                    onClick={addCustomModel}
                                    className="p-button-success"
                                />
                            </div>
                            <Message 
                                severity="info" 
                                text="Enter the full HuggingFace model identifier (e.g., microsoft/phi-2)"
                                style={{ marginTop: '10px' }}
                            />
                        </Card>
                    )}
                </div>

                <Divider />

                {/* Strategy Selection */}
                <div className="config-group">
                    <label className="config-label">
                        <i className="pi pi-sitemap"></i> Agent Strategy
                    </label>
                    <div className="strategy-grid-prime">
                        {strategies.map(strategy => (
                            <Card 
                                key={strategy.key}
                                className={`strategy-card-prime ${agentStrategy === strategy.key ? 'selected' : ''}`}
                                onClick={() => !isProcessing && setAgentStrategy(strategy.key)}
                            >
                                <div className="strategy-icon">{strategy.icon}</div>
                                <div className="strategy-name">{strategy.name}</div>
                                <div className="strategy-desc">{strategy.description}</div>
                                <Tag 
                                    value={`Best for: ${strategy.bestFor}`}
                                    severity="info"
                                    style={{ marginTop: '8px' }}
                                />
                                <Chip 
                                    label={`Agents: ${strategy.agentRange[0]}-${strategy.agentRange[1]}`}
                                    style={{ marginTop: '8px' }}
                                />
                            </Card>
                        ))}
                    </div>
                </div>

                <Divider />

                {/* Number of Agents */}
                <div className="config-group">
                    <label className="config-label">
                        <i className="pi pi-users"></i> Number of Agents: <Badge value={numAgents} severity="info" />
                        {validateAgentCount() && (
                            <Tag severity="warning" value={validateAgentCount()} icon="pi pi-exclamation-triangle" style={{ marginLeft: '10px' }} />
                        )}
                    </label>
                    <div className="agent-slider-prime">
                        <input 
                            type="range"
                            min="2" 
                            max="10" 
                            value={numAgents}
                            onChange={(e) => setNumAgents(parseInt(e.target.value))}
                            disabled={isProcessing}
                            className="slider"
                        />
                        <div className="slider-labels">
                            <span>2</span>
                            <span>5</span>
                            <span>10</span>
                        </div>
                    </div>
                </div>
            </Card>

            {/* RAG Configuration Section */}
            <Card title="🗄️ RAG Configuration (Optional)" className="rag-config-card">
                <div className="rag-toggle-section">
                    <div className="p-field-checkbox">
                        <InputSwitch 
                            checked={useRAG}
                            onChange={(e) => {
                                setUseRAG(e.value);
                                if (e.value) {
                                    setExpandedRagSection(true);
                                }
                            }}
                            disabled={isProcessing}
                        />
                        <label style={{ marginLeft: '10px', fontWeight: '600' }}>
                            Enable RAG (Retrieval-Augmented Generation)
                        </label>
                    </div>
                    {useRAG && (
                        <Tag 
                            value="Knowledge Base Enabled" 
                            severity="success" 
                            icon="pi pi-check"
                            style={{ marginLeft: '10px' }}
                        />
                    )}
                </div>

                {useRAG && (
                    <Accordion activeIndex={expandedRagSection ? 0 : null} onTabChange={(e) => setExpandedRagSection(e.index === 0)}>
                        <AccordionTab header="RAG Settings">
                            <Message 
                                severity="info"
                                style={{ marginBottom: '20px' }}
                            >
                                <div>
                                    <strong>🎯 Hybrid RAG System</strong><br/>
                                    Supports both Ollama (for Mistral/Llama embeddings) and HuggingFace (sentence-transformers).
                                    Select the embedding model that matches your ChromaDB collection's dimensions.
                                </div>
                            </Message>

                            <div className="rag-settings-grid">
                                <div className="p-field">
                                    <label htmlFor="chromaURL">
                                        <i className="pi pi-database"></i> ChromaDB URL
                                        <Tooltip target=".chroma-help" />
                                        <i className="pi pi-question-circle chroma-help" 
                                           data-pr-tooltip="Shared with testRAGBat - changes here will affect both interfaces"
                                           style={{ marginLeft: '5px', cursor: 'pointer', color: '#667eea' }}
                                        />
                                    </label>
                                    <InputText 
                                        id="chromaURL"
                                        value={chromaURL}
                                        onChange={(e) => handleChromaURLChange(e.target.value)}
                                        placeholder="http://127.0.0.1:8000"
                                        disabled={isProcessing}
                                        style={{ width: '100%' }}
                                    />
                                </div>

                                <div className="p-field">
                                    <label htmlFor="ollamaURL">
                                        <i className="pi pi-server"></i> Ollama Server URL
                                        <Tooltip target=".ollama-help" />
                                        <i className="pi pi-question-circle ollama-help" 
                                           data-pr-tooltip="Required for Ollama embedding models (Mistral, Llama3, etc.)"
                                           style={{ marginLeft: '5px', cursor: 'pointer', color: '#667eea' }}
                                        />
                                    </label>
                                    <InputText 
                                        id="ollamaURL"
                                        value={ollamaURL}
                                        onChange={(e) => handleOllamaURLChange(e.target.value)}
                                        placeholder="http://127.0.0.1:11434"
                                        disabled={isProcessing}
                                        style={{ width: '100%' }}
                                    />
                                </div>

                                <div className="p-field">
                                    <label htmlFor="collection">
                                        <i className="pi pi-folder-open"></i> Knowledge Base Collection
                                        <Tooltip target=".collection-help" />
                                        <i className="pi pi-question-circle collection-help" 
                                           data-pr-tooltip="Select the ChromaDB collection containing your knowledge base"
                                           style={{ marginLeft: '5px', cursor: 'pointer' }}
                                        />
                                    </label>
                                    <div className="p-inputgroup">
                                        <Dropdown 
                                            id="collection"
                                            value={collectionName}
                                            options={collectionOptions}
                                            onChange={(e) => setCollectionName(e.value)}
                                            placeholder="Select a collection..."
                                            disabled={isProcessing || ragStatus === 'loading'}
                                            emptyMessage="No collections found"
                                            style={{ flex: 1 }}
                                        />
                                        <Button 
                                            icon={ragStatus === 'loading' ? 'pi pi-spin pi-spinner' : 'pi pi-refresh'}
                                            onClick={fetchCollections}
                                            disabled={isProcessing}
                                            tooltip="Refresh collections"
                                            tooltipOptions={{ position: 'top' }}
                                        />
                                    </div>
                                    {ragStatus === 'success' && availableCollections.length === 0 && (
                                        <Message 
                                            severity="warn" 
                                            text="No collections found. Make sure ChromaDB is running and has collections."
                                            style={{ marginTop: '10px' }}
                                        />
                                    )}
                                    {ragStatus === 'error' && (
                                        <Message 
                                            severity="error" 
                                            text={`Failed to connect to ChromaDB at ${chromaURL}. Check if ChromaDB is running.`}
                                            style={{ marginTop: '10px' }}
                                        />
                                    )}
                                </div>

                                <div className="p-field">
                                    <label htmlFor="embeddingModel">
                                        <i className="pi pi-chart-line"></i> Embedding Model
                                        <Tooltip target=".embedding-help" />
                                        <i className="pi pi-question-circle embedding-help" 
                                           data-pr-tooltip="Choose embedding model matching your collection's dimension. Ollama models need Ollama server running!"
                                           style={{ marginLeft: '5px', cursor: 'pointer', color: '#667eea' }}
                                        />
                                    </label>
                                    <Dropdown 
                                        id="embeddingModel"
                                        value={embeddingModel}
                                        options={embeddingModels.map(model => ({
                                            label: model.label,
                                            value: model.value
                                        }))}
                                        onChange={(e) => setEmbeddingModel(e.value)}
                                        placeholder="Select embedding model"
                                        disabled={isProcessing}
                                        className="w-full"
                                    />
                                    
                                    {embeddingModel && (
                                        <Card className="model-info-card" style={{ marginTop: '10px' }}>
                                            {(() => {
                                                const model = embeddingModels.find(m => m.value === embeddingModel);
                                                return (
                                                    <>
                                                        <div className="info-row">
                                                            <strong>Source:</strong>
                                                            <Tag value={model?.source.toUpperCase()} 
                                                                 severity={model?.source === 'ollama' ? 'warning' : 'info'} />
                                                        </div>
                                                        <div className="info-row">
                                                            <strong>Dimensions:</strong>
                                                            <Tag value={`${model?.dimensions} dims`} severity="info" />
                                                        </div>
                                                        <div className="info-row">
                                                            <strong>Description:</strong>
                                                            <span>{model?.description}</span>
                                                        </div>
                                                        {model?.source === 'ollama' && (
                                                            <Message 
                                                                severity="warn"
                                                                text={`⚠️ Ollama model requires Ollama server at ${ollamaURL}. Make sure it's running with: ollama list`}
                                                                style={{ marginTop: '10px' }}
                                                            />
                                                        )}
                                                    </>
                                                );
                                            })()}
                                        </Card>
                                    )}
                                </div>

                                <div className="p-field">
                                    <label htmlFor="topKDocs">
                                        <i className="pi pi-list"></i> Documents to Retrieve (Top K): <Badge value={topKDocs} />
                                    </label>
                                    <input 
                                        type="range"
                                        id="topKDocs"
                                        min="1" 
                                        max="20" 
                                        value={topKDocs}
                                        onChange={(e) => setTopKDocs(parseInt(e.target.value))}
                                        disabled={isProcessing}
                                        className="slider"
                                        style={{ width: '100%' }}
                                    />
                                    <div className="slider-labels">
                                        <span>1</span>
                                        <span>10</span>
                                        <span>20</span>
                                    </div>
                                </div>
                            </div>
                        </AccordionTab>
                    </Accordion>
                )}
            </Card>

            {/* Query Input Section */}
            <Card title="📝 Query Input" className="input-card">
                <div className="p-field">
                    <label htmlFor="query">
                        Test Query *
                        {queryTokens > 0 && (
                            <Tag 
                                value={`~${queryTokens} tokens`} 
                                severity="info" 
                                style={{ marginLeft: '10px' }}
                            />
                        )}
                    </label>
                    <textarea
                        id="query"
                        value={testQuery}
                        onChange={(e) => setTestQuery(e.target.value)}
                        placeholder="Enter your question or query..."
                        rows={4}
                        disabled={isProcessing}
                        className="p-inputtextarea"
                        style={{ width: '100%', resize: 'vertical' }}
                    />
                </div>

                <div className="p-field">
                    <label htmlFor="context">
                        Context (Optional)
                        {contextTokens > 0 && (
                            <Tag 
                                value={`~${contextTokens} tokens`} 
                                severity={contextTokens > 10000 ? 'warning' : 'info'}
                                style={{ marginLeft: '10px' }}
                            />
                        )}
                    </label>
                    <textarea
                        id="context"
                        value={context}
                        onChange={(e) => setContext(e.target.value)}
                        placeholder="Provide additional context for the agents..."
                        rows={4}
                        disabled={isProcessing}
                        className="p-inputtextarea"
                        style={{ width: '100%', resize: 'vertical' }}
                    />
                </div>

                {/* NEW: Context Size Configuration */}
                <Card className="context-size-card" style={{ marginTop: '20px', backgroundColor: '#f8f9fa' }}>
                    <h4 style={{ marginBottom: '15px' }}>
                        <i className="pi pi-sliders-h"></i> Context Window Size
                        <Tooltip target=".context-help" />
                        <i className="pi pi-question-circle context-help" 
                           data-pr-tooltip="Smaller context windows = faster processing. Auto mode selects optimal size based on your input."
                           style={{ marginLeft: '8px', cursor: 'pointer', color: '#667eea' }}
                        />
                    </h4>

                    <div className="p-field-checkbox" style={{ marginBottom: '15px' }}>
                        <InputSwitch 
                            checked={contextSize === null}
                            onChange={(e) => setContextSize(e.value ? null : getRecommendedContextSize())}
                            disabled={isProcessing}
                        />
                        <label style={{ marginLeft: '10px', fontWeight: '600' }}>
                            Auto-detect optimal size
                            {contextSize === null && (
                                <Tag 
                                    value={`Will use: ${getRecommendedContextSize().toLocaleString()} tokens`}
                                    severity="success"
                                    style={{ marginLeft: '10px' }}
                                />
                            )}
                        </label>
                    </div>

                    {contextSize !== null && (
                        <div>
                            <label htmlFor="contextSize">
                                Custom Context Size: <Badge value={contextSize?.toLocaleString()} severity="info" />
                            </label>
                            <div style={{ display: 'flex', gap: '10px', alignItems: 'center', marginTop: '10px' }}>
                                <Dropdown 
                                    id="contextSize"
                                    value={contextSize}
                                    options={[
                                        { label: '4K tokens (Fast)', value: 4096 },
                                        { label: '8K tokens (Balanced)', value: 8192 },
                                        { label: '16K tokens (Standard)', value: 16384 },
                                        { label: '32K tokens (Large)', value: 32768 },
                                        { label: '64K tokens (XL)', value: 65536 },
                                        { label: '128K tokens (Maximum)', value: 128000 },
                                    ]}
                                    onChange={(e) => setContextSize(e.value)}
                                    placeholder="Select context size"
                                    disabled={isProcessing}
                                    style={{ flex: 1 }}
                                />
                                <Button 
                                    icon="pi pi-times"
                                    onClick={() => setContextSize(null)}
                                    className="p-button-text p-button-danger"
                                    tooltip="Reset to auto"
                                    disabled={isProcessing}
                                />
                            </div>
                        </div>
                    )}

                    {/* Performance estimates */}
                    <Message 
                        severity="info"
                        style={{ marginTop: '15px' }}
                    >
                        <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                            <div>
                                <strong>📊 Input Analysis:</strong>
                            </div>
                            <div style={{ display: 'flex', gap: '15px', flexWrap: 'wrap' }}>
                                <Chip label={`Query: ~${queryTokens} tokens`} />
                                <Chip label={`Context: ~${contextTokens} tokens`} />
                                <Chip 
                                    label={`Total: ~${queryTokens + contextTokens} tokens`} 
                                    style={{ backgroundColor: queryTokens + contextTokens > 10000 ? '#f59e0b' : '#10b981' }}
                                />
                            </div>
                            <div style={{ marginTop: '8px' }}>
                                <strong>⚡ Performance Tip:</strong> Smaller context windows significantly speed up generation. 
                                Use auto mode for optimal balance between speed and quality.
                            </div>
                        </div>
                    </Message>
                </Card>

                {/* Sample Queries */}
                <Panel header="📚 Quick Start - Load Sample Query" toggleable collapsed={true} className="sample-panel">
                    <div className="sample-grid-prime">
                        {sampleQueries.map((sample, idx) => (
                            <Card 
                                key={idx} 
                                className="sample-card-prime"
                                onClick={() => !isProcessing && loadSampleQuery(sample)}
                                style={{ cursor: isProcessing ? 'not-allowed' : 'pointer' }}
                            >
                                <div className="sample-name">{sample.name}</div>
                                <div style={{ display: 'flex', gap: '5px', margin: '8px 0' }}>
                                    <Chip label={sample.category} className="p-chip-sm" />
                                    {sample.suggestRAG && (
                                        <Chip label="RAG Suggested" icon="pi pi-database" className="p-chip-sm" style={{ backgroundColor: '#10b981' }} />
                                    )}
                                </div>
                                <div className="sample-preview">{sample.query.substring(0, 80)}...</div>
                            </Card>
                        ))}
                    </div>
                </Panel>

                {/* Action Buttons */}
                <div className="action-buttons-prime">
                    {!isProcessing ? (
                        <Button 
                            label="Run MultiAgent Test"
                            icon="pi pi-play"
                            onClick={handleTestMultiAgent}
                            disabled={!testQuery.trim() || serverStatus === 'offline'}
                            className="p-button-lg p-button-success run-btn-prime"
                            style={{ flex: 1 }}
                        />
                    ) : (
                        <Button 
                            label="Cancel Processing"
                            icon="pi pi-stop"
                            onClick={cancelProcessing}
                            className="p-button-lg p-button-danger"
                            style={{ flex: 1 }}
                        />
                    )}
                </div>
            </Card>

            {/* Progress Section */}
            {isProcessing && (
                <Card className="progress-card">
                    <h3><i className="pi pi-spin pi-spinner"></i> Processing...</h3>
                    <div className="progress-info-prime">
                        <span>{progressMessage}</span>
                        <span style={{ fontWeight: 'bold', color: '#667eea' }}>{progress}%</span>
                    </div>
                    <ProgressBar value={progress} style={{ height: '20px' }} />
                    <Message 
                        severity="info" 
                        text="💡 First query may take 3-5 minutes for model loading..."
                        style={{ marginTop: '16px' }}
                    />
                </Card>
            )}

            {/* Error Display */}
            {error && (
                <Message 
                    severity="error" 
                    style={{ marginBottom: '24px' }}
                >
                    <div>
                        <strong>⚠️ Error:</strong> {error}
                        {serverStatus === 'offline' && (
                            <div style={{ marginTop: '10px' }}>
                                <p>To start the server on <code>{multiAgentAPI}</code>:</p>
                                <code style={{ display: 'block', padding: '10px', background: '#f3f4f6', borderRadius: '4px', margin: '5px 0' }}>
                                    python multiagent_server_api.py --host 0.0.0.0 --port 8004
                                </code>
                            </div>
                        )}
                    </div>
                </Message>
            )}

            {/* Results Section - (keeping existing results display code) */}
            {results && (
                <Card title="📊 MultiAgent Results" className="results-card">
                    {/* Summary Cards */}
                    <div className="results-summary-prime">
                        <Card className="summary-card-prime">
                            <div className="summary-icon">🎯</div>
                            <div className="summary-label">Strategy</div>
                            <div className="summary-value">{results.strategy}</div>
                        </Card>
                        <Card className="summary-card-prime">
                            <div className="summary-icon">👥</div>
                            <div className="summary-label">Agents</div>
                            <div className="summary-value">{results.num_agents}</div>
                        </Card>
                        <Card className="summary-card-prime">
                            <div className="summary-icon">⏱️</div>
                            <div className="summary-label">Time</div>
                            <div className="summary-value">{results.processing_time}s</div>
                        </Card>
                        <Card className="summary-card-prime">
                            <div className="summary-icon">🎲</div>
                            <div className="summary-label">Consensus</div>
                            <div className="summary-value">{results.consensus_score}%</div>
                        </Card>
                    </div>

                    {/* RAG Information */}
                    {results.rag_info && results.rag_info.enabled && (
                        <Card className="rag-info-card">
                            <h3><i className="pi pi-database"></i> RAG Information</h3>
                            <div className="rag-stats-grid">
                                <div className="rag-stat">
                                    <Chip label="Collection" icon="pi pi-folder-open" />
                                    <span className="rag-stat-value">{results.rag_info.collection}</span>
                                </div>
                                <div className="rag-stat">
                                    <Chip label="Embedding Model" icon="pi pi-chart-line" />
                                    <span className="rag-stat-value">{results.rag_info.embedding_model}</span>
                                </div>
                                <div className="rag-stat">
                                    <Chip label="Embedding Source" icon="pi pi-server" />
                                    <Tag value={results.rag_info.embedding_source.toUpperCase()} 
                                         severity={results.rag_info.embedding_source === 'ollama' ? 'warning' : 'info'} />
                                </div>
                                <div className="rag-stat">
                                    <Chip label="Total Docs Retrieved" icon="pi pi-file" />
                                    <Badge value={results.rag_info.total_documents_retrieved} severity="success" size="large" />
                                </div>
                                <div className="rag-stat">
                                    <Chip label="Avg per Agent" icon="pi pi-chart-bar" />
                                    <Badge value={results.rag_info.avg_docs_per_agent} severity="info" size="large" />
                                </div>
                            </div>
                        </Card>
                    )}

                    {/* Final Answer */}
                    <Panel header="🎯 Final Answer" className="final-answer-panel">
                        <div className="answer-content">
                            {results.final_answer}
                        </div>
                    </Panel>

                    {/* Individual Agent Responses */}
                    {results.agent_responses && results.agent_responses.length > 0 && (
                        <Panel header={`🤖 Individual Agent Responses (${results.agent_responses.length})`} toggleable className="agent-responses-panel">
                            <Accordion multiple activeIndex={[0]}>
                                {results.agent_responses.map((agent, idx) => (
                                    <AccordionTab 
                                        key={idx}
                                        header={
                                            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', width: '100%' }}>
                                                <span>{agent.agent_name}</span>
                                                <div style={{ display: 'flex', gap: '10px' }}>
                                                    <Badge value={`${agent.confidence}%`} severity="success" />
                                                    {agent.rag_enabled && (
                                                        <Chip label={`${agent.rag_docs_used} docs`} icon="pi pi-database" />
                                                    )}
                                                </div>
                                            </div>
                                        }
                                    >
                                        <div className="agent-response">
                                            {agent.response}
                                        </div>
                                        {agent.reasoning && (
                                            <Message 
                                                severity="info"
                                                style={{ marginTop: '12px' }}
                                                content={
                                                    <div>
                                                        <strong>💭 Reasoning:</strong> {agent.reasoning}
                                                    </div>
                                                }
                                            />
                                        )}
                                        {agent.retrieved_sources && agent.retrieved_sources.length > 0 && (
                                            <Panel header={`📚 Retrieved Sources (Top ${agent.retrieved_sources.length})`} toggleable collapsed className="sources-panel">
                                                {agent.retrieved_sources.map((source, sidx) => (
                                                    <Card key={sidx} className="source-card" style={{ marginBottom: '10px' }}>
                                                        <div className="source-content">{source.content}</div>
                                                        {source.metadata && Object.keys(source.metadata).length > 0 && (
                                                            <div className="source-metadata">
                                                                {Object.entries(source.metadata).map(([key, value]) => (
                                                                    <Chip key={key} label={`${key}: ${value}`} className="p-chip-sm" />
                                                                ))}
                                                            </div>
                                                        )}
                                                    </Card>
                                                ))}
                                            </Panel>
                                        )}
                                    </AccordionTab>
                                ))}
                            </Accordion>
                        </Panel>
                    )}

                    {/* Performance Metrics */}
                    {results.metrics && (
                        <Panel header="📈 Performance Metrics" toggleable className="metrics-panel">
                            <div className="metrics-grid-prime">
                                <Card className="metric-card-prime">
                                    <div className="metric-icon">🤝</div>
                                    <div className="metric-label">Agreement Rate</div>
                                    <div className="metric-value">{results.metrics.agreement_rate}%</div>
                                    <ProgressBar value={results.metrics.agreement_rate} showValue={false} />
                                </Card>
                                <Card className="metric-card-prime">
                                    <div className="metric-icon">⭐</div>
                                    <div className="metric-label">Quality Score</div>
                                    <div className="metric-value">{results.metrics.quality_score}/10</div>
                                    <ProgressBar value={results.metrics.quality_score * 10} showValue={false} />
                                </Card>
                                <Card className="metric-card-prime">
                                    <div className="metric-icon">🔗</div>
                                    <div className="metric-label">Coherence</div>
                                    <div className="metric-value">{results.metrics.coherence}%</div>
                                    <ProgressBar value={results.metrics.coherence} showValue={false} />
                                </Card>
                                <Card className="metric-card-prime">
                                    <div className="metric-icon">📊</div>
                                    <div className="metric-label">Coverage</div>
                                    <div className="metric-value">{results.metrics.coverage}%</div>
                                    <ProgressBar value={results.metrics.coverage} showValue={false} />
                                </Card>
                            </div>
                        </Panel>
                    )}

                    {/* Strategy Analysis */}
                    {results.strategy_analysis && (
                        <Panel header="🔍 Strategy Analysis" toggleable className="analysis-panel">
                            <div className="analysis-grid-prime">
                                <Card className="analysis-card-prime strength">
                                    <h4><i className="pi pi-check-circle"></i> Strengths</h4>
                                    <p>{results.strategy_analysis.strengths}</p>
                                </Card>
                                <Card className="analysis-card-prime weakness">
                                    <h4><i className="pi pi-exclamation-triangle"></i> Weaknesses</h4>
                                    <p>{results.strategy_analysis.weaknesses}</p>
                                </Card>
                                <Card className="analysis-card-prime recommendation">
                                    <h4><i className="pi pi-lightbulb"></i> Recommendations</h4>
                                    <p>{results.strategy_analysis.recommendations}</p>
                                </Card>
                            </div>
                        </Panel>
                    )}

                    {/* Export Results */}
                    <div className="export-section-prime">
                        <Button 
                            label="Export Results (JSON)"
                            icon="pi pi-download"
                            onClick={() => {
                                const dataStr = JSON.stringify(results, null, 2);
                                const dataBlob = new Blob([dataStr], { type: 'application/json' });
                                const url = URL.createObjectURL(dataBlob);
                                const link = document.createElement('a');
                                link.href = url;
                                link.download = `multiagent-results-${Date.now()}.json`;
                                link.click();
                                URL.revokeObjectURL(url);
                            }}
                            className="p-button-outlined"
                        />
                        <Button 
                            label="Copy Answer"
                            icon="pi pi-copy"
                            onClick={() => {
                                navigator.clipboard.writeText(results.final_answer);
                                alert('Answer copied to clipboard!');
                            }}
                            className="p-button-outlined"
                        />
                    </div>
                </Card>
            )}
        </div>
    );
};

export default TestMultiAgentBat;