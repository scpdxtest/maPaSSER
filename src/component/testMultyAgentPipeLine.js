import React, { useState, useEffect, useCallback, useMemo, useRef } from 'react';
import { DataTable } from 'primereact/datatable';
import { Column } from 'primereact/column';
import { InputText } from 'primereact/inputtext';
import { Button } from 'primereact/button';
import { Card } from 'primereact/card';
import { Dropdown } from 'primereact/dropdown';
import { InputSwitch } from 'primereact/inputswitch';
import { ProgressBar } from 'primereact/progressbar';
import { Message } from 'primereact/message';
import { Chip } from 'primereact/chip';
import { Tag } from 'primereact/tag';
import { Badge } from 'primereact/badge';
import { Panel } from 'primereact/panel';
import { Accordion, AccordionTab } from 'primereact/accordion';
import { ChromaClient } from "chromadb";
import axios from 'axios';
import './testMultiAgentBatch.css';
import configuration from './configuration.json';

const TestMultiAgentBatch = () => {
    // MultiAgent Configuration
    const [multiAgentAPI, setMultiAgentAPI] = useState('http://127.0.0.1:8004');
    const [selectedModel, setSelectedModel] = useState('mistral-7b');
    const [agentStrategy, setAgentStrategy] = useState('collaborative');
    const [numAgents, setNumAgents] = useState(3);
    const [contextSize, setContextSize] = useState(null);
    const [serverStatus, setServerStatus] = useState('checking');

    // RAG Configuration
    const [useRAG, setUseRAG] = useState(false);
    const [chromaURL, setChromaURL] = useState('http://127.0.0.1:8000');
    const [ollamaURL, setOllamaURL] = useState('http://127.0.0.1:11434');
    const [collectionName, setCollectionName] = useState('');
    const [embeddingModel, setEmbeddingModel] = useState('mistral');
    const [embeddingSource, setEmbeddingSource] = useState('ollama');
    const [topKDocs, setTopKDocs] = useState(7);
    const [availableCollections, setAvailableCollections] = useState([]);

    // Batch Testing
    const [testName, setTestName] = useState('');
    const [QAJSON, setQAJSON] = useState([{}]);
    const [isTesting, setIsTesting] = useState(false);
    const [total, setTotal] = useState(0);
    const [completed, setCompleted] = useState(0);
    const [results, setResults] = useState('');
    const [globalAccuracy, setGlobalAccuracy] = useState(0);
    const [bertRtScores, setBertRtScores] = useState([]);
    const [currentQuestionIndex, setCurrentQuestionIndex] = useState(-1);

    // Performance metrics tracking
    const [performanceMetrics, setPerformanceMetrics] = useState({
        totalRetrievalMs: 0,
        totalGenMs: 0,
        totalConsensusMs: 0,
        totalEndToEndMs: 0,
        totalPromptTokens: 0,
        totalCompletionTokens: 0,
        totalTokens: 0,
        avgRetrievalMs: 0,
        avgGenMs: 0,
        avgConsensusMs: 0,
        avgEndToEndMs: 0,
        avgPromptTokens: 0,
        avgCompletionTokens: 0,
        avgTokens: 0,
        avgAgentsParticipated: 0,
        avgConsensusRounds: 0,
        avgDisagreementRate: 0,
        questionsCompleted: 0
    });

    const wsRef = useRef(null);
    const initRef = useRef(false);
    const endRef = useRef(null);
    const timeoutRef = useRef(null);
    const cancelRef = useRef({ current: false });

    // Model options
    const modelOptions = [
        { label: '⭐ Mistral 7B (Recommended)', value: 'mistral-7b' },
        { label: '⭐ Llama 3.2 3B (Recommended)', value: 'llama-3.2-3b' },
        { label: '⭐ Granite 3.3 8B (Recommended)', value: 'granite-3.3-8b' },
        { label: 'Llama 3.1 8B', value: 'llama-3.1-8b' },
        { label: 'CodeGen 2B', value: 'codegen-2b' },
        { label: 'CodeGen 350M (Fast)', value: 'codegen-350m' },
    ];

    const strategyOptions = [
        { label: '🤝 Collaborative', value: 'collaborative' },
        { label: '➡️ Sequential', value: 'sequential' },
        { label: '🏆 Competitive', value: 'competitive' },
        { label: '🎯 Hierarchical', value: 'hierarchical' },
    ];

    const contextSizeOptions = [
        { label: 'Auto (Recommended)', value: null },
        { label: '4K tokens (Fast)', value: 4096 },
        { label: '8K tokens (Balanced)', value: 8192 },
        { label: '16K tokens (Standard)', value: 16384 },
        { label: '32K tokens (Large)', value: 32768 },
        { label: '64K tokens (XL)', value: 65536 },
        { label: '128K tokens (Maximum)', value: 128000 },
    ];

    const embeddingModels = [
        { label: '🦙 Mistral (Ollama)', value: 'mistral', source: 'ollama' },
        { label: '🦙 Llama3 (Ollama)', value: 'llama3', source: 'ollama' },
        { label: '🦙 mxbai-embed-large', value: 'mxbai-embed-large', source: 'ollama' },
        { label: '🤗 all-MiniLM-L6-v2', value: 'sentence-transformers/all-MiniLM-L6-v2', source: 'huggingface' },
        { label: '🤗 all-mpnet-base-v2', value: 'sentence-transformers/all-mpnet-base-v2', source: 'huggingface' },
    ];

    // Initialize from localStorage
    useEffect(() => {
        if (initRef.current) return;
        initRef.current = true;

        const storedMultiAgentAPI = localStorage.getItem("selectedMultiAgent") || 'http://127.0.0.1:8004';
        setMultiAgentAPI(storedMultiAgentAPI);

        const storedChromaURL = localStorage.getItem("selectedChromaDB") || 'http://127.0.0.1:8000';
        setChromaURL(storedChromaURL);

        const storedOllamaURL = localStorage.getItem("selectedOllama") || 'http://127.0.0.1:11434';
        setOllamaURL(storedOllamaURL);

        checkServerStatus();

        return () => {
            if (wsRef.current) {
                wsRef.current.close();
                wsRef.current = null;
            }
            if (timeoutRef.current) {
                clearTimeout(timeoutRef.current);
                timeoutRef.current = null;
            }
        };
    }, []);

    useEffect(() => {
        const selectedModelInfo = embeddingModels.find(m => m.value === embeddingModel);
        if (selectedModelInfo) {
            setEmbeddingSource(selectedModelInfo.source);
        }
    }, [embeddingModel]);

    useEffect(() => {
        if (useRAG && chromaURL) {
            fetchCollections();
        }
    }, [useRAG, chromaURL]);

    useEffect(() => {
        endRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [results]);

    const checkServerStatus = async () => {
        try {
            const response = await fetch(`${multiAgentAPI}/health`, { timeout: 5000 });
            const data = await response.json();
            setServerStatus('online');
            console.log("✅ MultiAgent API server is online:", data);
        } catch (err) {
            console.error('Server check failed:', err);
            setServerStatus('offline');
        }
    };

    const fetchCollections = async () => {
        if (!chromaURL) return;

        try {
            console.log("Fetching collections from ChromaDB:", chromaURL);
            const client = new ChromaClient({ path: chromaURL });
            const collections = await client.listCollections();
            
            const collectionNames = collections.map(col => {
                return typeof col === 'string' ? col : col.name;
            });
            
            setAvailableCollections(collectionNames);
            console.log(`✅ Successfully fetched ${collectionNames.length} collections`);
        } catch (err) {
            console.error('Failed to fetch ChromaDB collections:', err);
            setAvailableCollections([]);
        }
    };

    const handleFileChange = (files) => {
        if (files.length > 0) {
            const file = files[0];
            const reader = new FileReader();
            reader.onload = (e) => {
                const contents = e.target.result;
                try {
                    const json = JSON.parse(contents);
                    setQAJSON(json);
                    console.log(`Loaded ${json.length} questions from file`);
                } catch (error) {
                    console.error('Error parsing JSON', error);
                    alert('Error parsing JSON file. Please check the format.');
                }
            };
            reader.readAsText(file);
        }
    };

    // Update performance metrics
    const updatePerformanceMetrics = (newMetrics) => {
        setPerformanceMetrics(prev => {
            const questionsCompleted = prev.questionsCompleted + 1;
            
            return {
                totalRetrievalMs: prev.totalRetrievalMs + (newMetrics.retrieval_ms || 0),
                totalGenMs: prev.totalGenMs + (newMetrics.gen_ms || 0),
                totalConsensusMs: prev.totalConsensusMs + (newMetrics.consensus_ms || 0),
                totalEndToEndMs: prev.totalEndToEndMs + (newMetrics.end_to_end_ms || 0),
                totalPromptTokens: prev.totalPromptTokens + (newMetrics.prompt_tokens_total || 0),
                totalCompletionTokens: prev.totalCompletionTokens + (newMetrics.completion_tokens_total || 0),
                totalTokens: prev.totalTokens + (newMetrics.tokens_total || 0),
                avgRetrievalMs: (prev.totalRetrievalMs + (newMetrics.retrieval_ms || 0)) / questionsCompleted,
                avgGenMs: (prev.totalGenMs + (newMetrics.gen_ms || 0)) / questionsCompleted,
                avgConsensusMs: (prev.totalConsensusMs + (newMetrics.consensus_ms || 0)) / questionsCompleted,
                avgEndToEndMs: (prev.totalEndToEndMs + (newMetrics.end_to_end_ms || 0)) / questionsCompleted,
                avgPromptTokens: (prev.totalPromptTokens + (newMetrics.prompt_tokens_total || 0)) / questionsCompleted,
                avgCompletionTokens: (prev.totalCompletionTokens + (newMetrics.completion_tokens_total || 0)) / questionsCompleted,
                avgTokens: (prev.totalTokens + (newMetrics.tokens_total || 0)) / questionsCompleted,
                avgAgentsParticipated: ((prev.avgAgentsParticipated * prev.questionsCompleted) + (newMetrics.agents_participated || 0)) / questionsCompleted,
                avgConsensusRounds: ((prev.avgConsensusRounds * prev.questionsCompleted) + (newMetrics.consensus_rounds || 0)) / questionsCompleted,
                avgDisagreementRate: ((prev.avgDisagreementRate * prev.questionsCompleted) + (newMetrics.disagreement_rate || 0)) / questionsCompleted,
                questionsCompleted
            };
        });
    };

    const processQuestionWithMultiAgent = async (question, context = '') => {
        return new Promise((resolve, reject) => {
            const wsURL = multiAgentAPI.replace('http://', 'ws://').replace('https://', 'wss://');
            const ws = new WebSocket(`${wsURL}/multiagent-query`);
            wsRef.current = ws;
    
            let result = null;
            let lastProgressTime = Date.now();
            let heartbeatInterval = null;
            let keepaliveReceived = false;
    
            ws.onopen = () => {
                console.log(`Connected to MultiAgent WebSocket for question: ${question.substring(0, 50)}...`);
                
                const requestData = {
                    query: question,
                    context: context,
                    model: selectedModel,
                    strategy: agentStrategy,
                    num_agents: numAgents,
                    context_size: contextSize,
                    use_rag: useRAG,
                    chroma_url: chromaURL,
                    collection_name: collectionName,
                    embedding_model: embeddingModel,
                    embedding_source: embeddingSource,
                    ollama_url: ollamaURL,
                    top_k_docs: topKDocs,
                };
    
                ws.send(JSON.stringify(requestData));
    
                // FIX: More aggressive heartbeat - every 20 seconds instead of 30
                heartbeatInterval = setInterval(() => {
                    if (ws.readyState === WebSocket.OPEN) {
                        try {
                            ws.send(JSON.stringify({ type: 'ping' }));
                            console.log('💓 Sent ping to keep connection alive');
                        } catch (e) {
                            console.warn('Failed to send ping:', e);
                        }
                    }
                }, 20000); // Every 20 seconds
    
                // FIX: Much longer timeout - 2 hours without ANY message
                timeoutRef.current = setTimeout(() => {
                    const timeSinceLastProgress = Date.now() - lastProgressTime;
                    console.warn(`⏰ No messages for ${Math.floor(timeSinceLastProgress/1000)}s, timing out`);
                    if (heartbeatInterval) clearInterval(heartbeatInterval);
                    ws.close(1000, 'Client timeout - no progress');
                    reject(new Error('Request timeout - no progress for 2 hours'));
                }, 7200000); // 2 hours maximum
            };
    
            ws.onmessage = (event) => {
                const data = JSON.parse(event.data);
    
                // FIX: Update last progress time on ANY message
                lastProgressTime = Date.now();
                
                // Reset timeout on every message
                if (timeoutRef.current) {
                    clearTimeout(timeoutRef.current);
                    timeoutRef.current = setTimeout(() => {
                        const timeSinceLastProgress = Date.now() - lastProgressTime;
                        if (timeSinceLastProgress > 7200000) { // 2 hours
                            console.warn(`⏰ No messages for ${Math.floor(timeSinceLastProgress/1000)}s`);
                            if (heartbeatInterval) clearInterval(heartbeatInterval);
                            ws.close(1000, 'Client timeout');
                            reject(new Error('Request timeout - no progress for 2 hours'));
                        }
                    }, 7200000); // 2 hours
                }
    
                if (data.type === 'progress') {
                    console.log(`Progress: ${data.progress}% - ${data.message}`);
                } else if (data.type === 'complete') {
                    result = data.results;
                    if (timeoutRef.current) {
                        clearTimeout(timeoutRef.current);
                        timeoutRef.current = null;
                    }
                    if (heartbeatInterval) {
                        clearInterval(heartbeatInterval);
                        heartbeatInterval = null;
                    }
                    ws.close();
                } else if (data.type === 'error') {
                    if (timeoutRef.current) {
                        clearTimeout(timeoutRef.current);
                        timeoutRef.current = null;
                    }
                    if (heartbeatInterval) {
                        clearInterval(heartbeatInterval);
                        heartbeatInterval = null;
                    }
                    reject(new Error(data.message));
                    ws.close();
                } else if (data.type === 'cancelled') {
                    if (timeoutRef.current) {
                        clearTimeout(timeoutRef.current);
                        timeoutRef.current = null;
                    }
                    if (heartbeatInterval) {
                        clearInterval(heartbeatInterval);
                        heartbeatInterval = null;
                    }
                    reject(new Error('Processing cancelled'));
                    ws.close();
                } else if (data.type === 'keepalive' || data.type === 'ping') {
                    // Acknowledge keepalive
                    keepaliveReceived = true;
                    console.log('💚 Received keepalive from server');
                }
            };
    
            ws.onerror = (error) => {
                console.error('WebSocket error:', error);
                if (timeoutRef.current) {
                    clearTimeout(timeoutRef.current);
                    timeoutRef.current = null;
                }
                if (heartbeatInterval) {
                    clearInterval(heartbeatInterval);
                    heartbeatInterval = null;
                }
                reject(new Error('WebSocket connection error'));
            };
    
            ws.onclose = (event) => {
                console.log(`WebSocket closed. Code: ${event.code}, Reason: ${event.reason || 'No reason provided'}`);
                wsRef.current = null;
                
                if (timeoutRef.current) {
                    clearTimeout(timeoutRef.current);
                    timeoutRef.current = null;
                }
                
                if (heartbeatInterval) {
                    clearInterval(heartbeatInterval);
                    heartbeatInterval = null;
                }
                
                if (result) {
                    resolve(result);
                } else if (event.code === 1000) {
                    // Normal closure might mean timeout
                    if (event.reason && event.reason.includes('timeout')) {
                        reject(new Error(`Connection timeout: ${event.reason}`));
                    } else {
                        reject(new Error('Connection closed by server without result'));
                    }
                } else if (event.code === 1005) {
                    // 1005 = no status received
                    if (result) {
                        resolve(result);
                    } else {
                        reject(new Error('Connection closed without result'));
                    }
                } else if (event.code === 1006) {
                    reject(new Error('Connection closed abnormally - server might have crashed'));
                } else if (event.code === 1011) {
                    reject(new Error(`Server error: ${event.reason || 'Internal error'}`));
                } else {
                    reject(new Error(`Connection closed (code: ${event.code}): ${event.reason || 'No reason'}`));
                }
            };
        });
    };

    const cancelBatchTest = () => {
        console.log('🛑 Cancelling batch test...');
        
        cancelRef.current.current = true;
        
        if (wsRef.current) {
            console.log('Closing WebSocket connection...');
            wsRef.current.close();
            wsRef.current = null;
        }
        
        if (timeoutRef.current) {
            clearTimeout(timeoutRef.current);
            timeoutRef.current = null;
        }
        
        setIsTesting(false);
        setCurrentQuestionIndex(-1);
        
        setResults(prevResults => 
            prevResults + 
            `\n\n❌ ========================================\n` +
            `   BATCH TEST CANCELLED BY USER\n` +
            `   Completed: ${completed}/${total} questions\n` +
            `   Time: ${new Date().toLocaleTimeString()}\n` +
            `========================================\n`
        );
        
        console.log('✅ Batch test cancelled');
    };

    const startBatchTest = async () => {
        if (!QAJSON || QAJSON.length === 0 || !QAJSON[0].question) {
            alert('Please load a valid JSON file with questions first.');
            return;
        }

        if (!testName || testName.length > 12 || !/^[a-z1-5]+$/.test(testName)) {
            alert('Test name must be 1-12 characters, lowercase letters and numbers 1-5 only.');
            return;
        }

        if (serverStatus === 'offline') {
            alert(`MultiAgent API server is offline at ${multiAgentAPI}. Please start the server first.`);
            return;
        }

        if (useRAG && !collectionName) {
            alert('Please select a ChromaDB collection for RAG.');
            return;
        }

        cancelRef.current.current = false;
        
        setIsTesting(true);
        setTotal(QAJSON.length);
        setResults('');
        setCompleted(0);
        setBertRtScores([]);
        setGlobalAccuracy(0);
        setCurrentQuestionIndex(-1);
        
        // Reset performance metrics
        setPerformanceMetrics({
            totalRetrievalMs: 0,
            totalGenMs: 0,
            totalConsensusMs: 0,
            totalEndToEndMs: 0,
            totalPromptTokens: 0,
            totalCompletionTokens: 0,
            totalTokens: 0,
            avgRetrievalMs: 0,
            avgGenMs: 0,
            avgConsensusMs: 0,
            avgEndToEndMs: 0,
            avgPromptTokens: 0,
            avgCompletionTokens: 0,
            avgTokens: 0,
            avgAgentsParticipated: 0,
            avgConsensusRounds: 0,
            avgDisagreementRate: 0,
            questionsCompleted: 0
        });

        const MAX_RETRIES = 2;

        for (let i = 0; i < QAJSON.length; i++) {
            if (cancelRef.current.current) {
                console.log('Batch test was cancelled, stopping...');
                break;
            }

            setCurrentQuestionIndex(i);
            console.log(`\n=== Processing Question ${i + 1}/${QAJSON.length} ===`);
            const question = QAJSON[i].question;
            const referenceAnswer = QAJSON[i].answer;

            let attempt = 0;
            let success = false;

            while (attempt <= MAX_RETRIES && !success && !cancelRef.current.current) {
                try {
                    const retryText = attempt > 0 ? ` (Retry ${attempt}/${MAX_RETRIES})` : '';
                    
                    setResults(prevResults => 
                        prevResults + 
                        `\n${i + 1}/${QAJSON.length} -> Question: ${question}\n` +
                        `Reference: ${referenceAnswer}\n` +
                        `Status: ⏳ Processing with MultiAgent system${retryText}...\n`
                    );

                    const multiAgentResult = await processQuestionWithMultiAgent(question, '');

                    if (cancelRef.current.current) {
                        console.log('Cancelled during processing');
                        break;
                    }

                    const candidateAnswer = multiAgentResult.final_answer;
                    
                    console.log(`✅ MultiAgent response received (${multiAgentResult.processing_time}s)`);
                    console.log(`Strategy: ${multiAgentResult.strategy}, Consensus: ${multiAgentResult.consensus_score}%`);

                    // Update performance metrics
                    if (multiAgentResult.performance_metrics) {
                        updatePerformanceMetrics(multiAgentResult.performance_metrics);
                        console.log(`📊 Performance Metrics:`, multiAgentResult.performance_metrics);
                    }

                    setResults(prevResults => 
                        prevResults.replace(
                            `${i + 1}/${QAJSON.length} -> Question: ${question}\nReference: ${referenceAnswer}\nStatus: ⏳ Processing with MultiAgent system${retryText}...\n`,
                            `${i + 1}/${QAJSON.length} -> Question: ${question}\n` +
                            `Reference: ${referenceAnswer}\n` +
                            `Status: 📊 Calculating metrics...\n`
                        )
                    );

                    // NEW: Build additional metrics array from performance_metrics
                    const additionalMetrics = [];
                    if (multiAgentResult.performance_metrics) {
                        const pm = multiAgentResult.performance_metrics;
                        additionalMetrics.push(
                            pm.retrieval_ms || 0,
                            pm.gen_ms || 0,
                            pm.consensus_ms || 0,
                            pm.end_to_end_ms || 0,
                            pm.prompt_tokens_total || 0,
                            pm.completion_tokens_total || 0,
                            pm.tokens_total || 0,
                            pm.messages_total || 0,
                            pm.turns_total || 0,
                            pm.agents_participated || 0,
                            pm.consensus_rounds || 0,
                            pm.disagreement_rate !== null ? pm.disagreement_rate : 0,
                            multiAgentResult.consensus_score || 0,
                            multiAgentResult.processing_time || 0,
                            numAgents,
                            // Add strategy as numeric value
                            agentStrategy === 'collaborative' ? 1 : 
                            agentStrategy === 'sequential' ? 2 : 
                            agentStrategy === 'competitive' ? 3 : 
                            agentStrategy === 'hierarchical' ? 4 : 0,
                            useRAG ? 1 : 0,
                            multiAgentResult.rag_info ? multiAgentResult.rag_info.total_documents_retrieved : 0
                        );
                        
                        console.log(`📋 Prepared ${additionalMetrics.length} additional metrics for backend`);
                    }

                    // Send to backend with additional metrics
                    const metrics = {
                        reference: referenceAnswer,
                        candidate: candidateAnswer,
                        userID: localStorage.getItem("wharf_user_name") || 'testuser',
                        testID: testName,
                        // description: `question: ${question.replace(/[^\x00-\x7F]/g, "")} | answer: ${candidateAnswer.substring(0, 200).replace(/[^\x00-\x7F]/g, "")}`,
                        description: `question: ${question.replace(/[^\x00-\x7F]/g, "")} | answer: ${candidateAnswer.replace(/[^\x00-\x7F]/g, "")}`,
                        additional_metrics: additionalMetrics  // NEW: Include performance metrics
                    };

                    try {
                        const response = await axios.post(configuration.passer.PythonMAScore, metrics, {
                            timeout: 30000
                        });
                        
                        if (response.data && response.data.metrics) {
                            const calculatedMetrics = response.data.metrics;
                            const bertRtAvgScore = calculatedMetrics.bert_rt_score.avg_score;
                            const checksum = response.data.checksum;
                            
                            const updatedBertRtScores = [...bertRtScores, bertRtAvgScore];
                            setBertRtScores(updatedBertRtScores);
                            
                            const meanBertRtScore = updatedBertRtScores.reduce((sum, score) => sum + score, 0) / updatedBertRtScores.length;
                            const calculatedGlobalAccuracy = (meanBertRtScore / 5.0) * 100;
                            setGlobalAccuracy(calculatedGlobalAccuracy);
                            
                            const matchPercentage = (bertRtAvgScore / 5.0) * 100;
                            const isMatch = matchPercentage >= 70;
                            
                            console.log(`📊 BERT RT Score: ${bertRtAvgScore.toFixed(2)}`);
                            console.log(`${isMatch ? '✅' : '❌'} Match: ${matchPercentage.toFixed(1)}%`);
                            console.log(`🎯 Global Accuracy: ${calculatedGlobalAccuracy.toFixed(1)}%`);
                            console.log(`🔐 Checksum: ${checksum}`);
                            
                            // Build performance info string
                            let perfInfo = '';
                            if (multiAgentResult.performance_metrics) {
                                const pm = multiAgentResult.performance_metrics;
                                perfInfo = `Perf: Gen:${(pm.gen_ms/1000).toFixed(1)}s | ` +
                                          `Tokens:${pm.tokens_total} (${pm.prompt_tokens_total}+${pm.completion_tokens_total}) | ` +
                                          `Agents:${pm.agents_participated} | ` +
                                          `Consensus:${pm.consensus_rounds} rounds`;
                                if (pm.disagreement_rate !== null) {
                                    perfInfo += ` | Disagreement:${(pm.disagreement_rate * 100).toFixed(1)}%`;
                                }
                                perfInfo += `\nChecksum: ${checksum} | Metrics: ${additionalMetrics.length} additional\n`;
                            }
                            
                            setResults(prevResults => 
                                prevResults.replace(
                                    `${i + 1}/${QAJSON.length} -> Question: ${question}\nReference: ${referenceAnswer}\nStatus: 📊 Calculating metrics...\n`,
                                    `${i + 1}/${QAJSON.length} -> Question: ${question}\n` +
                                    `Reference: ${referenceAnswer}\n` +
                                    // `MultiAgent Answer: ${candidateAnswer.substring(0, 300)}${candidateAnswer.length > 300 ? '...' : ''}\n` +
                                    `MultiAgent Answer: ${candidateAnswer}}\n` +
                                    `Strategy: ${multiAgentResult.strategy} | Agents: ${multiAgentResult.num_agents} | Time: ${multiAgentResult.processing_time}s\n` +
                                    `Consensus: ${multiAgentResult.consensus_score}% | BERT RT: ${bertRtAvgScore.toFixed(2)} | Match: ${isMatch ? '✅' : '❌'} (${matchPercentage.toFixed(1)}%)\n` +
                                    perfInfo +
                                    `Status: ✅ Complete\n\n`
                                )
                            );

                            success = true;
                        } else {
                            throw new Error('No metrics returned from scoring API');
                        }
                    } catch (error) {
                        console.error('❌ Error calculating metrics:', error);
                        setResults(prevResults => 
                            prevResults.replace(
                                `${i + 1}/${QAJSON.length} -> Question: ${question}\nReference: ${referenceAnswer}\nStatus: 📊 Calculating metrics...\n`,
                                `${i + 1}/${QAJSON.length} -> Question: ${question}\n` +
                                `Reference: ${referenceAnswer}\n` +
                                `MultiAgent Answer: ${candidateAnswer.substring(0, 200)}...\n` +
                                `Strategy: ${multiAgentResult.strategy} | Time: ${multiAgentResult.processing_time}s\n` +
                                `Status: ⚠️ Metrics calculation failed: ${error.message}\n\n`
                            )
                        );
                        
                        success = true;
                    }

                } catch (error) {
                    attempt++;
                    console.error(`❌ Error processing question ${i + 1} (attempt ${attempt}):`, error);
                    
                    if (attempt > MAX_RETRIES) {
                        setResults(prevResults => 
                            prevResults.replace(
                                new RegExp(`${i + 1}/${QAJSON.length} -> Question: ${question.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}.*?Status: ⏳.*?\\n`, 's'),
                                ''
                            ) +
                            `${i + 1}/${QAJSON.length} -> Question: ${question}\n` +
                            `Reference: ${referenceAnswer}\n` +
                            `Status: ❌ FAILED after ${MAX_RETRIES} retries - ${error.message}\n\n`
                        );
                    } else if (!cancelRef.current.current) {
                        console.log(`🔄 Retrying in 3 seconds...`);
                        await new Promise(resolve => setTimeout(resolve, 3000));
                    }
                }
            }

            if (cancelRef.current.current) {
                console.log('Batch test cancelled, breaking loop');
                break;
            }

            setCompleted(i + 1);
            
            if (i < QAJSON.length - 1 && !cancelRef.current.current) {
                await new Promise(resolve => setTimeout(resolve, 1000));
            }
        }

        setIsTesting(false);
        setCurrentQuestionIndex(-1);
        
        const finalCompleted = completed + (cancelRef.current.current ? 0 : 1);
        
        if (finalCompleted === total && !cancelRef.current.current) {
            console.log('\n=== Batch Test Complete ===');
            console.log(`📊 Final Global Accuracy: ${globalAccuracy.toFixed(1)}%`);
            console.log(`✅ Completed: ${finalCompleted}/${total}`);
            
            // Build final performance summary
            let perfSummary = '';
            if (performanceMetrics.questionsCompleted > 0) {
                perfSummary = `   Performance Summary:\n` +
                             `   - Avg Generation Time: ${(performanceMetrics.avgGenMs/1000).toFixed(1)}s\n` +
                             `   - Avg Consensus Time: ${(performanceMetrics.avgConsensusMs/1000).toFixed(1)}s\n` +
                             `   - Avg Total Tokens: ${performanceMetrics.avgTokens.toFixed(0)} (${performanceMetrics.avgPromptTokens.toFixed(0)} + ${performanceMetrics.avgCompletionTokens.toFixed(0)})\n` +
                             `   - Avg Agents Participated: ${performanceMetrics.avgAgentsParticipated.toFixed(1)}\n` +
                             `   - Avg Consensus Rounds: ${performanceMetrics.avgConsensusRounds.toFixed(1)}\n` +
                             `   - Avg Disagreement Rate: ${(performanceMetrics.avgDisagreementRate * 100).toFixed(1)}%\n`;
            }
            
            setResults(prevResults => 
                prevResults + 
                `\n\n✅ ========================================\n` +
                `   BATCH TEST COMPLETED\n` +
                `   Total Questions: ${total}\n` +
                `   Completed: ${finalCompleted}\n` +
                `   Global Accuracy: ${globalAccuracy.toFixed(1)}%\n` +
                perfSummary +
                `   Time: ${new Date().toLocaleTimeString()}\n` +
                `========================================\n`
            );
        }
    };

    const collectionOptions = availableCollections.map(col => ({
        label: col,
        value: col
    }));

    return (
        <div style={{ width: '100%', display: 'flex', justifyContent: 'space-between', padding: '20px', gap: '20px' }}>
            {/* Configuration Panel - Same as before */}
            <div style={{ width: '35%' }}>
                <Card title="⚙️ MultiAgent Batch Test Configuration" className="config-card">
                    {/* Server Status */}
                    <div style={{ marginBottom: '20px', padding: '12px', backgroundColor: serverStatus === 'online' ? '#e8f5e8' : '#fff5f5', border: `2px solid ${serverStatus === 'online' ? '#4caf50' : '#f44336'}`, borderRadius: '8px' }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                            <Chip 
                                label={serverStatus === 'online' ? 'Server Online' : 'Server Offline'}
                                icon={`pi ${serverStatus === 'online' ? 'pi-check-circle' : 'pi-times-circle'}`}
                                className={serverStatus === 'online' ? 'p-chip-success' : 'p-chip-danger'}
                            />
                            <Button 
                                icon="pi pi-refresh" 
                                className="p-button-sm p-button-text"
                                onClick={checkServerStatus}
                            />
                        </div>
                        <div style={{ fontSize: '12px', marginTop: '5px', color: '#666' }}>
                            API: {multiAgentAPI}
                        </div>
                    </div>

                    {/* Model Selection */}
                    <div className="p-field" style={{ marginBottom: '15px' }}>
                        <label><strong>Language Model</strong></label>
                        <Dropdown 
                            value={selectedModel}
                            options={modelOptions}
                            onChange={(e) => setSelectedModel(e.value)}
                            disabled={isTesting}
                            style={{ width: '100%', marginTop: '5px' }}
                        />
                    </div>

                    {/* Strategy Selection */}
                    <div className="p-field" style={{ marginBottom: '15px' }}>
                        <label><strong>Agent Strategy</strong></label>
                        <Dropdown 
                            value={agentStrategy}
                            options={strategyOptions}
                            onChange={(e) => setAgentStrategy(e.value)}
                            disabled={isTesting}
                            style={{ width: '100%', marginTop: '5px' }}
                        />
                    </div>

                    {/* Number of Agents */}
                    <div className="p-field" style={{ marginBottom: '15px' }}>
                        <label><strong>Number of Agents: <Badge value={numAgents} severity="info" /></strong></label>
                        <input 
                            type="range"
                            min="2" 
                            max="10" 
                            value={numAgents}
                            onChange={(e) => setNumAgents(parseInt(e.target.value))}
                            disabled={isTesting}
                            className="slider"
                            style={{ width: '100%', marginTop: '5px' }}
                        />
                        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '12px', color: '#666' }}>
                            <span>2</span>
                            <span>5</span>
                            <span>10</span>
                        </div>
                    </div>

                    {/* Context Size */}
                    <div className="p-field" style={{ marginBottom: '15px' }}>
                        <label><strong>Context Window Size</strong></label>
                        <Dropdown 
                            value={contextSize}
                            options={contextSizeOptions}
                            onChange={(e) => setContextSize(e.value)}
                            disabled={isTesting}
                            style={{ width: '100%', marginTop: '5px' }}
                        />
                    </div>

                    {/* RAG Configuration */}
                    <Accordion activeIndex={useRAG ? 0 : null}>
                        <AccordionTab header="🗄️ RAG Configuration (Optional)">
                            <div className="p-field-checkbox" style={{ marginBottom: '15px' }}>
                                <InputSwitch 
                                    checked={useRAG}
                                    onChange={(e) => setUseRAG(e.value)}
                                    disabled={isTesting}
                                />
                                <label style={{ marginLeft: '10px' }}>Enable RAG</label>
                            </div>

                            {useRAG && (
                                <>
                                    <div className="p-field" style={{ marginBottom: '15px' }}>
                                        <label><strong>ChromaDB URL</strong></label>
                                        <InputText 
                                            value={chromaURL}
                                            onChange={(e) => {
                                                setChromaURL(e.target.value);
                                                localStorage.setItem("selectedChromaDB", e.target.value);
                                            }}
                                            disabled={isTesting}
                                            style={{ width: '100%', marginTop: '5px' }}
                                        />
                                    </div>

                                    <div className="p-field" style={{ marginBottom: '15px' }}>
                                        <label><strong>Ollama URL</strong></label>
                                        <InputText 
                                            value={ollamaURL}
                                            onChange={(e) => {
                                                setOllamaURL(e.target.value);
                                                localStorage.setItem("selectedOllama", e.target.value);
                                            }}
                                            disabled={isTesting}
                                            style={{ width: '100%', marginTop: '5px' }}
                                        />
                                    </div>

                                    <div className="p-field" style={{ marginBottom: '15px' }}>
                                        <label><strong>Collection</strong></label>
                                        <div className="p-inputgroup">
                                            <Dropdown 
                                                value={collectionName}
                                                options={collectionOptions}
                                                onChange={(e) => setCollectionName(e.value)}
                                                disabled={isTesting}
                                                placeholder="Select collection"
                                                style={{ flex: 1 }}
                                            />
                                            <Button 
                                                icon="pi pi-refresh"
                                                onClick={fetchCollections}
                                                disabled={isTesting}
                                            />
                                        </div>
                                    </div>

                                    <div className="p-field" style={{ marginBottom: '15px' }}>
                                        <label><strong>Embedding Model</strong></label>
                                        <Dropdown 
                                            value={embeddingModel}
                                            options={embeddingModels}
                                            onChange={(e) => setEmbeddingModel(e.value)}
                                            disabled={isTesting}
                                            style={{ width: '100%', marginTop: '5px' }}
                                        />
                                    </div>

                                    <div className="p-field" style={{ marginBottom: '15px' }}>
                                        <label><strong>Top K Documents: <Badge value={topKDocs} /></strong></label>
                                        <input 
                                            type="range"
                                            min="1" 
                                            max="20" 
                                            value={topKDocs}
                                            onChange={(e) => setTopKDocs(parseInt(e.target.value))}
                                            disabled={isTesting}
                                            className="slider"
                                            style={{ width: '100%', marginTop: '5px' }}
                                        />
                                    </div>
                                </>
                            )}
                        </AccordionTab>
                    </Accordion>

                    {/* Global Accuracy Display */}
                    {bertRtScores.length > 0 && (
                        <div style={{ 
                            marginTop: '20px',
                            padding: '15px', 
                            backgroundColor: globalAccuracy >= 70 ? '#e8f5e8' : '#fff5f5', 
                            border: `2px solid ${globalAccuracy >= 70 ? '#4caf50' : '#ff9800'}`, 
                            borderRadius: '8px'
                        }}>
                            <strong>🎯 Global Accuracy</strong>
                            <div style={{ fontSize: '28px', fontWeight: 'bold', marginTop: '8px', color: globalAccuracy >= 70 ? '#4caf50' : '#ff9800' }}>
                                {globalAccuracy.toFixed(1)}%
                            </div>
                            <div style={{ fontSize: '12px', color: '#666', marginTop: '5px' }}>
                                Based on {bertRtScores.length} completed tests
                            </div>
                        </div>
                    )}

                    {/* Performance Metrics Display */}
                    {performanceMetrics.questionsCompleted > 0 && (
                        <div style={{ 
                            marginTop: '20px',
                            padding: '15px', 
                            backgroundColor: '#f5f9ff', 
                            border: '2px solid #2196f3', 
                            borderRadius: '8px'
                        }}>
                            <strong>📊 Performance Metrics</strong>
                            <div style={{ fontSize: '11px', marginTop: '10px', color: '#333', lineHeight: '1.6' }}>
                                <div><strong>Generation:</strong> {(performanceMetrics.avgGenMs/1000).toFixed(1)}s avg</div>
                                <div><strong>Consensus:</strong> {(performanceMetrics.avgConsensusMs/1000).toFixed(1)}s avg</div>
                                <div><strong>Tokens:</strong> {performanceMetrics.avgTokens.toFixed(0)} avg ({performanceMetrics.avgPromptTokens.toFixed(0)} + {performanceMetrics.avgCompletionTokens.toFixed(0)})</div>
                                <div><strong>Agents:</strong> {performanceMetrics.avgAgentsParticipated.toFixed(1)} participated</div>
                                <div><strong>Consensus:</strong> {performanceMetrics.avgConsensusRounds.toFixed(1)} rounds</div>
                                <div><strong>Disagreement:</strong> {(performanceMetrics.avgDisagreementRate * 100).toFixed(1)}%</div>
                            </div>
                        </div>
                    )}

                    {/* Timeout Info */}
                    {isTesting && (
                        <Message 
                            severity="info"
                            style={{ marginTop: '20px' }}
                            text="⏱️ Timeout: 2 hours per question. Each message resets the timer. Heartbeat sent every 20s."
                        />
                    )}
                </Card>
            </div>

            {/* Testing Panel - Same as before */}
            <div style={{ width: '60%' }}>
                <Card title="📋 Batch Testing" className="testing-card">
                    {/* Test Configuration */}
                    <div style={{ display: 'flex', gap: '15px', marginBottom: '20px' }}>
                        <div style={{ flex: 1 }}>
                            <label><strong>Test Name (1-12 chars, lowercase a-z, 1-5)</strong></label>
                            <InputText 
                                value={testName}
                                onChange={(e) => {
                                    const value = e.target.value;
                                    if (value === '' || (value.length <= 12 && /^[a-z1-5]+$/.test(value))) {
                                        setTestName(value);
                                    }
                                }}
                                disabled={isTesting}
                                placeholder="multiagent1"
                                style={{ width: '100%', marginTop: '5px' }}
                            />
                        </div>
                        <div style={{ flex: 1 }}>
                            <label><strong>Questions JSON File</strong></label>
                            <input 
                                type="file"
                                accept=".json"
                                onChange={(e) => handleFileChange(e.target.files)}
                                disabled={isTesting}
                                style={{ width: '100%', marginTop: '5px' }}
                            />
                        </div>
                    </div>

                    {/* Status Message */}
                    <Message 
                        severity={
                            !testName || !QAJSON[0]?.question ? 'warn' :
                            serverStatus === 'offline' ? 'error' :
                            useRAG && !collectionName ? 'warn' : 'success'
                        }
                        text={
                            !testName ? '⚠️ Please enter a test name' :
                            !QAJSON[0]?.question ? '⚠️ Please load a JSON file with questions' :
                            serverStatus === 'offline' ? '❌ MultiAgent server is offline' :
                            useRAG && !collectionName ? '⚠️ RAG enabled but no collection selected' :
                            `✅ Ready to test ${QAJSON.length} questions with ${numAgents} agents (${agentStrategy} strategy)`
                        }
                        style={{ marginBottom: '20px' }}
                    />

                    {/* Action Buttons */}
                    <div style={{ display: 'flex', gap: '10px', marginBottom: '20px' }}>
                        {!isTesting ? (
                            <Button 
                                label="🚀 Start Batch Test"
                                onClick={startBatchTest}
                                disabled={
                                    !testName || 
                                    !QAJSON[0]?.question || 
                                    serverStatus === 'offline' ||
                                    (useRAG && !collectionName)
                                }
                                className="p-button-success"
                                style={{ flex: 1, padding: '12px', fontSize: '16px', fontWeight: 'bold' }}
                            />
                        ) : (
                            <>
                                <Button 
                                    label={`🔄 Testing... ${completed}/${total}`}
                                    disabled
                                    className="p-button-info"
                                    style={{ flex: 1, padding: '12px', fontSize: '16px', fontWeight: 'bold' }}
                                />
                                <Button 
                                    label="🛑 Cancel"
                                    icon="pi pi-stop"
                                    onClick={cancelBatchTest}
                                    className="p-button-danger"
                                    style={{ padding: '12px', fontSize: '16px', fontWeight: 'bold' }}
                                />
                            </>
                        )}
                    </div>

                    {/* Progress Bar */}
                    {isTesting && (
                        <div style={{ marginTop: '20px' }}>
                            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
                                <span><strong>Progress: {completed}/{total}</strong></span>
                                <span><strong>{total > 0 ? ((completed / total) * 100).toFixed(1) : 0}%</strong></span>
                            </div>
                            <ProgressBar 
                                value={total > 0 ? ((completed / total) * 100) : 0}
                                style={{ height: '20px' }}
                            />
                            {currentQuestionIndex >= 0 && (
                                <div style={{ marginTop: '10px', fontSize: '13px', color: '#666' }}>
                                    <i className="pi pi-spin pi-spinner" style={{ marginRight: '8px' }}></i>
                                    Currently processing question {currentQuestionIndex + 1}...
                                </div>
                            )}
                        </div>
                    )}

                    {/* Results Panel */}
                    <div style={{ 
                        marginTop: '20px', 
                        backgroundColor: '#000', 
                        color: '#fff', 
                        padding: '15px', 
                        borderRadius: '8px',
                        overflow: 'auto'
                    }}>
                        <div style={{ 
                            display: 'flex', 
                            justifyContent: 'space-between', 
                            alignItems: 'center',
                            marginBottom: '15px',
                            borderBottom: '1px solid #333',
                            paddingBottom: '8px'
                        }}>
                            <strong style={{ fontSize: '16px' }}>📊 Test Results</strong>
                            {completed > 0 && (
                                <div style={{ fontSize: '13px', color: '#00ff88', textAlign: 'right' }}>
                                    <div>Completed: {completed}/{total}</div>
                                    {bertRtScores.length > 0 && (
                                        <div>Accuracy: {globalAccuracy.toFixed(1)}%</div>
                                    )}
                                </div>
                            )}
                        </div>
                        <pre style={{ 
                            fontFamily: 'Consolas, Monaco, monospace',
                            whiteSpace: 'pre-wrap',
                            height: '500px',
                            overflowY: 'auto',
                            margin: 0,
                            fontSize: '13px',
                            lineHeight: '1.5',
                            color: '#f0f0f0'
                        }}>
                            {results || 'MultiAgent batch test results will appear here...\n\nConfigure settings, load a JSON file, and click "Start Batch Test" to begin.'}
                            <div ref={endRef} />
                        </pre>
                    </div>
                </Card>
            </div>
        </div>
    );
};

export default TestMultiAgentBatch;