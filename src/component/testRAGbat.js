import React, { useState, useEffect, useCallback, useMemo, useRef } from 'react';
import { DataTable } from 'primereact/datatable';
import { Column } from 'primereact/column';
import { InputText } from 'primereact/inputtext';
import { Button } from 'primereact/button';
import { Ollama } from '@langchain/ollama';
import { OllamaEmbeddings } from '@langchain/ollama';
import { Chroma } from "@langchain/community/vectorstores/chroma";
import { ChromaClient } from "chromadb";
import { ProgressBar } from 'primereact/progressbar';
import { RetrievalQAChain } from "langchain/chains";
import { loadQAStuffChain } from "langchain/chains";
import axios from 'axios';
import { FilterMatchMode } from 'primereact/api';
import './testRAGbat.css';
import configuration from './configuration.json';
import { ScoreThresholdRetriever } from "langchain/retrievers/score_threshold";

const TestRAGbat = () => {
    const [selectedChromaDB, setSelectedChromaDB] = useState('');
    const [selectedOllama, setSelectedOllama] = useState(null);
    const [selectedModel, setSelectedModel] = useState(null);
    const [collections, setCollections] = useState([]);
    const [dialogVisible, setDialogVisible] = useState(true);
    const [messages, setMessages] = useState([]);
    const [newMessage, setNewMessage] = useState('');
    const [selectedDB, setSelectedDB] = useState('');
    const [isSubmitting, setIsSubmitting] = useState(false);
    const [expectedAnswer, setExpectedAnswer] = useState('');
    const [testName, setTestName] = useState('');
    const [QAJSON, setQAJSON] = useState([{}]);
    const [isTesting, setIsTesting] = useState(false);
    const [total, setTotal] = useState(0);
    const [completed, setCompleted] = useState(0);
    const [symScore, setSymScore] = useState(0);
    const [k, setK] = useState(0);
    const [kInc, setKInc] = useState(0);
    const [results, setResults] = useState('');
    
    // Add new state variables for BERT RT tracking
    const [bertRtScores, setBertRtScores] = useState([]);
    const [globalAccuracy, setGlobalAccuracy] = useState(0);

    // Add separate embedding model state
    const [selectedEmbeddingModel, setSelectedEmbeddingModel] = useState('mxbai-embed-large');
    const [embeddingModelStatus, setEmbeddingModelStatus] = useState('unknown'); // 'working', 'failed', 'unknown'
    
    // Use refs to prevent multiple initialization runs
    const initRef = useRef(false);
    const embeddingTestRef = useRef(false);
    
    // Common embedding models to try as fallbacks
    const FALLBACK_EMBEDDING_MODELS = [
        'mxbai-embed-large',
        // 'nomic-embed-text',
        // 'all-minilm',
        // 'snowflake-arctic-embed',
        // 'bge-large',
        // 'bge-base'
    ];

    // Memoize the embeddings instance to prevent recreating on every render
    const embeddings_open = useMemo(() => {
        if (!selectedEmbeddingModel || !selectedOllama) return null;
        return new OllamaEmbeddings({
            model: selectedEmbeddingModel,
            baseUrl: selectedOllama 
        });
    }, [selectedEmbeddingModel, selectedOllama]);

    const mdl = new Ollama({
        baseUrl: selectedOllama, 
        model: selectedModel, 
        requestOptions: {
            num_gpu: 1,
        }
    });  

    // Enhanced listCollections function with better caching
    const listCollections = useCallback(async (ch) => {
        if (!ch) return;
        
        try {
            console.log("Fetching collections from:", ch);
            const client = new ChromaClient({path: ch});
            let collection2 = await client.listCollections();
            console.log("Collections received:", collection2);
            const collectionsWithIdAndName = collection2.map((collection, index) => ({
                name: collection,
                id: index,
            }));
            setCollections(collectionsWithIdAndName);
        } catch (e) {
            console.error("Error fetching collections:", e);
            setCollections([]);
        }
    }, []);

    // ...existing code...
    
    
    // Simplified embedding check (replace the complex version)
    const testEmbeddingSupport = useCallback(async (modelName, ollamaUrl) => {
        if (embeddingTestRef.current) {
            console.log("Embedding test already completed, skipping...");
            return;
        }

        embeddingTestRef.current = true;
        console.log("Testing embedding support for model:", modelName);

        try {
            const testEmbeddings = new OllamaEmbeddings({
                model: modelName,
                baseUrl: ollamaUrl
            });
            
            // Test with a simple query with timeout
            const result = await Promise.race([
                testEmbeddings.embedQuery("test"),
                new Promise((_, reject) => 
                    setTimeout(() => reject(new Error("Embedding test timeout")), 8000)
                )
            ]);
            
            if (result && Array.isArray(result) && result.length > 0) {
                setSelectedEmbeddingModel(modelName);
                setEmbeddingModelStatus('working');
                console.log("✅ Model supports embeddings:", modelName);
                return true;
            } else {
                console.log("❌ Model returned invalid embedding result");
            }
            
        } catch (error) {
            console.log("❌ Embedding test failed for", modelName, ":", error.message);
        }

        // Try fallback models with expanded list
        console.log("🔄 Trying fallback embedding models...");
        const FALLBACK_EMBEDDING_MODELS = [
            'mxbai-embed-large',
            'nomic-embed-text',
            'all-minilm',
            'snowflake-arctic-embed',
            'bge-large',
            'bge-base'
        ];
        
        for (const fallbackModel of FALLBACK_EMBEDDING_MODELS) {
            if (fallbackModel === modelName) continue; // Skip if it's the same model we just tried
            
            try {
                console.log("Testing fallback model:", fallbackModel);
                const fallbackEmbeddings = new OllamaEmbeddings({
                    model: fallbackModel,
                    baseUrl: ollamaUrl
                });
                
                const fallbackResult = await Promise.race([
                    fallbackEmbeddings.embedQuery("test"),
                    new Promise((_, reject) => 
                        setTimeout(() => reject(new Error("Fallback test timeout")), 5000)
                    )
                ]);

                if (fallbackResult && Array.isArray(fallbackResult) && fallbackResult.length > 0) {
                    setSelectedEmbeddingModel(fallbackModel);
                    setEmbeddingModelStatus('working');
                    console.log("✅ Fallback embedding model working:", fallbackModel);
                    return true;
                }
            } catch (fallbackError) {
                console.log("❌ Fallback model failed:", fallbackModel, fallbackError.message);
            }
        }
        
        // If all models fail, set status but keep a default
        setEmbeddingModelStatus('failed');
        setSelectedEmbeddingModel('mxbai-embed-large'); // Keep default for manual override
        console.log("❌ All embedding models failed - you may need to pull an embedding model");
        return false;
    }, []);
    
    // Simplified useEffect (remove complex branching)
    useEffect(() => {
        const fetchData = async () => {
            if (initRef.current) {
                console.log("Initialization already completed, skipping...");
                return;
            }
            
            initRef.current = true;
            console.log("useEffect triggered - initializing data");
            
            setSymScore(Number(localStorage.getItem("symScore")) || 0.9);
            setK(Number(localStorage.getItem("k")) || 100);
            setKInc(Number(localStorage.getItem("kInc")) || 2);
            const ch = localStorage.getItem("selectedChromaDB") || 'http://127.0.0.1:8000';
            setSelectedChromaDB(ch);
            const ol = localStorage.getItem("selectedOllama") || 'http://127.0.0.1:11434';
            setSelectedOllama(ol);
            const init_mdl = localStorage.getItem("selectedLLMModel") || 'mistral';
            setSelectedModel(init_mdl);

            console.log("Initial values:", { ch, ol, init_mdl });

            // Fetch collections first
            await listCollections(ch);
            
            // Test embedding support with simpler approach
            if (ol && init_mdl) {
                await testEmbeddingSupport(init_mdl, ol);
            } else {
                console.log("❌ Missing Ollama URL or model - using default embedding model");
                setSelectedEmbeddingModel('mxbai-embed-large');
                setEmbeddingModelStatus('unknown');
            }

            console.log("Python", configuration.passer.PythonScore);
            console.log("✅ Initialization completed");
        };

        fetchData();
    }, []);

    // Get status colors for embedding model
    const getEmbeddingStatusColor = () => {
        switch (embeddingModelStatus) {
            case 'working': return '#4caf50';
            case 'failed': return '#f44336';
            default: return '#ff9800';
        }
    };

    const getEmbeddingStatusText = () => {
        switch (embeddingModelStatus) {
            case 'working': return '✅ Working';
            case 'failed': return '❌ Failed';
            default: return '⚠️ Unknown';
        }
    };

    const rowClass = (rowData) => {
        return {
            'p-highlight': rowData.name === selectedDB,
        };
    };

    const onRowSelect = async (e) => {
        setSelectedDB(e.data.name);
    };

    function handleFileChange(files) {
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
    }
    
    const startTest = async () => {
        if (!QAJSON || QAJSON.length === 0 || !QAJSON[0].question) {
            alert('Please load a valid JSON file with questions first.');
            return;
        }

        if (!selectedDB) {
            alert('Please select a database collection first.');
            return;
        }

        if (embeddingModelStatus === 'failed') {
            alert(`Cannot start test: No working embedding model available.\n\nPlease:\n1. Check if you have an embedding model installed in Ollama\n2. Try running: ollama pull ${selectedEmbeddingModel}\n3. Or try another embedding model`);
            return;
        }

        setIsTesting(true);
        setTotal(QAJSON.length);
        setResults('');
        setCompleted(0);
        
        // Reset BERT RT tracking for new test
        setBertRtScores([]);
        setGlobalAccuracy(0);
        
        var retriever;
        
        for (let i = 0; i < QAJSON.length; i++) {
            console.log("QAJSON[i]", QAJSON[i], i, selectedDB);
            const question = QAJSON[i].question;
            const answer = QAJSON[i].answer;

            const mdl1 = new Ollama({
                baseUrl: selectedOllama,
                model: selectedModel 
            });  
            
            const vectorStore1 = await Chroma.fromExistingCollection(
                embeddings_open, // Use the memoized embeddings instance
                {
                    collectionName: selectedDB,
                    url: selectedChromaDB 
                });

            if (localStorage.getItem("retriever") === 'Normal') {
                retriever = vectorStore1.asRetriever({k: 100});
                console.log("retriever NORMAL");
            } else {
                retriever = ScoreThresholdRetriever.fromVectorStore(vectorStore1, {
                    minSimilarityScore: Number(localStorage.getItem("symScore")) || 0.9,
                    maxK: Number(localStorage.getItem("k")) || 100,
                    kIncrement: Number(localStorage.getItem("kInc")) || 2,
                    k: 100,
                });
                console.log("retriever SCORE");
            }

            console.log("retriever", retriever, 'MDL', mdl, 'Embedding Model', embeddings_open);

            const chain = new RetrievalQAChain({
                combineDocumentsChain: loadQAStuffChain(mdl),
                retriever,
                returnSourceDocuments: true,
                inputKey: "question",
            });
    
            try {
                const res = await chain.invoke({
                    question: question,
                });  
        
                console.log("answer", res.text);
                
                const metrics = {
                    reference: QAJSON[i].answer, 
                    candidate: res.text, 
                    userID: localStorage.getItem("wharf_user_name"), 
                    testID: testName,
                    description: 'question: ' + question.replace(/[^\x00-\x7F]/g, "") + ', answer: ' + res.text.replace(/[^\x00-\x7F]/g, "")
                };
                
                console.log("to back ----> ", metrics.description);
                
                try {
                    const response = await axios.post(configuration.passer.PythonScore, metrics);
                    
                    // Now you can access the returned metrics
                    if (response.data && response.data.metrics) {
                        const calculatedMetrics = response.data.metrics;
                        console.log("Received metrics:", calculatedMetrics);
                        
                        // Get BERT RT score and update tracking
                        const bertRtAvgScore = calculatedMetrics.bert_rt_score.avg_score;
                        
                        // Update BERT RT scores array
                        const updatedBertRtScores = [...bertRtScores, bertRtAvgScore];
                        setBertRtScores(updatedBertRtScores);
                        
                        // Calculate global accuracy based on mean BERT RT score
                        const meanBertRtScore = updatedBertRtScores.reduce((sum, score) => sum + score, 0) / updatedBertRtScores.length;
                        const calculatedGlobalAccuracy = (meanBertRtScore / 5.0) * 100; // Convert to percentage
                        setGlobalAccuracy(calculatedGlobalAccuracy);
                        
                        // Calculate individual match based on BERT RT score
                        const matchPercentage = (bertRtAvgScore / 5.0) * 100;
                        const matchThreshold = 70; // You can adjust this threshold
                        const isMatch = matchPercentage >= matchThreshold;
                        
                        console.log(`BERT RT Score: ${bertRtAvgScore.toFixed(2)}`);
                        console.log(`Global Accuracy: ${calculatedGlobalAccuracy.toFixed(1)}%`);
                        console.log(`Individual Match: ${isMatch} (${matchPercentage.toFixed(1)}%)`);
                        
                        setResults(prevResults => 
                            prevResults + 
                            `${i+1}-> Question: ${question}\n` + 
                            `Reference: ${QAJSON[i].answer}\n` + 
                            `Answer: ${res.text}\n` +
                            `\n\n`
                        );
                    } else {
                        // Fallback to original display if metrics not available
                        setResults(prevResults => 
                            prevResults + 
                            `${i+1}-> Question: ${question}\n` + 
                            `Reference: ${QAJSON[i].answer}\n` + 
                            `Answer: ${res.text}\n\n`
                        );
                    }
                    
                } catch (error) {
                    console.error('Error sending data to Python server:', error);
                    setResults(prevResults => 
                        prevResults + 
                        `${i+1}-> Question: ${question}\n` + 
                        `Reference: ${QAJSON[i].answer}\n` + 
                        `Answer: ${res.text}\n` +
                        `Error: Could not calculate metrics\n\n`
                    );
                }
                
            } catch (error) {
                console.error('Error in RAG chain:', error);
                setResults(prevResults => 
                    prevResults + 
                    `${i+1}-> Question: ${question}\n` + 
                    `Reference: ${QAJSON[i].answer}\n` + 
                    `Answer: ERROR - ${error.message}\n` +
                    `Status: ❌ ERROR\n\n`
                );
            }
            
            setCompleted(i + 1);
        }
        setIsTesting(false);
    };

    const [filters, setFilters] = useState({
        global: { value: null, matchMode: FilterMatchMode.CONTAINS },
        name: { value: null, matchMode: FilterMatchMode.STARTS_WITH },
    });

    const retrieverColor = localStorage.getItem("retriever") === "Normal" ? '#4caf50' : '#ff9800';
    const endRef = React.useRef(null);

    React.useEffect(() => {
        endRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [results]);

    return (
        <div style={{ width: '100%', display: 'flex', justifyContent: 'space-between'}}>
            {/* Database & Configuration Panel */}
            <div style={{ width: '35%', padding: '20px' }}>
                <h3>🗄️ RAG Database Configuration</h3>
                
                {/* ChromaDB URL */}
                <div style={{ marginBottom: '15px' }}>
                    <label><strong>ChromaDB URL:</strong></label>
                    <InputText 
                        style={{ width: '100%', marginTop: '5px' }} 
                        value={selectedChromaDB || ''} 
                        onChange={(e) => {
                            setSelectedChromaDB(e.target.value);
                            localStorage.setItem("selectedChromaDB", e.target.value);
                            listCollections(e.target.value);
                        }}
                        placeholder="http://127.0.0.1:8000"
                    />
                </div>

                {/* Ollama URL */}
                <div style={{ marginBottom: '15px' }}>
                    <label><strong>Ollama Server URL:</strong></label>
                    <InputText 
                        style={{ width: '100%', marginTop: '5px' }} 
                        value={selectedOllama || ''} 
                        onChange={(e) => {
                            setSelectedOllama(e.target.value);
                            localStorage.setItem("selectedOllama", e.target.value);
                        }}
                        placeholder="http://127.0.0.1:11434"
                    />
                </div>

                {/* Model Name */}
                <div style={{ marginBottom: '15px' }}>
                    <label><strong>Model Name:</strong></label>
                    <InputText 
                        style={{ width: '100%', marginTop: '5px' }} 
                        value={selectedModel || ''} 
                        onChange={(e) => {
                            setSelectedModel(e.target.value);
                            localStorage.setItem("selectedLLMModel", e.target.value);
                        }}
                        placeholder="mistral"
                    />
                </div>

                {/* Embedding Model Selector */}
                <div style={{ marginBottom: '15px' }}>
                    <label><strong>Embedding Model:</strong></label>
                    <InputText 
                        style={{ width: '100%', marginTop: '5px' }} 
                        value={selectedEmbeddingModel} 
                        onChange={(e) => {
                            const value = e.target.value;
                            setSelectedEmbeddingModel(value);
                            setEmbeddingModelStatus('unknown');
                            // Reset embedding test when model changes
                            embeddingTestRef.current = false;
                        }}
                        placeholder="mxbai-embed-large"
                    />
                    <div style={{ display: 'flex', gap: '5px', marginTop: '5px' }}>
                        <Button 
                            label="Test Model" 
                            size="small"
                            style={{ fontSize: '12px', flex: 1 }}
                            onClick={async () => {
                                embeddingTestRef.current = false;
                                setEmbeddingModelStatus('unknown');
                                await testEmbeddingSupport(selectedEmbeddingModel, selectedOllama, false); // Pass false for manual test
                            }}
                        />
                        <div style={{ 
                            display: 'flex', 
                            alignItems: 'center', 
                            padding: '5px 10px',
                            backgroundColor: getEmbeddingStatusColor(),
                            color: 'white',
                            borderRadius: '4px',
                            fontSize: '12px',
                            minWidth: '80px',
                            justifyContent: 'center'
                        }}>
                            {getEmbeddingStatusText()}
                        </div>
                    </div>
                </div>

                {/* Retriever Status */}
                <div style={{ 
                    padding: '12px', 
                    backgroundColor: '#f5f5f5', 
                    border: `2px solid ${retrieverColor}`, 
                    borderRadius: '8px',
                    marginBottom: '15px',
                    fontSize: '14px'
                }}>
                    <strong style={{ color: retrieverColor }}>
                        🔍 {localStorage.getItem("retriever") || "Normal"} Retriever
                    </strong>
                    {localStorage.getItem("retriever") === "Score" && (
                        <div style={{ fontSize: '12px', color: '#666', marginTop: '5px' }}>
                            Similarity: {localStorage.getItem("symScore") || '0.9'} | 
                            K: {localStorage.getItem("k") || '100'} | 
                            K-Inc: {localStorage.getItem("kInc") || '2'}
                        </div>
                    )}
                </div>

                {/* Database Collections */}
                <div style={{ marginBottom: '15px' }}>
                    <label><strong>Available Collections:</strong></label>
                    <DataTable 
                        style={{ width: '100%', marginTop: '5px' }} 
                        value={collections}   
                        onRowSelect={onRowSelect} 
                        selectionMode="single" 
                        selection={selectedDB}
                        rowClassName={rowClass} 
                        filters={filters} 
                        filterDisplay="row"
                        size="small" 
                        showGridlines 
                        stripedRows
                        scrollable
                        scrollHeight="200px"
                    >
                        <Column field="name" header="Collection Name" filter filterPlaceholder="Filter by name"></Column>
                    </DataTable>
                </div>

                {/* Selected Database Status */}
                <div style={{ 
                    padding: '12px', 
                    backgroundColor: selectedDB ? '#e8f5e8' : '#fff5f5', 
                    border: `2px solid ${selectedDB ? '#4caf50' : '#f44336'}`, 
                    borderRadius: '8px',
                    marginBottom: '15px',
                    fontSize: '14px'
                }}>
                    <strong>Selected Database:</strong> {selectedDB || 'None selected'}
                </div>

                {/* Global Accuracy Display */}
                {bertRtScores.length > 0 && (
                    <div style={{ 
                        padding: '15px', 
                        backgroundColor: globalAccuracy >= 70 ? '#e8f5e8' : '#fff5f5', 
                        border: `2px solid ${globalAccuracy >= 70 ? '#4caf50' : '#ff9800'}`, 
                        borderRadius: '8px',
                        fontSize: '14px',
                        marginBottom: '15px'
                    }}>
                        <strong>🎯 Global RAG Accuracy</strong><br/>
                        <div style={{ fontSize: '24px', fontWeight: 'bold', marginTop: '8px', color: globalAccuracy >= 70 ? '#4caf50' : '#ff9800' }}>
                            {globalAccuracy.toFixed(1)}%
                        </div>
                        <div style={{ fontSize: '12px', color: '#666', marginTop: '5px' }}>
                            Mean BERT RT Score: {(bertRtScores.reduce((sum, score) => sum + score, 0) / bertRtScores.length).toFixed(2)}
                        </div>
                    </div>
                )}

                {/* RAG Testing Info */}
                <div style={{ 
                    padding: '15px', 
                    backgroundColor: '#f0f8ff', 
                    border: '1px solid #87ceeb', 
                    borderRadius: '8px',
                    fontSize: '13px',
                    lineHeight: '1.4',
                    marginBottom: '15px'
                }}>
                    <strong>ℹ️ RAG Testing</strong><br/>
                    This interface tests your RAG system using vector databases and document retrieval. 
                    Select a collection and configure your retriever settings for optimal results.
                </div>

                {/* Enhanced Embedding Configuration Status */}
                <div style={{ 
                    padding: '15px', 
                    backgroundColor: embeddingModelStatus === 'working' ? '#e8f5e8' : (embeddingModelStatus === 'failed' ? '#ffebee' : '#fff3e0'), 
                    border: `1px solid ${getEmbeddingStatusColor()}`, 
                    borderRadius: '8px',
                    fontSize: '13px',
                    lineHeight: '1.4'
                }}>
                    <strong>🔧 Embedding Configuration</strong><br/>
                    <strong>Status:</strong> {getEmbeddingStatusText()}<br/>
                    <strong>Model:</strong> {selectedEmbeddingModel}<br/>
                    {embeddingModelStatus === 'failed' && (
                        <>
                            <br/>
                            <strong style={{ color: '#f44336' }}>⚠️ No working embedding model found!</strong><br/>
                            <small>Run: <code>ollama pull {selectedEmbeddingModel}</code></small>
                        </>
                    )}
                    {embeddingModelStatus === 'working' && selectedEmbeddingModel !== selectedModel && (
                        <>
                            <br/>
                            <small style={{ color: '#666' }}>
                                💡 Using dedicated embedding model (recommended)
                            </small>
                        </>
                    )}
                </div>
            </div>

            {/* Testing Panel */}
            {dialogVisible &&
            <div style={{ width: '60%', marginLeft: '20px', padding: '20px'}}>
                <h3>📋 RAG System Testing</h3>
                
                <div style={{ 
                    display: 'flex', 
                    flexDirection: 'row', 
                    alignItems: 'center', 
                    justifyContent: 'space-between', 
                    width: '100%',
                    marginBottom: '20px'
                }}>
                    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', width: '45%' }}>
                        <label><strong>Test Name:</strong></label>
                        <InputText 
                            id="testName" 
                            style={{ width: '100%', marginTop: '5px' }} 
                            value={testName} 
                            onChange={(e) => {
                                const value = e.target.value;
                                if (value === '' || (value.length <= 12 && /^[a-z1-5]+$/.test(value))) {
                                    setTestName(e.target.value);
                                }
                            }}
                            placeholder="test123"
                        />   
                    </div>
                    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', width: '45%' }}>
                        <label htmlFor="fileInput"><strong>Select JSON Questions:</strong></label>
                        <input 
                            type="file" 
                            id="fileInput" 
                            style={{ width: '100%', marginTop: '5px' }} 
                            accept=".json"
                            onChange={(e) => handleFileChange(e.target.files)} 
                        />
                    </div>
                </div>
                
                {/* Enhanced Test Status */}
                <div style={{ 
                    padding: '12px', 
                    backgroundColor: (QAJSON && QAJSON.length > 0 && QAJSON[0].question && selectedDB && embeddingModelStatus === 'working') ? '#e8f5e8' : '#fff5f5', 
                    border: `2px solid ${(QAJSON && QAJSON.length > 0 && QAJSON[0].question && selectedDB && embeddingModelStatus === 'working') ? '#4caf50' : '#f44336'}`, 
                    borderRadius: '8px',
                    marginBottom: '15px',
                    fontSize: '14px'
                }}>
                    <strong>Status:</strong> {
                        !selectedDB ? '⚠️ Please select a database collection first' :
                        embeddingModelStatus === 'failed' ? '❌ Embedding model not available - cannot test' :
                        embeddingModelStatus === 'unknown' ? '⚠️ Embedding model status unknown - test embedding model first' :
                        (QAJSON && QAJSON.length > 0 && QAJSON[0].question) 
                            ? `✅ Ready to test ${QAJSON.length} questions with ${selectedDB}` 
                            : '⚠️ Please load a JSON file with questions in format: [{"question":"q1", "answer":"a1"}, ...]'
                    }
                </div>

                <Button 
                    label={isTesting ? "🔄 Testing..." : "🚀 Start RAG Test"} 
                    style={{
                        marginTop: '10px', 
                        width: '100%',
                        padding: '12px',
                        fontSize: '16px',
                        fontWeight: 'bold'
                    }} 
                    onClick={() => startTest()} 
                    disabled={isTesting || !QAJSON || QAJSON.length === 0 || !QAJSON[0].question || !selectedDB || embeddingModelStatus === 'failed'}
                    className="p-ml-2" 
                />    
                
                {isTesting && (
                    <div style={{ marginTop: '15px' }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
                            <span><strong>Progress: {completed}/{total}</strong></span>
                            <span><strong>{((completed / total) * 100).toFixed(2)}%</strong></span>
                        </div>
                        <ProgressBar                             
                            value={((completed / total) * 100).toFixed(2)} 
                            style={{ width: '100%', height: '20px' }} 
                        />
                    </div>
                )}
                
                {/* Results Panel */}
                <div style={{ 
                    marginTop: '20px', 
                    backgroundColor: '#000', 
                    color: '#fff', 
                    padding: '15px', 
                    borderRadius: '8px', 
                    width: '100%', 
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
                        <strong style={{ fontSize: '16px' }}>📊 RAG Test Results</strong>
                        {completed > 0 && (
                            <div style={{ fontSize: '13px', color: '#00ff88', textAlign: 'right' }}>
                                <div>Individual Matches: {(results.match(/✅ MATCH/g) || []).length}/{completed}</div>
                                {bertRtScores.length > 0 && (
                                    <div>Global RAG Accuracy: {globalAccuracy.toFixed(1)}%</div>
                                )}
                            </div>
                        )}
                    </div>
                    <pre style={{ 
                        fontFamily: 'Consolas, Monaco, monospace', 
                        whiteSpace: 'pre-wrap', 
                        height: '400px', 
                        overflowY: 'auto', 
                        textAlign: 'left',
                        margin: 0,
                        fontSize: '13px',
                        lineHeight: '1.5',
                        color: '#f0f0f0'
                    }}>
                        {results || 'RAG test results will appear here...\n\nSelect a database collection, load a JSON file, and click "Start RAG Test" to begin.'}
                        <div ref={endRef} />
                    </pre>                
                </div>
            </div>}
        </div>
    );
}

export default TestRAGbat;