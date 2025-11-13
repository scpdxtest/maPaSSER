import React, { useState, useEffect, useCallback, useMemo, useRef } from 'react';
import { DataTable } from 'primereact/datatable';
import { Column } from 'primereact/column';
import {Ollama} from '@langchain/ollama';
import {OllamaEmbeddings} from '@langchain/ollama';
import { Chroma } from "@langchain/community/vectorstores/chroma";
import { ChromaClient } from "chromadb";
import { Button } from 'primereact/button';
import { InputText } from 'primereact/inputtext';
import './chatFromDB.css';
import { ProgressSpinner } from 'primereact/progressspinner';
import { BufferMemory } from "langchain/memory";
import { ConversationalRetrievalQAChain } from "langchain/chains";
import { FilterMatchMode } from 'primereact/api';
import { ScoreThresholdRetriever } from "langchain/retrievers/score_threshold";

const ChatFromDB = () => {
    const [selectedChromaDB, setSelectedChromaDB] = useState('');
    const [selectedOllama, setSelectedOllama] = useState('');
    const [selectedModel, setSelectedModel] = useState('');
    const [collections, setCollections] = useState([]);
    const [dialogVisible] = useState(true);
    const [messages, setMessages] = useState([]);
    const [newMessage, setNewMessage] = useState('');
    const [selectedDB, setSelectedDB] = useState('');
    const [selectedCollection, setSelectedCollection] = useState(null);
    const [isSubmitting, setIsSubmitting] = useState(false);
    const [tempreture, setTempreture] = useState(localStorage.getItem("chatTempreture") || '0.2');
    const [selectedEmbeddingModel, setSelectedEmbeddingModel] = useState('mxbai-embed-large');
    const [embeddingModelStatus, setEmbeddingModelStatus] = useState('unknown'); // 'working', 'failed', 'unknown'

    const [filters] = useState({
        global: { value: null, matchMode: FilterMatchMode.CONTAINS },
        name: { value: null, matchMode: FilterMatchMode.STARTS_WITH },
    });

    // Use refs to prevent multiple initialization runs
    const initRef = useRef(false);
    const embeddingTestRef = useRef(false);

    // Common embedding models to try as fallbacks
    const FALLBACK_EMBEDDING_MODELS = [
        'mxbai-embed-large',
        'nomic-embed-text',
        'all-minilm',
        'snowflake-arctic-embed',
        'bge-large',
        'bge-base'
    ];

    // Memoize the embeddings instance to prevent recreating on every render
    const embeddings_open = useMemo(() => {
        if (!selectedEmbeddingModel || !selectedOllama) return null;
        return new OllamaEmbeddings({
            model: selectedEmbeddingModel,
            baseUrl: selectedOllama 
        });
    }, [selectedEmbeddingModel, selectedOllama]);

    // Memoize the LLM instance
    const mdl = useMemo(() => {
        if (!selectedModel || !selectedOllama) return null;
        return new Ollama({
            baseUrl: selectedOllama, 
            model: selectedModel, 
            requestOptions: {
                num_gpu: 1,
                tempreture: tempreture,
            }
        });
    }, [selectedModel, selectedOllama, tempreture]);

    const [memory, setMemory] = useState(new BufferMemory({
        memoryKey: "chat_history",
        returnMessages: true,
    }));

    // listCollections function with better caching
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

    // Test embedding support function with multiple fallbacks
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
                    setTimeout(() => reject(new Error("Embedding test timeout")), 5000)
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

        // Try fallback models
        console.log("🔄 Trying fallback embedding models...");
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
                        setTimeout(() => reject(new Error("Fallback test timeout")), 3000)
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
    
    useEffect(() => {
        const fetchData = async () => {
            // Prevent multiple runs using ref
            if (initRef.current) {
                console.log("Initialization already completed, skipping...");
                return;
            }
            
            initRef.current = true;
            console.log("useEffect triggered - initializing data");
            
            const ch = localStorage.getItem("selectedChromaDB") || 'http://127.0.0.1:8000';
            const ol = localStorage.getItem("selectedOllama");
            const init_mdl = localStorage.getItem("selectedLLMModel") || 'mistral';
            
            console.log("Initial values:", { ch, ol, init_mdl });
            
            setSelectedChromaDB(ch);
            setSelectedOllama(ol);
            setSelectedModel(init_mdl);
            
            // Fetch collections first
            await listCollections(ch);
            
            // Test embedding support with better error handling
            if (ol && init_mdl) {
                await testEmbeddingSupport(init_mdl, ol);
            } else {
                console.log("❌ Missing Ollama URL or model - using default embedding model");
                setSelectedEmbeddingModel('mxbai-embed-large');
                setEmbeddingModelStatus('unknown');
            }
            
            console.log("✅ Initialization completed");
        };
        
        fetchData();
    }, []); // Empty dependency array to run only once
    
    const onRowDoubleClick = async (e) => {
        if (!embeddings_open || !selectedChromaDB) return;
        
        try {
            const client = new ChromaClient({path: selectedChromaDB});
            let collection2 = await client.getCollection({name: e.data.name});
            console.log("Collection count:", await collection2.count());
            console.log("Collection peek:", await collection2.peek(10));

            const vectorStore1 = await Chroma.fromExistingCollection(
                embeddings_open,
                {
                    collectionName: e.data.name,
                    url: selectedChromaDB,
                });

            let retriever;
            if (localStorage.getItem("retriever") === 'Normal') {
                retriever = vectorStore1.asRetriever();
            } else {
                retriever = ScoreThresholdRetriever.fromVectorStore(vectorStore1, {
                    minSimilarityScore: 0.9,
                    maxK: 100,
                    kIncrement: 2,
                });
            }    
            console.log(`Double clicked on collection with ID: ${e.data.id}`, retriever);
        } catch (error) {
            console.error("Error in onRowDoubleClick:", error);
        }
    };

    const rowClass = (rowData) => {
        return {
            'p-highlight': rowData.name === selectedDB,
        };
    };

    const copyToClipboard = (textToCopy) => {
        navigator.clipboard.writeText(textToCopy);
    };

    const testCollectionCompatibility = useCallback(async (collectionName, embeddingModel) => {
        if (!embeddings_open || !collectionName) {
            console.log("❌ Cannot test compatibility - missing requirements");
            return false;
        }
        
        try {
            console.log("Testing collection compatibility...");
            console.log("Using embedding model:", embeddingModel);
            console.log("Testing collection:", collectionName);
            
            const vectorStore = await Chroma.fromExistingCollection(
                embeddings_open,
                {
                    collectionName: collectionName,
                    url: selectedChromaDB 
                });
            
            // Try a simple query to test compatibility
            const testResult = await vectorStore.similaritySearch("test", 1);
            console.log("✅ Collection compatibility test passed");
            console.log("Test query returned:", testResult.length, "documents");
            
            if (testResult.length > 0) {
                console.log("Sample document:", {
                    content: testResult[0].pageContent.substring(0, 100) + "...",
                    source: testResult[0].metadata?.source || "unknown"
                });
            }
            
            return true;
        } catch (error) {
            console.error("❌ Collection compatibility test failed:", error);
            console.error("Error details:", {
                name: error.name,
                message: error.message,
                stack: error.stack?.substring(0, 200) + "..."
            });
            return false;
        }
    }, [embeddings_open, selectedChromaDB]);
    
    // Update the onRowSelect function to pass current values directly
    const onRowSelect = async (e) => {
        console.log("Row selected:", e.data);
        const newCollectionName = e.data.name;
        
        // Update state immediately
        setSelectedDB(newCollectionName);
        setSelectedCollection(e.data);
        setMemory(new BufferMemory({
            memoryKey: "chat_history",
            returnMessages: true,
        }));
        
        // Test compatibility with the newly selected collection (pass values directly)
        if (embeddings_open && embeddingModelStatus === 'working') {
            console.log("Testing compatibility for newly selected collection:", newCollectionName);
            const isCompatible = await testCollectionCompatibility(newCollectionName, selectedEmbeddingModel);
            if (!isCompatible) {
                console.warn("⚠️ Collection may not be compatible with current embedding model");
            }
        } else {
            console.log("⚠️ Skipping compatibility test - embedding model not working");
        }
    };

    const CUSTOM_QUESTION_GENERATOR_CHAIN_PROMPT = `Given the following conversation and a follow up question, return the conversation history excerpt that includes any relevant context to the question if it exists and rephrase the follow up question to be a standalone question.
    Chat History:
    {chat_history}
    Follow Up Input: {question}
    Your answer should follow the following format:
    \`\`\`
    Use the following pieces of context to answer the users question.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    ----------------
    <Relevant chat history excerpt as context here>
    Standalone question: <Rephrased question here>
    \`\`\`
    Your answer:`;
    
    const handleSubmit = async () => {
        if (!embeddings_open || !mdl || !selectedDB) {
            console.log("❌ Missing required components:", {
                embeddings_open: !!embeddings_open,
                mdl: !!mdl,
                selectedDB: !!selectedDB
            });
            return;
        }
        
        if (embeddingModelStatus === 'failed') {
            setMessages([...messages, 
                { text: newMessage, sender: 'user', stat: '' }, 
                { text: '❌ Cannot process question: No working embedding model available.\n\nPlease:\n1. Check if you have an embedding model installed in Ollama\n2. Try running: ollama pull mxbai-embed-large\n3. Or try another embedding model', sender: selectedModel, stat: '' }
            ]);
            setNewMessage('');
            return;
        }
        
        setIsSubmitting(true);
        try {
            const question = newMessage;
            setNewMessage('');
            setMessages([...messages, { text: question, sender: 'user', stat: '' }, { text: '', sender: selectedModel, stat: '' }]);
    
            // Add error handling and validation for vector store connection
            let vectorStore1;
            try {
                console.log("Attempting to connect to collection:", selectedDB);
                console.log("Using embedding model:", selectedEmbeddingModel);
                console.log("ChromaDB URL:", selectedChromaDB);
                
                vectorStore1 = await Chroma.fromExistingCollection(
                    embeddings_open,
                    {
                        collectionName: selectedDB,
                        url: selectedChromaDB 
                    });
                
                console.log("✅ Vector store connected successfully");
            } catch (vectorError) {
                console.error("❌ Vector store connection failed:", vectorError);
                
                // Update the last message with error
                setMessages(prevMessages => {
                    let newMessages = [...prevMessages];
                    if (newMessages.length >= 0 && newMessages[newMessages.length - 1].sender === selectedModel) {
                        newMessages[newMessages.length - 1].text = `❌ Error connecting to database: ${vectorError.message}\n\nThis might be due to:\n1. Embedding model mismatch\n2. Collection not found\n3. ChromaDB connection issues\n4. Embedding model not available\n\nTry running: ollama pull ${selectedEmbeddingModel}`;
                    }
                    return newMessages;
                });
                setIsSubmitting(false);
                return;
            }
    
            var retriever;
            try {
                console.log("Creating retriever with config:", {
                    retrieverType: localStorage.getItem("retriever"),
                    symScore: localStorage.getItem("symScore"),
                    k: localStorage.getItem("k"),
                    kInc: localStorage.getItem("kInc")
                });
    
                if (localStorage.getItem("retriever") === 'Normal') {
                    retriever = vectorStore1.asRetriever({k: 100});
                    console.log("✅ Normal Retriever created successfully");
                } else {
                    retriever = ScoreThresholdRetriever.fromVectorStore(vectorStore1, {
                        minSimilarityScore: Number(localStorage.getItem("symScore")) || 0.9,
                        maxK: Number(localStorage.getItem("k")) || 100,
                        kIncrement: Number(localStorage.getItem("kInc")) || 2,
                        k: 100,
                    });
                    console.log("✅ ScoreThresholdRetriever created successfully");
                }
            } catch (retrieverError) {
                console.error("❌ Retriever creation failed:", retrieverError);
                
                setMessages(prevMessages => {
                    let newMessages = [...prevMessages];
                    if (newMessages.length >= 0 && newMessages[newMessages.length - 1].sender === selectedModel) {
                        newMessages[newMessages.length - 1].text = `❌ Error creating retriever: ${retrieverError.message}`;
                    }
                    return newMessages;
                });
                setIsSubmitting(false);
                return;
            }
    
            let retrievedDocs;
            try {
                console.log("Retrieving documents for question:", question);
                retrievedDocs = await retriever.getRelevantDocuments(question);
                console.log("✅ Document retrieval successful");
                console.log("Number of chunks retrieved:", retrievedDocs.length);
                console.log("Retrieved chunks preview:", retrievedDocs.map(doc => ({
                    content: doc.pageContent.substring(0, 100) + "...",
                    source: doc.metadata?.source || "unknown"
                })));
            } catch (retrievalError) {
                console.error("❌ Document retrieval failed:", retrievalError);
                
                setMessages(prevMessages => {
                    let newMessages = [...prevMessages];
                    if (newMessages.length >= 0 && newMessages[newMessages.length - 1].sender === selectedModel) {
                        newMessages[newMessages.length - 1].text = `❌ Error retrieving documents: ${retrievalError.message}\n\nThis usually indicates an embedding model mismatch. The collection was likely created with a different embedding model than the one currently selected (${selectedEmbeddingModel}).\n\nTry running: ollama pull ${selectedEmbeddingModel}`;
                    }
                    return newMessages;
                });
                setIsSubmitting(false);
                return;
            }
            
            try {
                console.log("Creating conversational chain...");
                const chain = ConversationalRetrievalQAChain.fromLLM(
                    mdl,
                    retriever,
                    { 
                        memory, 
                        questionGeneratorChainOptions: {
                            template: CUSTOM_QUESTION_GENERATOR_CHAIN_PROMPT,
                        },
                    },
                );
                console.log("✅ Chain created successfully");
    
                console.log("Invoking chain with question:", question);
                const res = await chain.invoke({
                    question: question,
                });  
    
                console.log("✅ Chain execution successful");
                console.log("Answer received:", res.text);
    
                setMessages(prevMessages => {
                    let newMessages = [...prevMessages];
                    if (newMessages.length >= 0 && newMessages[newMessages.length - 1].sender === selectedModel) {
                        newMessages[newMessages.length - 1].text = res.text;
                    } else {
                        newMessages.push({ text: res.text, sender: selectedModel, stat: '' });
                    }
                    return newMessages;
                });
            } catch (chainError) {
                console.error("❌ Chain execution failed:", chainError);
                
                setMessages(prevMessages => {
                    let newMessages = [...prevMessages];
                    if (newMessages.length >= 0 && newMessages[newMessages.length - 1].sender === selectedModel) {
                        newMessages[newMessages.length - 1].text = `❌ Error in AI processing: ${chainError.message}\n\nThis could be due to:\n1. LLM server issues\n2. Memory/context problems\n3. Chain configuration errors`;
                    }
                    return newMessages;
                });
            }
        } catch (error) {
            console.error("❌ Unexpected error in handleSubmit:", error);
            
            // Update the UI with a more helpful error message
            setMessages(prevMessages => {
                let newMessages = [...prevMessages];
                if (newMessages.length >= 0 && newMessages[newMessages.length - 1].sender === selectedModel) {
                    newMessages[newMessages.length - 1].text = `❌ Chat Error: ${error.message}\n\nPossible solutions:\n1. Check if the embedding model matches the one used to create the collection\n2. Verify ChromaDB server is running and accessible\n3. Ensure the collection exists and has data\n4. Try running: ollama pull ${selectedEmbeddingModel}`;
                }
                return newMessages;
            });
        } finally {
            setIsSubmitting(false);
        }
    };
    
    const retrieverColor = localStorage.getItem("retriever") === "Normal" ? '#4caf50' : '#ff9800';

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

    return (
        <div style={{ width: '100%', display: 'flex', justifyContent: 'space-between'}}>
            {/* Configuration Panel */}
            <div style={{ width: '35%', padding: '20px' }}>
                <h3>🗄️ Chat Database Configuration</h3>
                
                {/* ChromaDB URL */}
                <div style={{ marginBottom: '15px' }}>
                    <label><strong>ChromaDB URL:</strong></label>
                    <InputText 
                        style={{ width: '100%', marginTop: '5px' }} 
                        value={selectedChromaDB || ''} 
                        onChange={(e) => {
                            const value = e.target.value;
                            setSelectedChromaDB(value);
                            localStorage.setItem("selectedChromaDB", value);
                            listCollections(value);
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
                        readOnly
                    />
                </div>

                {/* Model Name */}
                <div style={{ marginBottom: '15px' }}>
                    <label><strong>Model Name:</strong></label>
                    <InputText 
                        style={{ width: '100%', marginTop: '5px' }} 
                        value={selectedModel || ''} 
                        readOnly
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
                                await testEmbeddingSupport(selectedEmbeddingModel, selectedOllama);
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

                {/* Temperature */}
                <div style={{ marginBottom: '15px' }}>
                    <label><strong>Temperature:</strong></label>
                    <InputText 
                        style={{ width: '100%', marginTop: '5px' }} 
                        value={tempreture} 
                        onChange={(e) => {
                            const value = e.target.value;
                            setTempreture(value);
                            localStorage.setItem("chatTempreture", value);
                        }}
                        placeholder="0.2"
                    />
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
                        onRowDoubleClick={onRowDoubleClick}
                        selectionMode="single" 
                        selection={selectedCollection}
                        rowClassName={rowClass} 
                        filters={filters} 
                        filterDisplay="row"
                        size="small" 
                        showGridlines 
                        stripedRows
                        scrollable
                        scrollHeight="200px"
                        emptyMessage="No collections found. Check your ChromaDB connection."
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

                {/* Embedding Support Status */}
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

            {/* Chat Panel */}
            {dialogVisible &&
            <div style={{ width: '60%', marginLeft: '20px', padding: '20px'}}>
                <h3>💬 Chat with your Documents</h3>
                
                {/* Chat Status */}
                <div style={{ 
                    padding: '12px', 
                    backgroundColor: (selectedDB && embeddingModelStatus === 'working') ? '#e8f5e8' : '#fff5f5', 
                    border: `2px solid ${(selectedDB && embeddingModelStatus === 'working') ? '#4caf50' : '#f44336'}`, 
                    borderRadius: '8px',
                    marginBottom: '15px',
                    fontSize: '14px'
                }}>
                    <strong>Status:</strong> {
                        !selectedDB 
                            ? '⚠️ Please select a database collection first'
                            : embeddingModelStatus === 'failed'
                            ? '❌ Embedding model not available - cannot chat'
                            : embeddingModelStatus === 'working'
                            ? `✅ Ready to chat with ${selectedDB} collection`
                            : '⚠️ Embedding model status unknown'
                    }
                </div>

                {/* Chat Messages */}
                <div style={{ 
                    backgroundColor: '#f8f9fa', 
                    border: '1px solid #dee2e6', 
                    borderRadius: '8px', 
                    padding: '15px',
                    height: '400px',
                    overflowY: 'auto',
                    marginBottom: '15px'
                }}>
                    {messages.length === 0 ? (
                        <div style={{ 
                            color: '#6c757d', 
                            textAlign: 'center', 
                            padding: '50px 20px',
                            fontSize: '16px'
                        }}>
                            <strong>Ready to start chatting!</strong><br/>
                            {embeddingModelStatus === 'working' 
                                ? 'Ask questions about your documents and get AI-powered answers.'
                                : 'First, make sure you have a working embedding model.'
                            }
                        </div>
                    ) : (
                        messages.map((message, index) => (
                            <div key={index} style={{ marginBottom: '15px' }}>
                                <div style={{ 
                                    padding: '12px',
                                    backgroundColor: message.sender === 'user' ? '#007bff' : '#28a745',
                                    color: 'white',
                                    borderRadius: '8px',
                                    marginBottom: '8px'
                                }}>
                                    <strong>{message.sender === 'user' ? '👤 You' : `🤖 ${selectedModel}`}:</strong>
                                    <pre style={{ 
                                        whiteSpace: 'pre-wrap', 
                                        margin: '8px 0 0 0',
                                        fontFamily: 'inherit',
                                        color: 'white'
                                    }}>
                                        {message.text}
                                    </pre>
                                </div>
                                <Button 
                                    icon="pi pi-copy" 
                                    className="p-button-sm p-button-outlined" 
                                    style={{ fontSize: '0.7rem', padding: '0.3rem' }} 
                                    onClick={() => copyToClipboard(`${message.sender}: ${message.text}`)} 
                                    tooltip="Copy message"
                                />
                            </div>
                        ))
                    )}
                </div>

                {/* Input Area */}
                <div style={{ 
                    display: 'flex', 
                    alignItems: 'center', 
                    gap: '10px'
                }}>
                    <InputText 
                        style={{ flex: 1 }}
                        value={newMessage} 
                        onChange={(e) => setNewMessage(e.target.value)} 
                        onKeyPress={(e) => {
                            if (e.key === 'Enter' && !isSubmitting && selectedDB && embeddingModelStatus === 'working') {
                                handleSubmit();
                            }
                        }} 
                        placeholder={
                            !selectedDB 
                                ? "Select a database first"
                                : embeddingModelStatus === 'failed'
                                ? "Fix embedding model first"
                                : "Ask a question about your documents..."
                        }
                        disabled={!selectedDB || isSubmitting || embeddingModelStatus === 'failed'}
                    />
                    <Button 
                        label={isSubmitting ? "Thinking..." : "Ask"} 
                        onClick={handleSubmit} 
                        disabled={!selectedDB || isSubmitting || !newMessage.trim() || embeddingModelStatus === 'failed'}
                        style={{ minWidth: '80px' }}
                    />
                    {isSubmitting && (
                        <ProgressSpinner 
                            style={{ width: '2em', height: '2em' }} 
                            strokeWidth="6" 
                        />
                    )}
                </div>

                {/* Help Text */}
                <div style={{ 
                    marginTop: '15px',
                    padding: '10px',
                    backgroundColor: '#e9ecef',
                    borderRadius: '6px',
                    fontSize: '12px',
                    color: '#6c757d'
                }}>
                    💡 <strong>Tips:</strong> {
                        embeddingModelStatus === 'failed' 
                            ? 'Install an embedding model first: ollama pull mxbai-embed-large'
                            : 'Ask specific questions about your documents. The AI will search through your collection and provide relevant answers based on the content.'
                    }
                </div>
            </div>}
        </div>
    );
}

export default ChatFromDB;