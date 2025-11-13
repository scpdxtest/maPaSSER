// Multi-agent version of RAG Q&A chat with LLM

import React, { useState, useEffect, memo } from 'react';
import { DataTable } from 'primereact/datatable';
import { Column } from 'primereact/column';
import { Ollama } from '@langchain/ollama';
import { OllamaEmbeddings } from '@langchain/ollama';
import { Chroma } from "@langchain/community/vectorstores/chroma";
import { ChromaClient } from "chromadb";
import { Button } from 'primereact/button';
import { InputText } from 'primereact/inputtext';
import { Card } from 'primereact/card';
import './chatFromDB.css';
import { ProgressSpinner } from 'primereact/progressspinner';
import { BufferMemory } from "langchain/memory";
import { FilterMatchMode } from 'primereact/api';
import { ScoreThresholdRetriever } from "langchain/retrievers/score_threshold";
import { tool } from "@langchain/core/tools";
import { z } from "zod";
import { AgentExecutor, createToolCallingAgent, createReactAgent } from "langchain/agents";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { RunnableAgent } from "@langchain/core/runnables";
import { RunnableSequence } from "@langchain/core/runnables";
import { convertToOpenAIFunction } from "@langchain/core/utils/function_calling";

const ChatFromDBMultiAgent = () => {
    const [selectedChromaDB, setSelectedChromaDB] = useState('');
    const [selectedOllama, setSelectedOllama] = useState(null);
    const [selectedModel, setSelectedModel] = useState(null);
    const [collections, setCollections] = useState([]);
    const [dialogVisible, setDialogVisible] = useState(true);
    const [messages, setMessages] = useState([]);
    const [newMessage, setNewMessage] = useState('');
    const [selectedDB, setSelectedDB] = useState('');
    const [isSubmitting, setIsSubmitting] = useState(false);
    const [tempreture, setTempreture] = useState(localStorage.getItem("chatTempreture") || '0.2');
    const [symScore, setSymScore] = useState(0);
    const [k, setK] = useState(0);
    const [kInc, setKInc] = useState(0);

    const [filters, setFilters] = useState({
        global: { value: null, matchMode: FilterMatchMode.CONTAINS },
        name: { value: null, matchMode: FilterMatchMode.STARTS_WITH },
    });

    const embeddings_open = new OllamaEmbeddings({
        model: selectedModel, 
        baseUrl: selectedOllama 
    });

    const mdl = new Ollama({
        baseUrl: selectedOllama, 
        model: selectedModel, 
        requestOptions: {
            num_gpu: 1,
            tempreture: localStorage.getItem("chatTempreture") || '0.2',
            format: 'json',
        }
    });  

    const [memory, setMemory] = useState(new BufferMemory({
        memoryKey: "chat_history",
        returnMessages: true,
    }));

    const listCollections = async (ch) => {
        const client = new ChromaClient({ path: ch });

        let collection2 = await client.listCollections();

        const collectionsWithIdAndName = collection2.map((collection) => ({
            name: collection,
            id: '1',
        }));
        setCollections(collectionsWithIdAndName);
    };

    useEffect(() => {
        setSymScore(Number(localStorage.getItem("symScore")) || 0.9);
        setK(Number(localStorage.getItem("k")) || 100);
        setKInc(Number(localStorage.getItem("kInc")) || 2);
        const ch = localStorage.getItem("selectedChromaDB") || 'http://127.0.0.1:8000';
        setSelectedChromaDB(ch);
        const ol = localStorage.getItem("selectedOllama") || 'http://127.0.0.1:11434';
        setSelectedOllama(ol);
        const mdl = localStorage.getItem("selectedLLMModel") || 'mistral';
        setSelectedModel(mdl);
        listCollections(ch);
    }, []);

    const onRowDoubleClick = async (e) => {
        const collection = e.data;

        const client = new ChromaClient({ path: selectedChromaDB });
        let collection2 = await client.getCollection({ name: e.data.name });
        let docCount = await collection2.count();
        const list = await collection2.peek(10);

        const vectorStore1 = await Chroma.fromExistingCollection(
            embeddings_open,
            {
                collectionName: e.data.name,
                url: selectedChromaDB,
            });

        var retriever;
        if (localStorage.getItem("retriever") === '0') {
            retriever = vectorStore1.asRetriever();
        } else {
            retriever = ScoreThresholdRetriever.fromVectorStore(vectorStore1, {
                minSimilarityScore: 0.9, // Finds results with at least this similarity score
                maxK: 100, // The maximum K value to use. Use it based to your chunk size to make sure you don't run out of tokens
                kIncrement: 2, // How much to increase K by each time. It'll fetch N results, then N + kIncrement, then N + kIncrement * 2, etc.
            });
        }    
        console.log(`Double clicked on collection with ID: ${collection.id}`);
    };

    const rowClass = (rowData) => {
        return {
            'p-highlight': rowData.name === selectedDB,
        };
    };

    const copyToClipboard = (textToCopy) => {
        navigator.clipboard.writeText(textToCopy);
    };

    const onRowSelect = async (e) => {
        setSelectedDB(e.data.name);
        setMemory(new BufferMemory({
            memoryKey: "chat_history",
            returnMessages: true,
        }));    
    };

    // Define a simple function tool that Ollama can understand
    const retrieve_documents = tool(
        async ({ input }) => {
            // Simple calculation for testing
            console.log("Input------->:", input);
            return { output: `The result is ${parseInt(input) + 2}` };
        },
        {
            name: "retrieveDocuments",
            description: "Retrieve documents from the selected DB",
            schema: z.object({
                input: z.string().describe("The input query"),
            }),
        }
    );
    
    // Use a simpler prompt for testing
    const prompt = ChatPromptTemplate.fromMessages([
        ["system", "You are a helpful assistant that can use tools to answer questions."],
        ["human", "{input}"],
        ["placeholder", "{agent_scratchpad}"],
    ]);
    const tools = [retrieve_documents];
    
    // Format the tools in a way that's easier for the model to understand
    const llmWithTools = mdl.bind({
        tools: tools.map((tool) => convertToOpenAIFunction(tool)),
    });

    // Define a simple prompt
    const prompt1 = ChatPromptTemplate.fromMessages([
        ["system", "You are a helpful assistant. Use the provided tools to answer questions."],
        ["human", "{input}"],
        ["placeholder", "{agent_scratchpad}"],
    ]);

    // Create a custom agent using a simpler approach
    const customAgent = RunnableSequence.from([
        {
            input: (input) => input.input,
            agent_scratchpad: (input) => input.agent_scratchpad || "",
        },
        prompt1,
        llmWithTools,
        (output) => ({ output: output.content })
    ]);

    // Create a custom agent using a simpler approach that handles different response formats
    const customAgent1 = RunnableSequence.from([
        {
            input: (input) => input.input,
            agent_scratchpad: (input) => input.agent_scratchpad || "",
        },
        prompt1,
        llmWithTools,
        (output) => {
            // Log what we're actually getting from the model
            console.log("Raw model output:", output);
            // Handle different possible response formats
            if (output && typeof output === 'object') {
                // If it has content property, use it
                if (output.content !== undefined) {
                    return { output: output.content };
                }
                // If it's a direct string response
                if (typeof output === 'string') {
                    return { output: output };
                }
                // If it's JSON with a text or response property
                if (output.text) return { output: output.text };
                if (output.response) return { output: output.response };
                
                // If we can stringify it, return that
                try {
                    return { output: JSON.stringify(output) };
                } catch (e) {
                    // Fallback
                    return { output: "Unable to parse model response" };
                }
            } else {
                return { output: output };
            }
            // Fallback for any other case
            return { output: "No valid response from the model" };
        }
    ]);

// Much simpler direct approach
    const handleSubmit = async () => {
        setIsSubmitting(true);

        const question = newMessage;
        setNewMessage('');
        setMessages([...messages, { text: question, sender: 'user', stat: '' }, { text: '', sender: selectedModel, stat: '' }]);

        try {
            let response;
            
            // Check if question is explicitly asking to use tool
            if (question.toLowerCase().includes("retrievedocuments")) {
                const match = question.match(/\((\d+)\)/);
                if (match) {
                    const input = match[1];
                    const result = await retrieve_documents.invoke({ input });
                    response = { output: result.output };
                } else {
                    // Direct call to the model without custom agent
                    const resp = await customAgent1.invoke({ input: question });
                    console.log("1. Response:", resp);
                    const modelResponse = await mdl.invoke(question);
                    response = { output: modelResponse.content || modelResponse };
                }
            } else {
                // Direct call to the model without custom agent
                const resp = await customAgent1.invoke({ input: question });
                console.log("2. Response----->:", resp);
                const modelResponse = await mdl.invoke(question);
                response = { output: modelResponse.content || modelResponse };
            }
            
            const resText = response.output;
            console.log("Answer:", resText);
            
            setMessages(prevMessages => {
                let newMessages = [...prevMessages];
                if (newMessages.length > 0) {
                    newMessages[newMessages.length - 1].text = resText;
                }
                return newMessages;
            });
        } catch (error) {
            console.error("Error:", error);
            setMessages(prevMessages => {
                let newMessages = [...prevMessages];
                if (newMessages.length >= 0 && newMessages[newMessages.length - 1].sender === selectedModel) {
                    newMessages[newMessages.length - 1].text = `Error: ${error.message}`;
                }
                return newMessages;
            });
        } finally {
            setIsSubmitting(false);
        }
    };
    
    const retrieverColor = localStorage.getItem("retriever") === "Normal" ? 'green' : 'red';

    return (
        <div style={{ width: '100%', display: 'flex', justifyContent: 'space-between'}}>
            <div style={{ width: '40%'}}>
                <h3>
                    <span>SelectDB (temperature={tempreture})</span>
                </h3>
                <DataTable style={{width: '90%'}} value={collections} onRowDoubleClick={onRowDoubleClick}
                    onRowSelect={onRowSelect} selectionMode="single" selection={selectedDB}
                    rowClassName={rowClass} filters={filters} filterDisplay="row"
                    size={"small"} showGridlines stripedRows>
                    <Column field="name" header="Name" filter filterPlaceholder="By name"></Column>
                    <Column field="id" header="ID"></Column>
                </DataTable>
            </div>

        {dialogVisible &&
            <div style={{ width: '90%', marginLeft: '20px'}}>
                <Card title='Chat with private LLM over selected DB' style={{ width: '95%' }}>
                <span style={{ fontSize: '1.2em', 
                                fontWeight: 'bold', 
                                color: retrieverColor, 
                                marginBottom: '10px' }}>  
                    {localStorage.getItem("retriever")}
                    {localStorage.getItem("retriever") === "Score" ? (
                        ` -> Similarity score = ${localStorage.getItem("symScore") || '0.9'} /
                        k = ${localStorage.getItem("k") || '100'} / 
                        kInc = ${localStorage.getItem("kInc") || '2' }`
                    ) : null}                
                </span>
                    {messages.map((message, index) => (
                        <div key={index}>
                            <pre style={{ textOverflow: 'ellipsis', whiteSpace: 'pre-wrap', wordWrap: 'break-word', textAlign: 'left' }}>
                                <b>{message.sender}:</b> {message.text}
                                {message.stat ? <span style={{ color: 'blue' }}>{message.stat}</span> : null}
                            </pre>
                            <Button icon="pi pi-copy" 
                                    className="p-button-sm p-button-success p-button-outlined" 
                                    style={{ fontSize: '0.6rem', padding: '0.2rem', top: '-0.9rem'}} 
                                    onClick={() => copyToClipboard(`${message.sender}: ${message.text}`)} />                
                        </div>
                    ))}
                    <div className="p-d-flex p-ai-center p-mt-2" style={{ flexDirection: 'row', alignItems: 'center' }}>
                        <InputText 
                            style={{ width: '80%' }}
                            value={newMessage} 
                            onChange={(e) => setNewMessage(e.target.value)} 
                            onKeyPress={(e) => {
                                if (e.key === 'Enter') {
                                    handleSubmit();
                                }
                            }} 
                            placeholder="Enter your question" 
                        />
                        <Button label="Ask" style={{marginLeft: '10px'}} onClick={handleSubmit} className="p-ml-2" />
                        {isSubmitting && <ProgressSpinner style={{ marginLeft: '5px', width: '2em', height: '2em' }} strokeWidth="8" />}
                    </div>
                </Card>
            </div>
        }

        </div>

    );
}

export default ChatFromDBMultiAgent;