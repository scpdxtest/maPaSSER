import React, { useEffect, useState } from 'react';
import { Dialog } from 'primereact/dialog';
import { Button } from 'primereact/button';
import { InputText } from 'primereact/inputtext';
import axios from 'axios';
import { Tooltip } from 'primereact/tooltip';
import { SplitButton } from 'primereact/splitbutton';
import './gpt.css';
import { ProgressSpinner } from 'primereact/progressspinner';

const Gpt = () => {
    const [messages, setMessages] = useState([]);
    const [newMessage, setNewMessage] = useState('');
    const [dialogVisible, setDialogVisible] = useState(true);
    const [selectedModel, setSelectedModel] = useState(localStorage.getItem("selectedLLMModel") || 'mistral');
    const [selectedOllama, setSelectedOllama] = useState(localStorage.getItem("selectedOllama"));
    const [stat, setStat] = useState('');
    const [buffer, setBuffer] = useState([]);
    const [isAnswering, setIsAnswering] = useState(false);

    useEffect(() => {
        setDialogVisible(true);
    }, []);

    const handleSubmit = async () => {
        setIsAnswering(true);
        const question = newMessage;
        setNewMessage('');
        setStat('');
        setMessages([...messages, { text: question, sender: 'user', stat: '' }, { text: '', sender: selectedModel, stat: '' }]);
    
        // Track the full response
        let fullModelResponse = '';
        let partialResponse = '';
    
        try {
            await axios.post(selectedOllama + '/api/generate', {"model": selectedModel, "prompt": question, "stream": true, "context": buffer }, {
                onDownloadProgress: (progressEvent) => {
                    const newChunk = progressEvent.event.target.responseText;
                    partialResponse += newChunk;
                    
                    // Process all complete JSON objects in the buffer
                    let processedLength = 0;
                    let jsonObjects = [];
                    
                    // Extract all complete JSON objects
                    let pos = 0;
                    while (pos < partialResponse.length) {
                        const openBracketPos = partialResponse.indexOf('{', pos);
                        if (openBracketPos === -1) break;
                        
                        // Find the matching closing bracket
                        let depth = 1;
                        let closeBracketPos = openBracketPos + 1;
                        
                        while (depth > 0 && closeBracketPos < partialResponse.length) {
                            if (partialResponse[closeBracketPos] === '{') {
                                depth++;
                            } else if (partialResponse[closeBracketPos] === '}') {
                                depth--;
                            }
                            closeBracketPos++;
                        }
                        
                        // If we found a complete JSON object
                        if (depth === 0) {
                            const jsonStr = partialResponse.substring(openBracketPos, closeBracketPos);
                            
                            try {
                                const jsonObj = JSON.parse(jsonStr);
                                jsonObjects.push(jsonObj);
                                processedLength = closeBracketPos;
                                pos = closeBracketPos;
                            } catch (error) {
                                // If parsing fails, move past this opening bracket
                                pos = openBracketPos + 1;
                            }
                        } else {
                            // Incomplete JSON, wait for more data
                            break;
                        }
                    }
                    
                    // Process all extracted JSON objects
                    jsonObjects.forEach(responseObject => {
                        if (responseObject.response) {
                            // Only append the new content
                            const newContent = responseObject.response;
                            if (!fullModelResponse.includes(newContent)) {
                                fullModelResponse += newContent;
                                
                                // Update the UI with the complete response so far
                                setMessages(prevMessages => {
                                    let newMessages = [...prevMessages];
                                    if (newMessages.length > 0 && newMessages[newMessages.length - 1].sender === selectedModel) {
                                        newMessages[newMessages.length - 1].text = fullModelResponse;
                                    } else {
                                        newMessages.push({ text: fullModelResponse, sender: selectedModel, stat: '' });
                                    }
                                    // console.log("newMessages", newMessages);
                                    return newMessages;
                                });
                            }
                        } else if (responseObject.context) {
                            const tokensPerSec = (Number(responseObject.eval_count) / Number(responseObject.eval_duration) * 1e9);
                            setStat("\nTokens per sec: " + tokensPerSec.toString());
                            setBuffer(prevBuffer => [...prevBuffer, ...responseObject.context]);
                            setMessages(prevMessages => {
                                let newMessages = [...prevMessages];
                                if (newMessages.length > 0) {
                                    newMessages[newMessages.length - 1].stat = "\nTokens per sec: " + tokensPerSec.toString();
                                }
                                return newMessages;
                            });
                        }
                    });
                    
                    // Keep only unprocessed part in the buffer
                    if (processedLength > 0) {
                        partialResponse = partialResponse.substring(processedLength);
                    }
                }
            })
        } catch (error) {
            console.error('Error while sending question to Ollama:', error);
        }
        setIsAnswering(false);
    };
    
    const copyToClipboard = (textToCopy) => {
        navigator.clipboard.writeText(textToCopy);
    };

    const handleNew = () => {
        setBuffer([]);
    };

    const saveContext = () => {
        const filename = window.prompt("Enter filename");
        if (filename) {
            const bufferString = buffer.join(',');
            const blob = new Blob([bufferString], {type: "text/plain;charset=utf-8"});
            const href = URL.createObjectURL(blob);
            const link = document.createElement('a');
            link.href = href;
            link.download = filename + ".context";
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
        }
    };

    const loadContext = () => {
        const input = document.createElement('input');
        input.type = 'file';
        input.accept = '.context';
        input.onchange = (event) => {
            const file = event.target.files[0];
            if (file) {
                const reader = new FileReader();
                reader.onload = function(e) {
                    const data = e.target.result.split(',').map(Number);
                    setBuffer(data);
                };
                reader.readAsText(file);
            }
        };
        input.click();
    };

    const items = [
        {label: 'Load', icon: 'pi pi-fw pi-file-import', command: () => {console.log("Load"); loadContext(); console.log("buffer", buffer);}},
        {label: 'Clear', icon: 'pi pi-fw pi-trash', command: () => {handleNew()}},
        {label: 'Save', icon: 'pi pi-fw pi-save', command: () => {console.log("Save"); saveContext()}}
    ];

    return (
        <Dialog header="Chat with private LLM" visible={dialogVisible} style={{ width: '98%' }} modal={true} onHide={() => {setDialogVisible(false)}}>
            {messages.map((message, index) => (
                <div key={index}>
                    <pre style={{ textOverflow: 'ellipsis', whiteSpace: 'pre-wrap', wordWrap: 'break-word' }}>
                        <b>{message.sender}:</b> {message.text}
                        {message.stat ? <span style={{ color: 'blue' }}>{message.stat}</span> : null}
                    </pre>
                    <Button icon="pi pi-copy" 
                            className="p-button-sm p-button-success p-button-outlined" 
                            style={{ fontSize: '0.6rem', padding: '0.2rem', top: '-0.9rem'}} 
                            onClick={() => copyToClipboard(`${message.sender}: ${message.text}`)} />                
                </div>
            ))}
            <div className="p-d-flex p-ai-center p-mt-2">
                <InputText 
                    style={{ width: '75%' }}
                    value={newMessage} 
                    onChange={(e) => setNewMessage(e.target.value)} 
                    onKeyPress={(e) => {
                        if (e.key === 'Enter') {
                            handleSubmit();
                        }
                    }} 
                    placeholder="Enter your question" 
                />
                <Button label="Ask" onClick={handleSubmit} className="p-ml-2" />
                <SplitButton id='contextButton' style={{marginLeft: '20px', backgroundColor: '#FF6961'}} label="Context" icon="pi pi-ellipsis-v" model={items} className="split-button" />
                {buffer.length > 0 && <i className="pi pi-slack" style={{fontSize: '2em', color: 'green', marginLeft: '10px'}} />}
                <Tooltip target="#contextButton" content="Load previously stored context, Clear context and start new chat, Save current context" />
                {isAnswering ? <ProgressSpinner style={{ width: '50px', height: '50px', marginLeft: '10px', width: '2em', height: '2em'}} strokeWidth="8"/> : null}      
            </div>
        </Dialog>
    );
};

export default Gpt;