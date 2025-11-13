import React, { useEffect, useState, useRef, useCallback } from 'react';
import { Dialog } from 'primereact/dialog';
import { Button } from 'primereact/button';
import { InputText } from 'primereact/inputtext';
import { InputTextarea } from 'primereact/inputtextarea';
import { Dropdown } from 'primereact/dropdown';
import { Card } from 'primereact/card';
import { ScrollPanel } from 'primereact/scrollpanel';
import { ProgressBar } from 'primereact/progressbar';
import { Toast } from 'primereact/toast';
import { Divider } from 'primereact/divider';
import { Badge } from 'primereact/badge';
import { Chip } from 'primereact/chip';
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
    const [selectedOllama, setSelectedOllama] = useState(localStorage.getItem("selectedOllama") || 'http://localhost:11434');
    const [stat, setStat] = useState('');
    const [buffer, setBuffer] = useState([]);
    const [isAnswering, setIsAnswering] = useState(false);
    const [connectionQuality, setConnectionQuality] = useState('good');
    const [responseProgress, setResponseProgress] = useState(0);
    const [currentResponseId, setCurrentResponseId] = useState(null);
    const [retryCount, setRetryCount] = useState(0);
    const [availableModels, setAvailableModels] = useState([]);
    const [streamingSpeed, setStreamingSpeed] = useState(0);
    const [isConnected, setIsConnected] = useState(true);
    const [inputMode, setInputMode] = useState('single'); // 'single' or 'multi'
    const [autoScroll, setAutoScroll] = useState(true); // Control auto-scrolling
    const [userScrolledUp, setUserScrolledUp] = useState(false); // Track if user manually scrolled up
    
    const messagesEndRef = useRef(null);
    const scrollPanelRef = useRef(null);
    const abortControllerRef = useRef(null);
    const toast = useRef(null);
    const responseStartTime = useRef(null);
    const lastUpdateTime = useRef(null);
    const responseBuffer = useRef('');
    const connectionTimeoutRef = useRef(null);
    const updateThrottleRef = useRef(null);
    const scrollTimeoutRef = useRef(null);

    // FIXED: Completely prevent auto-scroll during streaming
    useEffect(() => {
        // NEVER auto-scroll while streaming is active
        if (autoScroll && !userScrolledUp && !isAnswering) {
            clearTimeout(scrollTimeoutRef.current);
            scrollTimeoutRef.current = setTimeout(() => {
                messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
            }, 100);
        }
    }, [messages, autoScroll, userScrolledUp, isAnswering]);

    // FIXED: Better scroll detection that completely stops during streaming
    const handleScrollChange = useCallback(() => {
        // COMPLETELY disable scroll detection while answering
        if (scrollPanelRef.current && !isAnswering) {
            const scrollElement = scrollPanelRef.current.getElement();
            const { scrollTop, scrollHeight, clientHeight } = scrollElement;
            const isNearBottom = scrollHeight - scrollTop - clientHeight < 100;
            
            setUserScrolledUp(!isNearBottom);
            setAutoScroll(isNearBottom);
        }
    }, [isAnswering]);

    // Initialize and check connection
    useEffect(() => {
        setDialogVisible(true);
        checkConnection();
        fetchAvailableModels();
    }, [selectedOllama]);

    // Connection quality monitoring
    const checkConnection = useCallback(async () => {
        try {
            const startTime = Date.now();
            const response = await axios.get(selectedOllama + '/api/version', { 
                timeout: 5000,
                signal: AbortSignal.timeout(5000)
            });
            const latency = Date.now() - startTime;
            
            if (response.status === 200) {
                setIsConnected(true);
                if (latency < 500) {
                    setConnectionQuality('excellent');
                } else if (latency < 1000) {
                    setConnectionQuality('good');
                } else if (latency < 2000) {
                    setConnectionQuality('fair');
                } else {
                    setConnectionQuality('poor');
                }
            }
        } catch (error) {
            setIsConnected(false);
            setConnectionQuality('offline');
            console.error('Connection check failed:', error);
        }
    }, [selectedOllama]);

    // Fetch available models
    const fetchAvailableModels = useCallback(async () => {
        try {
            const response = await axios.get(selectedOllama + '/api/tags', { timeout: 10000 });
            if (response.data && response.data.models) {
                const models = response.data.models.map(model => ({
                    label: model.name,
                    value: model.name,
                    size: model.size
                }));
                setAvailableModels(models);
            }
        } catch (error) {
            console.error('Failed to fetch models:', error);
            toast.current?.show({
                severity: 'warn',
                summary: 'Warning',
                detail: 'Could not fetch available models',
                life: 3000
            });
        }
    }, [selectedOllama]);

    // IMPROVED: Stream processing with no auto-scroll interference
    const streamResponse = async (question, responseId) => {
        const controller = new AbortController();
        abortControllerRef.current = controller;
        
        try {
            const response = await fetch(selectedOllama + '/api/generate', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    model: selectedModel,
                    prompt: question,
                    stream: true,
                    context: buffer,
                    options: {
                        temperature: 0.7,
                        top_p: 0.9,
                        top_k: 40
                    }
                }),
                signal: controller.signal
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let streamBuffer = '';
            let tokenCount = 0;
            let lastUIUpdate = 0;
            responseStartTime.current = Date.now();

            while (true) {
                const { done, value } = await reader.read();
                
                if (done) break;
                
                streamBuffer += decoder.decode(value, { stream: true });
                const lines = streamBuffer.split('\n');
                streamBuffer = lines.pop() || '';
                
                for (const line of lines) {
                    if (line.trim()) {
                        try {
                            const jsonData = JSON.parse(line);
                            
                            if (jsonData.response) {
                                responseBuffer.current += jsonData.response;
                                tokenCount++;
                                
                                // Update streaming speed
                                const elapsed = (Date.now() - responseStartTime.current) / 1000;
                                const tokensPerSec = tokenCount / elapsed;
                                setStreamingSpeed(tokensPerSec);
                                setStat(`\n⚡ ${tokensPerSec.toFixed(1)} tokens/sec`);
                                
                                // FIXED: More aggressive throttling for UI updates
                                const now = Date.now();
                                if (now - lastUIUpdate > 300) { // Increased from 200ms to 300ms
                                    setMessages(prevMessages => {
                                        const newMessages = [...prevMessages];
                                        const lastMessage = newMessages[newMessages.length - 1];
                                        if (lastMessage && lastMessage.id === responseId) {
                                            lastMessage.text = responseBuffer.current;
                                            lastMessage.stat = `⚡ ${tokensPerSec.toFixed(1)} tokens/sec`;
                                        }
                                        return newMessages;
                                    });
                                    
                                    // Update progress with better calculation
                                    const estimatedProgress = Math.min(90, Math.floor((responseBuffer.current.length / 10)));
                                    setResponseProgress(estimatedProgress);
                                    
                                    lastUIUpdate = now;
                                }
                                
                                lastUpdateTime.current = Date.now();
                            }
                            
                            if (jsonData.context) {
                                setBuffer(jsonData.context);
                            }
                            
                            if (jsonData.done) {
                                // Final update when done
                                setIsAnswering(false);
                                setResponseProgress(100);
                                setCurrentResponseId(null);
                                
                                // Final stats
                                const totalTime = (Date.now() - responseStartTime.current) / 1000;
                                const finalTokensPerSec = tokenCount / totalTime;
                                
                                setMessages(prevMessages => {
                                    const newMessages = [...prevMessages];
                                    const lastMessage = newMessages[newMessages.length - 1];
                                    if (lastMessage && lastMessage.id === responseId) {
                                        lastMessage.text = responseBuffer.current; // Final complete text
                                        lastMessage.stat = `✅ Complete - ${finalTokensPerSec.toFixed(1)} tok/s - ${totalTime.toFixed(1)}s`;
                                    }
                                    return newMessages;
                                });
                                
                                // FIXED: Only auto-scroll after streaming is completely done
                                setTimeout(() => {
                                    if (autoScroll && !userScrolledUp) {
                                        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
                                    }
                                }, 100);
                                
                                if (connectionQuality === 'poor' || connectionQuality === 'fair') {
                                    toast.current?.show({
                                        severity: 'success',
                                        summary: 'Response Complete',
                                        detail: `Generated in ${totalTime.toFixed(1)}s`,
                                        life: 2000
                                    });
                                }
                                break;
                            }
                        } catch (parseError) {
                            console.warn('Failed to parse JSON line:', line, parseError);
                        }
                    }
                }
                
                // Check for stalled connection
                if (Date.now() - lastUpdateTime.current > 15000) {
                    console.warn('Stream appears stalled');
                    toast.current?.show({
                        severity: 'warn',
                        summary: 'Slow Response',
                        detail: 'Response is taking longer than usual...',
                        life: 3000
                    });
                }
            }
            
        } catch (error) {
            throw error;
        }
    };

    // FIXED: Enhanced submit with NO auto-scroll during streaming
    const handleSubmit = async (retryAttempt = 0) => {
        if (!newMessage.trim() || isAnswering) return;
        
        if (!isConnected) {
            toast.current?.show({
                severity: 'error',
                summary: 'Connection Error',
                detail: 'Not connected to Ollama server',
                life: 3000
            });
            return;
        }
    
        setIsAnswering(true);
        setRetryCount(retryAttempt);
        setResponseProgress(0);
        // FIXED: Don't force scroll state changes at start
        
        const question = newMessage.trim();
        const now = Date.now();
        const responseId = now.toString();
        const userMessageId = (now - 1).toString(); // FIXED: Ensure unique IDs
        setCurrentResponseId(responseId);
        setNewMessage('');
        setStat('');
        responseBuffer.current = '';
        lastUpdateTime.current = Date.now();
    
        // Add user message and placeholder for model response
        setMessages(prev => [
            ...prev,
            { 
                id: userMessageId, // FIXED: Use the unique ID
                text: question, 
                sender: 'user', 
                stat: '',
                timestamp: new Date().toLocaleTimeString()
            },
            { 
                id: responseId,
                text: '⌛ Generating response...', 
                sender: selectedModel, 
                stat: '',
                timestamp: new Date().toLocaleTimeString()
            }
        ]);
    
        // FIXED: Only scroll to new message if user was already at bottom
        if (autoScroll && !userScrolledUp) {
            setTimeout(() => {
                messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
            }, 100);
        }
    
        try {
            await streamResponse(question, responseId);
            
        } catch (error) {
            console.error('Error during streaming:', error);
            setIsAnswering(false);
            setResponseProgress(0);
            
            if (error.name === 'AbortError') {
                setMessages(prev => {
                    const newMessages = [...prev];
                    const lastMessage = newMessages[newMessages.length - 1];
                    if (lastMessage && lastMessage.id === responseId) {
                        lastMessage.text = '❌ Request was cancelled';
                        lastMessage.stat = '⚠️ Cancelled by user';
                    }
                    return newMessages;
                });
                
                toast.current?.show({
                    severity: 'info',
                    summary: 'Request Cancelled',
                    detail: 'Request was cancelled by user',
                    life: 3000
                });
                return;
            }
    
            // Retry logic for poor connections
            if (retryAttempt < 2 && (connectionQuality === 'poor' || connectionQuality === 'fair')) {
                toast.current?.show({
                    severity: 'warn',
                    summary: 'Retrying...',
                    detail: `Attempt ${retryAttempt + 2} of 3`,
                    life: 2000
                });
                
                // Remove the failed message before retrying
                setMessages(prev => prev.slice(0, -1));
                
                setTimeout(() => {
                    setNewMessage(question);
                    handleSubmit(retryAttempt + 1);
                }, 2000 * (retryAttempt + 1));
                return;
            }
    
            // Show error message
            const errorMessage = error.message || 'Unknown error occurred';
            
            setMessages(prev => {
                const newMessages = [...prev];
                const lastMessage = newMessages[newMessages.length - 1];
                if (lastMessage && lastMessage.id === responseId) {
                    lastMessage.text = `❌ Error: ${errorMessage}`;
                    lastMessage.stat = `⚠️ ${retryAttempt + 1} attempt(s) failed`;
                }
                return newMessages;
            });
    
            toast.current?.show({
                severity: 'error',
                summary: 'Request Failed',
                detail: errorMessage,
                life: 5000
            });
        }
    };

    // Cancel current request
    const handleCancel = () => {
        if (abortControllerRef.current) {
            abortControllerRef.current.abort();
            setIsAnswering(false);
            setResponseProgress(0);
            setCurrentResponseId(null);
        }
    };

    // IMPROVED: Manual scroll to bottom
    const scrollToBottom = () => {
        setAutoScroll(true);
        setUserScrolledUp(false);
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    };

    // Replace the copyToClipboard function (around line 364)
    const copyToClipboard = (textToCopy, messageId) => {
        // Modern clipboard API with fallback
        if (navigator.clipboard && navigator.clipboard.writeText) {
            // Modern approach
            navigator.clipboard.writeText(textToCopy).then(() => {
                toast.current?.show({
                    severity: 'success',
                    summary: 'Copied',
                    detail: 'Message copied to clipboard',
                    life: 1500
                });
            }).catch((error) => {
                console.error('Clipboard API failed:', error);
                // Fallback to legacy method
                fallbackCopyToClipboard(textToCopy);
            });
        } else {
            // Fallback for older browsers or insecure contexts
            fallbackCopyToClipboard(textToCopy);
        }
    };
    
    // Add fallback copy function
    const fallbackCopyToClipboard = (text) => {
        try {
            // Create a temporary textarea element
            const textArea = document.createElement('textarea');
            textArea.value = text;
            
            // Make it invisible
            textArea.style.position = 'fixed';
            textArea.style.left = '-999999px';
            textArea.style.top = '-999999px';
            
            // Add to DOM
            document.body.appendChild(textArea);
            
            // Select and copy
            textArea.focus();
            textArea.select();
            
            // Try to copy using execCommand (legacy method)
            const successful = document.execCommand('copy');
            
            // Remove from DOM
            document.body.removeChild(textArea);
            
            if (successful) {
                toast.current?.show({
                    severity: 'success',
                    summary: 'Copied',
                    detail: 'Message copied to clipboard',
                    life: 1500
                });
            } else {
                throw new Error('execCommand copy failed');
            }
        } catch (error) {
            console.error('Fallback copy failed:', error);
            toast.current?.show({
                severity: 'error',
                summary: 'Copy Failed',
                detail: 'Could not copy to clipboard. Please copy manually.',
                life: 3000
            });
        }
    };

    const handleNew = () => {
        setBuffer([]);
        setMessages([]);
        setAutoScroll(true);
        setUserScrolledUp(false);
        toast.current?.show({
            severity: 'info',
            summary: 'Chat Cleared',
            detail: 'New conversation started',
            life: 2000
        });
    };

    const saveContext = () => {
        const filename = window.prompt("Enter filename", `context-${new Date().toISOString().split('T')[0]}`);
        if (filename) {
            const contextData = {
                buffer: buffer,
                messages: messages,
                model: selectedModel,
                timestamp: new Date().toISOString()
            };
            
            const blob = new Blob([JSON.stringify(contextData, null, 2)], {type: "application/json"});
            const href = URL.createObjectURL(blob);
            const link = document.createElement('a');
            link.href = href;
            link.download = filename + ".context.json";
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
            
            toast.current?.show({
                severity: 'success',
                summary: 'Context Saved',
                detail: `Saved as ${filename}.context.json`,
                life: 3000
            });
        }
    };

    const loadContext = () => {
        const input = document.createElement('input');
        input.type = 'file';
        input.accept = '.context.json,.context';
        input.onchange = (event) => {
            const file = event.target.files[0];
            if (file) {
                const reader = new FileReader();
                reader.onload = function(e) {
                    try {
                        const data = JSON.parse(e.target.result);
                        if (data.buffer) {
                            setBuffer(data.buffer);
                            if (data.messages) setMessages(data.messages);
                            if (data.model) setSelectedModel(data.model);
                            
                            toast.current?.show({
                                severity: 'success',
                                summary: 'Context Loaded',
                                detail: `Loaded context from ${file.name}`,
                                life: 3000
                            });
                        } else {
                            // Legacy format
                            const bufferData = e.target.result.split(',').map(Number);
                            setBuffer(bufferData);
                        }
                    } catch (error) {
                        toast.current?.show({
                            severity: 'error',
                            summary: 'Load Failed',
                            detail: 'Invalid context file format',
                            life: 3000
                        });
                    }
                };
                reader.readAsText(file);
            }
        };
        input.click();
    };

    // Get connection status color and icon
    const getConnectionStatus = () => {
        const statusMap = {
            'excellent': { color: '#4CAF50', icon: 'pi-wifi', label: 'Excellent' },
            'good': { color: '#8BC34A', icon: 'pi-wifi', label: 'Good' },
            'fair': { color: '#FF9800', icon: 'pi-wifi', label: 'Fair' },
            'poor': { color: '#F44336', icon: 'pi-wifi', label: 'Poor' },
            'offline': { color: '#9E9E9E', icon: 'pi-times', label: 'Offline' }
        };
        return statusMap[connectionQuality] || statusMap.offline;
    };

    const connectionStatus = getConnectionStatus();

    const contextItems = [
        {label: 'Load Context', icon: 'pi pi-fw pi-file-import', command: loadContext},
        {label: 'Clear Chat', icon: 'pi pi-fw pi-trash', command: handleNew},
        {label: 'Save Context', icon: 'pi pi-fw pi-save', command: saveContext},
        {separator: true},
        {label: 'Refresh Models', icon: 'pi pi-fw pi-refresh', command: fetchAvailableModels},
        {label: 'Check Connection', icon: 'pi pi-fw pi-wifi', command: checkConnection}
    ];

    return (
        <>
            <Toast ref={toast} position="top-right" />
            <Dialog 
                header={
                    <div className="chat-header">
                        <div className="header-title">
                            <i className="pi pi-comments" style={{marginRight: '8px'}}></i>
                            Chat with Private LLM
                        </div>
                        <div className="header-status">
                            <Chip 
                                label={connectionStatus.label} 
                                icon={`pi ${connectionStatus.icon}`}
                                className={`connection-chip ${connectionQuality}`}
                                style={{backgroundColor: connectionStatus.color + '20', color: connectionStatus.color}}
                            />
                            {streamingSpeed > 0 && (
                                <Chip 
                                    label={`${streamingSpeed.toFixed(1)} tok/s`}
                                    icon="pi pi-gauge"
                                    className="speed-chip"
                                />
                            )}
                        </div>
                    </div>
                } 
                visible={dialogVisible} 
                style={{ width: '95%', maxWidth: '1200px', height: '85vh' }} 
                modal={true} 
                onHide={() => setDialogVisible(false)}
                maximizable
                className="chat-dialog"
            >
                <div className="chat-container">
                    {/* Model Selection and Settings */}
                    <div className="settings-panel">
                        <div className="settings-row">
                            <Dropdown
                                value={selectedModel}
                                options={availableModels}
                                onChange={(e) => {
                                    setSelectedModel(e.value);
                                    localStorage.setItem("selectedLLMModel", e.value);
                                }}
                                placeholder="Select Model"
                                className="model-dropdown"
                                showClear={false}
                                style={{minWidth: '200px'}}
                            />
                            
                            <Button
                                icon="pi pi-cog"
                                className="p-button-outlined p-button-sm"
                                onClick={() => setInputMode(inputMode === 'single' ? 'multi' : 'single')}
                                tooltip={inputMode === 'single' ? 'Switch to multi-line input' : 'Switch to single-line input'}
                            />
                            
                            <SplitButton 
                                label="Context" 
                                icon="pi pi-ellipsis-v" 
                                model={contextItems} 
                                className="context-button"
                                size="small"
                            />
                            
                            {buffer.length > 0 && (
                                <Badge 
                                    value={buffer.length} 
                                    className="context-badge"
                                    style={{marginLeft: '8px'}}
                                />
                            )}
                            
                            {/* Show scroll button when user scrolled up OR when streaming */}
                            {(userScrolledUp || isAnswering) && (
                                <Button
                                    icon="pi pi-arrow-down"
                                    className="p-button-outlined p-button-sm scroll-button animated"
                                    onClick={scrollToBottom}
                                    tooltip={isAnswering ? "Scroll to see latest response" : "New message available - scroll to bottom"}
                                />
                            )}
                        </div>
                    </div>

                    <Divider />

                    {/* Messages Area */}
                    <ScrollPanel 
                        ref={scrollPanelRef}
                        className="messages-container" 
                        style={{height: '50vh'}}
                        onScroll={handleScrollChange}
                    >
                        {messages.length === 0 && (
                            <div className="welcome-message">
                                <div className="welcome-content">
                                    <i className="pi pi-comments" style={{fontSize: '3em', color: '#ccc', marginBottom: '16px'}}></i>
                                    <h3>Welcome to Private LLM Chat</h3>
                                    <p>Start a conversation with your local language model</p>
                                    <div className="quick-actions">
                                        <Button 
                                            label="Load Previous Context" 
                                            icon="pi pi-folder-open" 
                                            className="p-button-outlined p-button-sm"
                                            onClick={loadContext}
                                        />
                                    </div>
                                </div>
                            </div>
                        )}
                        
                        {messages.map((message, index) => (
                            <Card key={`message-${message.id || index}-${message.sender}`} className={`message-card ${message.sender === 'user' ? 'user-message' : 'model-message'}`}>
                                <div className="message-header">
                                    <div className="sender-info">
                                        <i className={`pi ${message.sender === 'user' ? 'pi-user' : 'pi-desktop'}`}></i>
                                        <strong>{message.sender === 'user' ? 'You' : message.sender}</strong>
                                        {message.timestamp && (
                                            <span className="timestamp">{message.timestamp}</span>
                                        )}
                                    </div>
                                    <div className="message-actions">
                                        <Button 
                                            icon="pi pi-copy" 
                                            className="p-button-text p-button-sm copy-button"
                                            onClick={() => copyToClipboard(message.text, message.id)}
                                            tooltip="Copy message"
                                        />
                                    </div>
                                </div>
                                
                                <div className="message-content">
                                    <pre className="message-text">{message.text}</pre>
                                    {message.stat && (
                                        <div className="message-stats">
                                            <small>{message.stat}</small>
                                        </div>
                                    )}
                                </div>
                                
                                {/* Better progress bar with visible text */}
                                {message.id === currentResponseId && isAnswering && (
                                    <div className="response-progress">
                                        <div className="progress-header">
                                            <span className="progress-label">Generating response</span>
                                            <span className="progress-percentage">{responseProgress}%</span>
                                        </div>
                                        <ProgressBar 
                                            value={responseProgress} 
                                            className="response-progress-bar enhanced"
                                            showValue={false}
                                        />
                                        <div className="progress-stats">
                                            {streamingSpeed > 0 && (
                                                <small>⚡ {streamingSpeed.toFixed(1)} tokens/sec</small>
                                            )}
                                        </div>
                                    </div>
                                )}
                            </Card>
                        ))}
                        <div ref={messagesEndRef} />
                    </ScrollPanel>

                    <Divider />

                    {/* Input Area */}
                    <div className="input-container">
                        {isAnswering && (
                            <div className="answering-indicator">
                                <ProgressSpinner size="small" strokeWidth="4" />
                                <span>Generating response... ({responseProgress}%)</span>
                                <Button 
                                    label="Cancel" 
                                    icon="pi pi-times" 
                                    className="p-button-danger p-button-sm"
                                    onClick={handleCancel}
                                />
                                {retryCount > 0 && (
                                    <Badge value={`Retry ${retryCount}`} severity="warning" />
                                )}
                            </div>
                        )}
                        
                        <div className="input-row">
                            {inputMode === 'single' ? (
                                <InputText 
                                    value={newMessage} 
                                    onChange={(e) => setNewMessage(e.target.value)} 
                                    onKeyPress={(e) => {
                                        if (e.key === 'Enter' && !e.shiftKey) {
                                            e.preventDefault();
                                            handleSubmit();
                                        }
                                    }} 
                                    placeholder="Type your message... (Enter to send, Shift+Enter for new line)"
                                    className="message-input"
                                    disabled={isAnswering || !isConnected}
                                />
                            ) : (
                                <InputTextarea 
                                    value={newMessage} 
                                    onChange={(e) => setNewMessage(e.target.value)} 
                                    onKeyPress={(e) => {
                                        if (e.key === 'Enter' && !e.shiftKey) {
                                            e.preventDefault();
                                            handleSubmit();
                                        }
                                    }} 
                                    placeholder="Type your message... (Enter to send, Shift+Enter for new line)"
                                    className="message-input-textarea"
                                    rows={3}
                                    disabled={isAnswering || !isConnected}
                                />
                            )}
                            
                            <Button 
                                label="Send" 
                                icon="pi pi-send" 
                                onClick={handleSubmit} 
                                disabled={!newMessage.trim() || isAnswering || !isConnected}
                                className="send-button"
                            />
                        </div>
                        
                        {connectionQuality === 'poor' && (
                            <div className="connection-warning">
                                <i className="pi pi-exclamation-triangle"></i>
                                Slow connection detected. Responses may take longer.
                            </div>
                        )}
                    </div>
                </div>
            </Dialog>
        </>
    );
};

export default Gpt;