import React, { useState, useEffect } from 'react';
import { InputText } from 'primereact/inputtext';
import { Button } from 'primereact/button';
import { Ollama } from '@langchain/ollama';
import { ProgressBar } from 'primereact/progressbar';
import axios from 'axios';
import './testRAGbat.css';
import configuration from './configuration.json';

const FinetuneTestbat = () => {
    const [selectedOllama, setSelectedOllama] = useState(null);
    const [selectedModel, setSelectedModel] = useState(null);
    const [chatTemperature, setChatTemperature] = useState(0.7);
    const [dialogVisible, setDialogVisible] = useState(true);
    const [testName, setTestName] = useState('');
    const [QAJSON, setQAJSON] = useState([{}]);
    const [isTesting, setIsTesting] = useState(false);
    const [total, setTotal] = useState(0);
    const [completed, setCompleted] = useState(0);
    const [results, setResults] = useState('');
    const [topP, setTopP] = useState(0.9);
    const [maxTokens, setMaxTokens] = useState(50);
    
    // Add new state variables for BERT RT tracking
    const [bertRtScores, setBertRtScores] = useState([]);
    const [globalAccuracy, setGlobalAccuracy] = useState(0);

    useEffect(() => {
        const ol = localStorage.getItem("selectedOllama") || 'http://127.0.0.1:11434';
        setSelectedOllama(ol);
        const mdl = localStorage.getItem("selectedLLMModel") || 'mistral';
        setSelectedModel(mdl);
        const chatT = localStorage.getItem("chatTemperature") || '0.7';
        setChatTemperature(parseFloat(chatT));
        console.log("Python", configuration.passer.PythonScore);
    }, []);

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

        setIsTesting(true);
        setTotal(QAJSON.length);
        setResults('');
        setCompleted(0);
        
        // Reset BERT RT tracking for new test
        setBertRtScores([]);
        setGlobalAccuracy(0);
        
        for (let i = 0; i < QAJSON.length; i++) {
            console.log("QAJSON[i]", QAJSON[i], i);
            const question = QAJSON[i].question;
            const answer = QAJSON[i].answer;
    
            // Create model instance for direct querying
            const mdl1 = new Ollama({
                baseUrl: selectedOllama,
                model: selectedModel,
                temperature: chatTemperature,
                top_p: topP,
                num_predict: maxTokens,
            });  
    
            try {
                // Format prompt for fine-tuned model
                const formattedPrompt = `<|user|>\n${question}\n<|assistant|>\n`;
                
                // Query the fine-tuned model directly
                const res = await mdl1.invoke(formattedPrompt);
                
                // Clean up response
                let cleanResponse = res;
                if (cleanResponse.includes(question)) {
                    cleanResponse = cleanResponse.replace(question, '').trim();
                }
                
                cleanResponse = cleanResponse
                    .replace('endoftext', '')
                    .replace('<|endoftext|>', '')
                    .replace('<|user|>', '')
                    .replace('<|assistant|>', '')
                    .trim();

                console.log("Direct model answer:", cleanResponse);
                
                const metrics = {
                    reference: QAJSON[i].answer, 
                    candidate: cleanResponse, 
                    userID: localStorage.getItem("wharf_user_name"), 
                    testID: testName,
                    description: 'question: ' + question.replace(/[^\x00-\x7F]/g, "") + ', answer: ' + cleanResponse.replace(/[^\x00-\x7F]/g, "")
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
                            `Answer: ${cleanResponse}\n` +
                            // `BLEU Score: ${calculatedMetrics.bleu_score.toFixed(3)}\n` +
                            // `METEOR Score: ${calculatedMetrics.meteor_score.toFixed(3)}\n` +
                            // `ROUGE-1 F1: ${calculatedMetrics.rouge_1.f1.toFixed(3)}\n` +
                            // `BERT Score F1: ${calculatedMetrics.bert_score.f1.toFixed(3)}\n` +
                            // `BERT RT Score: ${bertRtAvgScore.toFixed(2)}/5.0 (${matchPercentage.toFixed(1)}%)\n` +
                            `Global Accuracy: ${calculatedGlobalAccuracy.toFixed(1)}% (${updatedBertRtScores.length} questions)\n` +
                            `Status: ${isMatch ? '✅ MATCH' : '❌ NO MATCH'}\n\n`
                        );
                    } else {
                        // Fallback to original display if metrics not available
                        const isMatch = cleanResponse.toLowerCase().includes(QAJSON[i].answer.toLowerCase()) || 
                                       QAJSON[i].answer.toLowerCase().includes(cleanResponse.toLowerCase());
                        
                        setResults(prevResults => 
                            prevResults + 
                            `${i+1}-> Question: ${question}\n` + 
                            `Reference: ${QAJSON[i].answer}\n` + 
                            `Answer: ${cleanResponse}\n` +
                            `Status: ${isMatch ? '✅ MATCH' : '❌ NO MATCH'}\n\n`
                        );
                    }
                    
                } catch (error) {
                    console.error('Error sending data to Python server:', error);
                    
                    // Fallback for error case
                    const isMatch = cleanResponse.toLowerCase().includes(QAJSON[i].answer.toLowerCase()) || 
                                   QAJSON[i].answer.toLowerCase().includes(cleanResponse.toLowerCase());
                    
                    setResults(prevResults => 
                        prevResults + 
                        `${i+1}-> Question: ${question}\n` + 
                        `Reference: ${QAJSON[i].answer}\n` + 
                        `Answer: ${cleanResponse}\n` +
                        `Error: Could not calculate metrics\n` +
                        `Status: ${isMatch ? '✅ MATCH' : '❌ NO MATCH'}\n\n`
                    );
                }
                
            } catch (error) {
                console.error('Error querying model:', error);
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
    
    const endRef = React.useRef(null);

    React.useEffect(() => {
        endRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [results]);

    return (
        <div style={{ width: '100%', display: 'flex', justifyContent: 'space-between'}}>
            {/* Model Configuration Panel */}
            <div style={{ width: '35%', padding: '20px' }}>
                <h3>🤖 Fine-tuned Model Configuration</h3>
                
                {/* ... existing configuration fields ... */}
                
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

                <div style={{ marginBottom: '15px' }}>
                    <label><strong>Model Name:</strong></label>
                    <InputText 
                        style={{ width: '100%', marginTop: '5px' }} 
                        value={selectedModel || ''} 
                        onChange={(e) => {
                            setSelectedModel(e.target.value);
                            localStorage.setItem("selectedLLMModel", e.target.value);
                        }}
                        placeholder="your-finetuned-model"
                    />
                </div>

                <div style={{ display: 'flex', gap: '10px', marginBottom: '15px' }}>
                    <div style={{ flex: 1 }}>
                        <label><strong>Temperature:</strong></label>
                        <InputText 
                            style={{ width: '100%', marginTop: '5px' }} 
                            type="number"
                            step="0.1"
                            min="0"
                            max="2"
                            value={chatTemperature} 
                            onChange={(e) => {
                                const val = parseFloat(e.target.value);
                                setChatTemperature(val);
                                localStorage.setItem("chatTemperature", val.toString());
                            }}
                        />
                    </div>
                    <div style={{ flex: 1 }}>
                        <label><strong>Top-p:</strong></label>
                        <InputText 
                            style={{ width: '100%', marginTop: '5px' }} 
                            type="number"
                            step="0.1"
                            min="0"
                            max="1"
                            value={topP} 
                            onChange={(e) => setTopP(parseFloat(e.target.value))}
                        />
                    </div>
                </div>

                <div style={{ marginBottom: '15px' }}>
                    <label><strong>Max Tokens:</strong></label>
                    <InputText 
                        style={{ width: '100%', marginTop: '5px' }} 
                        type="number"
                        min="10"
                        max="500"
                        value={maxTokens} 
                        onChange={(e) => setMaxTokens(parseInt(e.target.value))}
                    />
                </div>

                {/* Add Global Accuracy Display */}
                {bertRtScores.length > 0 && (
                    <div style={{ 
                        padding: '15px', 
                        backgroundColor: globalAccuracy >= 70 ? '#e8f5e8' : '#fff5f5', 
                        border: `2px solid ${globalAccuracy >= 70 ? '#4caf50' : '#ff9800'}`, 
                        borderRadius: '8px',
                        fontSize: '14px',
                        marginBottom: '15px'
                    }}>
                        <strong>🎯 Global BERT RT Accuracy</strong><br/>
                        <div style={{ fontSize: '24px', fontWeight: 'bold', marginTop: '8px', color: globalAccuracy >= 70 ? '#4caf50' : '#ff9800' }}>
                            {globalAccuracy.toFixed(1)}%
                        </div>
                        <div style={{ fontSize: '12px', color: '#666', marginTop: '5px' }}>
                            Based on {bertRtScores.length} completed questions<br/>
                            Mean BERT RT Score: {(bertRtScores.reduce((sum, score) => sum + score, 0) / bertRtScores.length).toFixed(2)}/5.0
                        </div>
                    </div>
                )}

                <div style={{ 
                    padding: '15px', 
                    backgroundColor: '#f0f8ff', 
                    border: '1px solid #87ceeb', 
                    borderRadius: '8px',
                    fontSize: '13px',
                    lineHeight: '1.4'
                }}>
                    <strong>ℹ️ Direct Model Testing</strong><br/>
                    This interface tests your fine-tuned model directly without using vector databases or document retrieval. 
                    Perfect for evaluating knowledge retention after fine-tuning.
                </div>
            </div>

            {/* Testing Panel */}
            {dialogVisible &&
            <div style={{ width: '60%', marginLeft: '20px', padding: '20px'}}>
                <h3>📋 Fine-tuned Model Testing</h3>
                
                {/* ... existing UI elements ... */}
                
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
                
                {/* Test Status */}
                <div style={{ 
                    padding: '12px', 
                    backgroundColor: QAJSON && QAJSON.length > 0 && QAJSON[0].question ? '#e8f5e8' : '#fff5f5', 
                    border: `2px solid ${QAJSON && QAJSON.length > 0 && QAJSON[0].question ? '#4caf50' : '#f44336'}`, 
                    borderRadius: '8px',
                    marginBottom: '15px',
                    fontSize: '14px'
                }}>
                    <strong>Status:</strong> {QAJSON && QAJSON.length > 0 && QAJSON[0].question 
                        ? `✅ Ready to test ${QAJSON.length} questions` 
                        : '⚠️ Please load a JSON file with questions in format: [{"question":"q1", "answer":"a1"}, ...]'}
                </div>

                <Button 
                    label={isTesting ? "🔄 Testing..." : "🚀 Start Test"} 
                    style={{
                        marginTop: '10px', 
                        width: '100%',
                        padding: '12px',
                        fontSize: '16px',
                        fontWeight: 'bold'
                    }} 
                    onClick={() => startTest()} 
                    disabled={isTesting || !QAJSON || QAJSON.length === 0 || !QAJSON[0].question}
                    className="p-ml-2" 
                />    
                
                {isTesting && (
                    <div style={{ marginTop: '15px' }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
                            <span><strong>Progress: {completed}/{total}</strong></span>
                            <span><strong>{((completed / total) * 100).toFixed(1)}%</strong></span>
                        </div>
                        <ProgressBar                             
                            value={((completed / total) * 100)} 
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
                        <strong style={{ fontSize: '16px' }}>📊 Test Results</strong>
                        {completed > 0 && (
                            <div style={{ fontSize: '13px', color: '#00ff88', textAlign: 'right' }}>
                                <div>Individual Matches: {(results.match(/✅ MATCH/g) || []).length}/{completed}</div>
                                {bertRtScores.length > 0 && (
                                    <div>Global BERT RT Accuracy: {globalAccuracy.toFixed(1)}%</div>
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
                        {results || 'Test results will appear here...\n\nLoad a JSON file and click "Start Test" to begin.'}
                        <div ref={endRef} />
                    </pre>                
                </div>
            </div>}
        </div>
    );
}

export default FinetuneTestbat;