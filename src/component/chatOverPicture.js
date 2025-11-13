import React, { useEffect, useState, useRef, useCallback } from 'react';
import { Dialog } from 'primereact/dialog';
import { Button } from 'primereact/button';
import { InputText } from 'primereact/inputtext';
import { InputTextarea } from 'primereact/inputtextarea';
import { Card } from 'primereact/card';
import { ScrollPanel } from 'primereact/scrollpanel';
import { Toast } from 'primereact/toast';
import { Divider } from 'primereact/divider';
import { Chip } from 'primereact/chip';
import { Dropdown } from 'primereact/dropdown';
import { ProgressSpinner } from 'primereact/progressspinner';
import axios from 'axios';
import './chatOverPicture.css';

const ACCEPTED_TYPES = ['image/png', 'image/jpeg', 'image/jpg', 'image/tiff', 'image/bmp'];

const ChatOverPicture = () => {
  const [dialogVisible, setDialogVisible] = useState(true);
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [processedDataUrl, setProcessedDataUrl] = useState(null);
  const [processing, setProcessing] = useState(false);
  const [messages, setMessages] = useState([]);
  const [newMessage, setNewMessage] = useState('');
  const [isAnswering, setIsAnswering] = useState(false);
  const [selectedOllama, setSelectedOllama] = useState(localStorage.getItem('selectedOllama') || 'http://localhost:11434');
  const [selectedModel, setSelectedModel] = useState(localStorage.getItem('selectedLLMModel') || 'mistral');
  const [availableModels, setAvailableModels] = useState([]);
  const [connectionQuality, setConnectionQuality] = useState('good');

  const toast = useRef(null);
  const abortControllerRef = useRef(null);
  const messagesEndRef = useRef(null);
  const responseBuffer = useRef('');

  useEffect(() => {
    fetchModels();
    checkConnection();
  }, [selectedOllama]);

  const handleFileChange = (evt) => {
    const file = evt.target.files?.[0];
    if (!file) return;
    if (!ACCEPTED_TYPES.includes(file.type)) {
      toast.current?.show({ severity: 'error', summary: 'Invalid file', detail: 'Supported: png, jpg, tiff, bmp', life: 3000 });
      return;
    }
    setSelectedFile(file);
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
    setProcessedDataUrl(null);
  };

  const processImage = async (maxDim = 1200, quality = 0.8) => {
    if (!selectedFile) return;
    setProcessing(true);
    try {
      const img = await loadImageFromFile(selectedFile);
      const { w, h } = fitDimensions(img.width, img.height, maxDim);
      const canvas = document.createElement('canvas');
      canvas.width = w;
      canvas.height = h;
      const ctx = canvas.getContext('2d');
      ctx.drawImage(img, 0, 0, w, h);
      // Force JPEG format
      const dataUrl = canvas.toDataURL('image/jpeg', quality);
      setProcessedDataUrl(dataUrl);
      toast.current?.show({ severity: 'success', summary: 'Image processed', detail: `Resized to ${w}×${h} (JPEG)`, life: 2000 });
    } catch (err) {
      console.error(err);
      toast.current?.show({ severity: 'error', summary: 'Processing failed', detail: err.message || String(err), life: 3000 });
    } finally {
      setProcessing(false);
    }
  };

  const loadImageFromFile = (file) => {
    return new Promise((resolve, reject) => {
      const url = URL.createObjectURL(file);
      const img = new Image();
      img.onload = () => {
        URL.revokeObjectURL(url);
        resolve(img);
      };
      img.onerror = () => {
        URL.revokeObjectURL(url);
        reject(new Error('Failed to load image'));
      };
      img.src = url;
    });
  };

  const fitDimensions = (w, h, maxDim) => {
    if (w <= maxDim && h <= maxDim) return { w, h };
    const ratio = Math.max(w / maxDim, h / maxDim);
    return { w: Math.round(w / ratio), h: Math.round(h / ratio) };
  };

  const fetchModels = useCallback(async () => {
    try {
      const resp = await axios.get(selectedOllama + '/api/tags', { timeout: 8000 });
      if (resp.data?.models) {
        setAvailableModels(resp.data.models.map((m) => ({ label: m.name, value: m.name })));
      }
    } catch (err) {
      console.warn('Failed to fetch models:', err);
    }
  }, [selectedOllama]);

  const checkConnection = useCallback(async () => {
    try {
      const start = Date.now();
      await axios.get(selectedOllama + '/api/version', { timeout: 4000 });
      const latency = Date.now() - start;
      setConnectionQuality(latency < 500 ? 'excellent' : latency < 1200 ? 'good' : 'fair');
    } catch {
      setConnectionQuality('offline');
    }
  }, [selectedOllama]);

  const streamResponse = async (prompt, responseId) => {
    const controller = new AbortController();
    abortControllerRef.current = controller;
    responseBuffer.current = '';
    setIsAnswering(true);

      // Debug logging
    // console.log('=== DEBUG INFO ===');
    // console.log('Selected model:', selectedModel);
    // console.log('Has processedDataUrl:', !!processedDataUrl);
    // console.log('Image format:', processedDataUrl ? processedDataUrl.substring(0, 50) + '...' : 'No image');
    // console.log('File type:', selectedFile?.type);
    // console.log('Prompt:', prompt);
  // Try different image formats
    let imagePayload = null;
    if (processedDataUrl) {
      // Remove data URL prefix and send just base64
      imagePayload = processedDataUrl.replace(/^data:[^;]+;base64,/, '');
    }

    // Alternative: try 'image' field with just base64
    const body = {
      model: selectedModel,
      prompt,
      stream: true,
      images: imagePayload ? [imagePayload] : [], // Try 'images' array instead of 'image'
      options: { temperature: 0.7 } // Increased temperature
    };

    // const body = {
    //   model: selectedModel,
    //   prompt,
    //   stream: true,
    //   image: processedDataUrl || null,
    //   options: { temperature: 0.2 }
    // };
    // console.log('Request body structure:', {
    //   model: body.model,
    //   prompt: body.prompt,
    //   hasImages: body.images.length > 0,
    //   imageLength: imagePayload ? imagePayload.length : 0
    // });

    // console.log("Processed Image Data URL:", processedDataUrl); // Add this line to log the image data

    const resp = await fetch(selectedOllama + '/api/generate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
      signal: controller.signal
    });

    if (!resp.ok) {
      const txt = await resp.text();
      throw new Error(`HTTP ${resp.status}: ${txt}`);
    }

    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let partial = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      partial += decoder.decode(value, { stream: true });
      const lines = partial.split('\n');
      partial = lines.pop() || '';
      for (const line of lines) {
        if (!line.trim()) continue;
        try {
          const json = JSON.parse(line);
          if (json.response) {
            responseBuffer.current += json.response;
            setMessages((prev) => {
              const next = [...prev];
              const last = next[next.length - 1];
              if (last && last.id === responseId) {
                last.text = responseBuffer.current;
              }
              return next;
            });
          }
          if (json.done) {
            setIsAnswering(false);
            return;
          }
        } catch (e) {
          responseBuffer.current += line;
          setMessages((prev) => {
            const next = [...prev];
            const last = next[next.length - 1];
            if (last && last.id === responseId) {
              last.text = responseBuffer.current;
            }
            return next;
          });
        }
      }
    }

    setIsAnswering(false);
  };

  const handleSubmit = async () => {
    if (!newMessage.trim()) return;
    if (!processedDataUrl) {
      toast.current?.show({ severity: 'warn', summary: 'No image', detail: 'Process and/or upload an image first', life: 2500 });
      return;
    }
    const question = newMessage.trim();
    const userMsg = { id: `u-${Date.now()}`, sender: 'You', text: question, timestamp: new Date().toLocaleTimeString() };
    const responseId = `r-${Date.now()}`;
    const placeholder = { id: responseId, sender: selectedModel, text: '⌛ Analyzing the image...', timestamp: new Date().toLocaleTimeString() };
    setMessages((prev) => [...prev, userMsg, placeholder]);
    setNewMessage('');
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  
    try {
      // Fixed: Use neutral prompt that doesn't bias the model
      await streamResponse(question, responseId);
      // await streamResponse(
      //   `Please analyze the provided image and answer the following question: ${question}`,
      //   responseId
      // );
    } catch (err) {
      console.error(err);
      setMessages((prev) => {
        const next = [...prev];
        const last = next[next.length - 1];
        if (last && last.id === responseId) {
          last.text = `❌ Error: ${err.message || String(err)}`;
        }
        return next;
      });
      toast.current?.show({ severity: 'error', summary: 'Request failed', detail: err.message || 'Error', life: 4000 });
      setIsAnswering(false);
    }
  };

  const handleCancel = () => {
    abortControllerRef.current?.abort();
    setIsAnswering(false);
  };

  const renderMessage = (m) => (
    <Card key={m.id} className={`message-card ${m.sender === 'You' ? 'user-message' : 'model-message'}`}>
      <div className="message-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div>
          <strong>{m.sender}</strong>
          <span style={{ marginLeft: 8, color: '#888' }}>{m.timestamp}</span>
        </div>
        <Button
          icon="pi pi-copy"
          className="p-button-text p-button-sm"
          onClick={() => navigator.clipboard.writeText(m.text)}
          tooltip="Copy to clipboard"
          tooltipOptions={{ position: 'top' }}
        />
      </div>
      <div className="message-content">
        {m.sender !== 'You' && processedDataUrl && (
          <div style={{ display: 'flex', gap: 12, alignItems: 'flex-start', marginBottom: 8 }}>
            <img src={processedDataUrl} alt="uploaded" style={{ width: 140, height: 'auto', borderRadius: 6, boxShadow: '0 2px 8px rgba(0,0,0,0.08)' }} />
            <div style={{ flex: 1, whiteSpace: 'pre-wrap' }}>
              <div style={{ fontFamily: 'ui-monospace, SFMono-Regular, monospace' }}>{m.text}</div>
            </div>
          </div>
        )}
        {m.sender === 'You' && <div style={{ whiteSpace: 'pre-wrap' }}>{m.text}</div>}
        {m.sender !== 'You' && !processedDataUrl && <div style={{ whiteSpace: 'pre-wrap' }}>{m.text}</div>}
      </div>
    </Card>
  );
  
  return (
    <>
      <Toast ref={toast} position="top-right" />
      <Dialog
        header={
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', width: '100%' }}>
            <div style={{ display: 'flex', gap: 12, alignItems: 'center' }}>
              <i className="pi pi-image" style={{ fontSize: 20 }} />
              <strong>Chat over Picture</strong>
              <Chip label={connectionQuality} style={{ marginLeft: 8 }} />
            </div>
            <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
              <Dropdown value={selectedModel} options={availableModels} onChange={(e) => { setSelectedModel(e.value); localStorage.setItem('selectedLLMModel', e.value); }} placeholder="Model" />
              <InputText
                value={selectedOllama}
                onChange={(e) => { setSelectedOllama(e.target.value); localStorage.setItem('selectedOllama', e.target.value); }}
                style={{ width: 300 }}
              />
            </div>
          </div>
        }
        visible={dialogVisible}
        style={{ width: '95%', maxWidth: 1100 }}
        modal
        onHide={() => setDialogVisible(false)}
        maximizable
      >
        <div style={{ display: 'flex', gap: 18 }}>
          <div style={{ width: 360 }}>
            <Card title="Image" className="image-panel">
              <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
                <input
                  id="fileInput"
                  type="file"
                  accept=".png,.jpg,.jpeg,.tiff,.bmp"
                  onChange={handleFileChange}
                  style={{ display: 'block' }}
                />
                {previewUrl && (
                  <div style={{ textAlign: 'center' }}>
                    <img src={previewUrl} alt="preview" style={{ maxWidth: '100%', borderRadius: 6, maxHeight: 260, objectFit: 'contain' }} />
                    <div style={{ marginTop: 8, color: '#666', fontSize: 12 }}>Original preview</div>
                  </div>
                )}
                <div style={{ display: 'flex', gap: 8 }}>
                  <Button label="Process" icon="pi pi-cog" onClick={() => processImage()} disabled={!selectedFile || processing} />
                  <Button label="Clear" icon="pi pi-trash" className="p-button-secondary" onClick={() => { setSelectedFile(null); setPreviewUrl(null); setProcessedDataUrl(null); }} />
                </div>
                {processing && <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}><ProgressSpinner /><small>Processing image...</small></div>}
              </div>
            </Card>
          </div>
          <div style={{ flex: 1, display: 'flex', flexDirection: 'column', minHeight: 480 }}>
            <Card title="Conversation">
              <ScrollPanel style={{ height: '52vh' }}>
                {messages.length === 0 && (
                  <div style={{ padding: 20, textAlign: 'center', color: '#777' }}>
                    Upload and process an image, then ask a question about it.
                  </div>
                )}
                <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
                  {messages.map((m) => renderMessage(m))}
                  <div ref={messagesEndRef} />
                </div>
              </ScrollPanel>
              <Divider />
              <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                <InputTextarea
                  value={newMessage}
                  onChange={(e) => setNewMessage(e.target.value)}
                  rows={3}
                  placeholder="Ask about the uploaded image..."
                  disabled={isAnswering}
                />
                <div style={{ display: 'flex', gap: 8, alignItems: 'center', justifyContent: 'space-between' }}>
                  <div style={{ display: 'flex', gap: 8 }}>
                    <Button label="Send" icon="pi pi-send" onClick={handleSubmit} disabled={isAnswering || !processedDataUrl || !newMessage.trim()} />
                    <Button label="Cancel" icon="pi pi-times" className="p-button-danger" onClick={handleCancel} disabled={!isAnswering} />
                  </div>
                  <Button label="Clear Chat" icon="pi pi-trash" className="p-button-text" onClick={() => setMessages([])} />
                </div>
              </div>
            </Card>
          </div>
        </div>
      </Dialog>
    </>
  );
};

export default ChatOverPicture;