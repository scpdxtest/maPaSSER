import React, { useState, useRef, useEffect } from "react";
import axios from "axios";
import { InputTextarea } from "primereact/inputtextarea";
import { InputNumber } from "primereact/inputnumber";
import { InputText } from "primereact/inputtext";
import { Button } from "primereact/button";
import { ProgressBar } from "primereact/progressbar";
import { Card } from "primereact/card";
import { Divider } from "primereact/divider";
import { Message } from "primereact/message";
import { Dropdown } from "primereact/dropdown";
import { FileUpload } from "primereact/fileupload";
import { Slider } from "primereact/slider";
import { TabView, TabPanel } from "primereact/tabview";
import "primereact/resources/themes/lara-light-indigo/theme.css";
import "primereact/resources/primereact.min.css";
import "primeicons/primeicons.css";
import "./ImageEditChat.css";

const T2ImgEdit = () => {
  // Image states
  const [originalImage, setOriginalImage] = useState(null);
  const [editedImage, setEditedImage] = useState(null);
  const [activeImageType, setActiveImageType] = useState('original'); // 'original' or 'edited'
  
  // Generation states
  const [prompt, setPrompt] = useState("");
  const [negativePrompt, setNegativePrompt] = useState("");
  const [numInferenceSteps, setNumInferenceSteps] = useState(20);
  const [guidanceScale, setGuidanceScale] = useState(7.5);
  const [width, setWidth] = useState(512);
  const [height, setHeight] = useState(512);
  
  // Edit-specific states
  const [editPrompt, setEditPrompt] = useState("");
  const [strength, setStrength] = useState(0.7);
  
  // UI states
  const [fileName, setFileName] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [isEditing, setIsEditing] = useState(false);
  const [progress, setProgress] = useState(0);
  const [remainingTime, setRemainingTime] = useState("");
  const [error, setError] = useState("");
  const [selectedPort, setSelectedPort] = useState(8002); // Port for edit API
  const [selectedApiUrl, setSelectedApiUrl] = useState("http://127.0.0.1");
  const [activeTabIndex, setActiveTabIndex] = useState(0);

  const wsRef = useRef(null);
  const cancelledRef = useRef(false);
  const fileUploadRef = useRef(null);

  const apiUrlOptions = [
    { label: "Local", value: "http://127.0.0.1" },
    { label: "Remote", value: "http://195.230.127.227" },
  ];

  const getPortOptions = () => {
    if (selectedApiUrl === "http://127.0.0.1") {
      return [
        { label: "8002 (Image Edit API)", value: 8002 },
        { label: "8001 (T2I with LoRA)", value: 8001 },
        { label: "8000 (Direct T2I)", value: 8000 },
      ];
    } else if (selectedApiUrl === "http://195.230.127.227") {
      return [
        { label: "8280 (FLUX-Dev.1)", value: 8280 },
        { label: "8281 (FLUX-Dev.1/EDIT)", value: 8281 }
      ];
    }
    return [];
  };

  useEffect(() => {
    return () => {
      cleanupWebSocket();
      if (originalImage) URL.revokeObjectURL(originalImage);
      if (editedImage) URL.revokeObjectURL(editedImage);
    };
  }, []);

  const cleanupWebSocket = () => {
    if (wsRef.current) {
      wsRef.current.onopen = null;
      wsRef.current.onmessage = null;
      wsRef.current.onerror = null;
      wsRef.current.onclose = null;
      if (
        wsRef.current.readyState === WebSocket.OPEN ||
        wsRef.current.readyState === WebSocket.CONNECTING
      ) {
        wsRef.current.close();
      }
      wsRef.current = null;
    }
  };

  const handleCancel = () => {
    cancelledRef.current = true;
    cleanupWebSocket();
    setIsLoading(false);
    setIsEditing(false);
    setProgress(0);
    setRemainingTime("");
    setError("Operation cancelled by user");
  };

  const getApiUrl = (endpoint) => `${selectedApiUrl}:${selectedPort}${endpoint}`;
  const getWebSocketUrl = (wsEndpoint) => `${selectedApiUrl.replace("http", "ws")}:${selectedPort}${wsEndpoint}`;

  // Convert image to base64
  const convertImageToBase64 = (imageUrl) => {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.crossOrigin = 'anonymous';
      img.onload = () => {
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        canvas.width = img.width;
        canvas.height = img.height;
        ctx.drawImage(img, 0, 0);
        const dataURL = canvas.toDataURL('image/png');
        resolve(dataURL.split(',')[1]); // Remove data:image/png;base64, prefix
      };
      img.onerror = reject;
      img.src = imageUrl;
    });
  };

  // Handle file upload
  const handleFileUpload = (event) => {
    const file = event.files[0];
    if (file) {
      if (originalImage) URL.revokeObjectURL(originalImage);
      const imageUrl = URL.createObjectURL(file);
      setOriginalImage(imageUrl);
      setActiveImageType('original');
      setError("");
      
      // Auto-fill dimensions based on uploaded image
      const img = new Image();
      img.onload = () => {
        setWidth(Math.round(img.width / 8) * 8); // Ensure divisible by 8
        setHeight(Math.round(img.height / 8) * 8);
      };
      img.src = imageUrl;
    }
    fileUploadRef.current.clear();
  };

  // Generate new image (T2I) using WebSocket
  const handleGenerate = async () => {
    if (!prompt.trim()) {
      setError("Please enter a prompt");
      return;
    }

    setIsLoading(true);
    setProgress(0);
    setRemainingTime("");
    setError("");
    cancelledRef.current = false;

    try {
      cleanupWebSocket();
      const ws = new WebSocket(getWebSocketUrl("/generate"));
      wsRef.current = ws;

      ws.onopen = () => {
        console.log(`WebSocket connected to ${getWebSocketUrl("/generate")}`);
        if (ws.readyState === WebSocket.OPEN && !cancelledRef.current) {
          const requestData = {
            prompt,
            negative_prompt: negativePrompt,
            num_inference_steps: numInferenceSteps,
            guidance_scale: guidanceScale,
            width,
            height,
            lora_weight: 1.0
          };
          ws.send(JSON.stringify(requestData));
        }
      };

      ws.onmessage = (event) => {
        if (cancelledRef.current) return;
        handleWebSocketMessage(event, () => fetchGeneratedImage());
      };

      ws.onerror = (err) => {
        if (cancelledRef.current) return;
        console.error("WebSocket error:", err);
        setError(`Failed to connect to ${getWebSocketUrl("/generate")}`);
        cleanupWebSocket();
        setIsLoading(false);
      };

      ws.onclose = (event) => {
        console.log("WebSocket closed:", event.code, event.reason);
        if (wsRef.current === ws) wsRef.current = null;
      };
    } catch (err) {
      if (cancelledRef.current) return;
      console.error("WebSocket setup error:", err);
      setError(`Failed to setup WebSocket connection on ${getWebSocketUrl("/generate")}`);
      cleanupWebSocket();
      setIsLoading(false);
    }
  };

  // Edit existing image using WebSocket for progress
  const handleEditImage = async () => {
    if (!originalImage) {
      setError("Please upload an image first");
      return;
    }
    if (!editPrompt.trim()) {
      setError("Please enter an edit prompt");
      return;
    }

    setIsEditing(true);
    setProgress(0);
    setRemainingTime("");
    setError("");
    cancelledRef.current = false;

    try {
      cleanupWebSocket();
      const ws = new WebSocket(getWebSocketUrl("/edit"));
      wsRef.current = ws;

      ws.onopen = async () => {
        console.log(`WebSocket connected to ${getWebSocketUrl("/edit")}`);
        if (ws.readyState === WebSocket.OPEN && !cancelledRef.current) {
          try {
            const imageBase64 = await convertImageToBase64(originalImage);
            const requestData = {
              image_base64: imageBase64,
              prompt: editPrompt,
              negative_prompt: negativePrompt,
              num_inference_steps: numInferenceSteps,
              guidance_scale: guidanceScale,
              strength: strength,
              lora_weight: 1.0
            };
            ws.send(JSON.stringify(requestData));
          } catch (err) {
            setError(`Failed to convert image: ${err.message}`);
            setIsEditing(false);
            cleanupWebSocket();
          }
        }
      };

      ws.onmessage = (event) => {
        if (cancelledRef.current) return;
        handleWebSocketMessage(event, () => fetchEditedImage());
      };

      ws.onerror = (err) => {
        if (cancelledRef.current) return;
        console.error("Edit WebSocket error:", err);
        setError(`Failed to connect to ${getWebSocketUrl("/edit")}`);
        cleanupWebSocket();
        setIsEditing(false);
      };

      ws.onclose = (event) => {
        console.log("Edit WebSocket closed:", event.code, event.reason);
        if (wsRef.current === ws) wsRef.current = null;
      };
    } catch (err) {
      if (cancelledRef.current) return;
      console.error("Edit WebSocket setup error:", err);
      setError(`Failed to setup edit WebSocket connection`);
      cleanupWebSocket();
      setIsEditing(false);
    }
  };

  // Handle WebSocket messages
  const handleWebSocketMessage = (event, onComplete) => {
    let payload;
    try {
      payload = JSON.parse(event.data);
    } catch (e) {
      payload = { type: "text", message: event.data };
    }

    if (payload.type === "progress") {
      const p = Number(payload.progress) || 0;
      setProgress(p);
      if (payload.eta_seconds != null && !isNaN(Number(payload.eta_seconds))) {
        const total = Math.max(0, Math.floor(Number(payload.eta_seconds)));
        const m = Math.floor(total / 60);
        const s = total % 60;
        const etaStr = m >= 60 ? `${Math.floor(m/60)}:${String(m%60).padStart(2,"0")}:${String(s).padStart(2,"0")}` : `${String(m).padStart(2,"0")}:${String(s).padStart(2,"0")}`;
        setRemainingTime(etaStr);
      } else if (payload.eta_str) {
        setRemainingTime(payload.eta_str);
      }
    } else if (payload.type === "complete") {
      setProgress(100);
      setRemainingTime("00:00");
      cleanupWebSocket();
      onComplete();
    } else if (payload.type === "error") {
      setError(`Error: ${payload.message || "unknown"}`);
      cleanupWebSocket();
      setIsLoading(false);
      setIsEditing(false);
    } else if (payload.type === "text") {
      // Handle status messages
      console.log("Status:", payload.message);
    }
  };

  // ...existing code...
  
  // Update the fetchGeneratedImage function - change the endpoint
  const fetchGeneratedImage = async () => {
    if (cancelledRef.current) return;
    try {
      await new Promise((r) => setTimeout(r, 1000));
      if (cancelledRef.current) return;
      
      // FIXED: Changed from "/get-image" to "/generate-image" to match the API
      const response = await axios.post(getApiUrl("/generate-image"), {}, {
        responseType: "blob",
        timeout: 30000,
      });
      
      if (cancelledRef.current) return;
      
      if (originalImage) URL.revokeObjectURL(originalImage);
      const imageUrl = URL.createObjectURL(response.data);
      setOriginalImage(imageUrl);
      setActiveImageType('original');
      setProgress(100);
    } catch (err) {
      if (cancelledRef.current) return;
      console.error("Error fetching image:", err);
      setError(`Failed to fetch the generated image`);
    } finally {
      if (!cancelledRef.current) setIsLoading(false);
    }
  };
  
  // Update the fetchEditedImage function - change the endpoint
  const fetchEditedImage = async () => {
    if (cancelledRef.current) return;
    try {
      await new Promise((r) => setTimeout(r, 1000));
      if (cancelledRef.current) return;
      
      // FIXED: Changed from "/get-image" to "/generate-image" to match the API
      const response = await axios.post(getApiUrl("/generate-image"), {}, {
        responseType: "blob",
        timeout: 30000,
      });
      
      if (cancelledRef.current) return;
      
      if (editedImage) URL.revokeObjectURL(editedImage);
      const imageUrl = URL.createObjectURL(response.data);
      setEditedImage(imageUrl);
      setActiveImageType('edited');
      setProgress(100);
    } catch (err) {
      if (cancelledRef.current) return;
      console.error("Error fetching edited image:", err);
      setError(`Failed to fetch the edited image`);
    } finally {
      if (!cancelledRef.current) setIsEditing(false);
    }
  };
  
  // Direct edit (alternative without WebSocket)
  const handleEditImageDirect = async () => {
    if (!originalImage) {
      setError("Please upload an image first");
      return;
    }
    if (!editPrompt.trim()) {
      setError("Please enter an edit prompt");
      return;
    }

    setIsEditing(true);
    setProgress(0);
    setError("");
    cancelledRef.current = false;

    try {
      const imageBase64 = await convertImageToBase64(originalImage);
      
      const response = await axios.post(getApiUrl("/edit-direct"), {
        image_base64: imageBase64,
        prompt: editPrompt,
        negative_prompt: negativePrompt,
        num_inference_steps: numInferenceSteps,
        guidance_scale: guidanceScale,
        strength: strength,
        lora_weight: 1.0
      }, {
        responseType: "blob",
        timeout: 120000,
      });

      if (cancelledRef.current) return;
      
      if (editedImage) URL.revokeObjectURL(editedImage);
      const newImageUrl = URL.createObjectURL(response.data);
      setEditedImage(newImageUrl);
      setActiveImageType('edited');
      setProgress(100);
    } catch (err) {
      if (cancelledRef.current) return;
      console.error("Error editing image:", err);
      setError(`Failed to edit image: ${err.response?.data?.detail || err.message}`);
    } finally {
      if (!cancelledRef.current) setIsEditing(false);
    }
  };

  // Save image
  const handleSave = () => {
    const imageToSave = activeImageType === 'edited' ? editedImage : originalImage;
    if (!imageToSave) {
      setError("No image to save");
      return;
    }
    const finalFileName = fileName.trim() || `${activeImageType}_image.png`;
    const link = document.createElement("a");
    link.href = imageToSave;
    link.download = finalFileName.endsWith(".png") ? finalFileName : `${finalFileName}.png`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    setError("");
  };

  // Use edited image as new original
  const handleUseAsOriginal = () => {
    if (!editedImage) return;
    if (originalImage) URL.revokeObjectURL(originalImage);
    setOriginalImage(editedImage);
    setEditedImage(null);
    setActiveImageType('original');
    setEditPrompt("");
  };

  const getModelInfo = () => {
    const portInfo = getPortOptions().find((option) => option.value === selectedPort);
    return portInfo ? portInfo.label : `Port ${selectedPort}`;
  };

  const getCurrentImage = () => {
    return activeImageType === 'edited' ? editedImage : originalImage;
  };

  const imageDisplay = () => (
    <div className="image-display-container">
      {getCurrentImage() ? (
        <div style={{ textAlign: "center" }}>
          <img 
            src={getCurrentImage()} 
            alt={activeImageType} 
            style={{ maxWidth: "100%", maxHeight: "400px", borderRadius: "8px", boxShadow: "0 4px 6px rgba(0,0,0,0.1)" }} 
          />
          <div className="image-type-buttons">
            {originalImage && (
              <Button 
                label="Original" 
                className={`p-button-sm ${activeImageType === 'original' ? 'p-button-primary' : 'p-button-outlined'}`}
                onClick={() => setActiveImageType('original')}
              />
            )}
            {editedImage && (
              <Button 
                label="Edited" 
                className={`p-button-sm ${activeImageType === 'edited' ? 'p-button-primary' : 'p-button-outlined'}`}
                onClick={() => setActiveImageType('edited')}
              />
            )}
          </div>
        </div>
      ) : (
        <div className="empty-image-state">
          <i className="pi pi-image"></i>
          <p>No image yet</p>
          <small>Upload an image or generate a new one</small>
        </div>
      )}
    </div>
  );

  return (
    <div className="image-edit-container">
      <div className="p-grid p-justify-center p-align-start p-mt-4" style={{ padding: "1rem" }}>
        <div className="p-col-12 p-md-6 image-panel">
          <Card title="Image Editor" style={{ height: "100%" }}>
            {imageDisplay()}
            <Divider />
            
            {editedImage && (
              <div className="p-field" style={{ marginBottom: "1rem" }}>
                <Button 
                  label="Use Edited as Original" 
                  icon="pi pi-arrow-right"
                  className="p-button-info p-button-sm"
                  onClick={handleUseAsOriginal}
                  style={{ width: "100%" }}
                />
              </div>
            )}

            <div className="p-fluid">
              <div className="p-field">
                <label htmlFor="fileName">File Name</label>
                <div className="p-inputgroup">
                  <InputText 
                    id="fileName" 
                    value={fileName} 
                    onChange={(e) => setFileName(e.target.value)} 
                    placeholder={`${activeImageType}_image.png`} 
                  />
                  <Button 
                    icon="pi pi-download" 
                    label="Save" 
                    className="p-button-success" 
                    onClick={handleSave} 
                    disabled={!getCurrentImage()} 
                  />
                </div>
              </div>
            </div>
          </Card>
        </div>

        <div className="p-col-12 p-md-6 controls-panel">
          <Card title={`Image Editor - ${getModelInfo()}`}>
            <div className="p-fluid">
              {error && <Message severity="error" text={error} style={{ marginBottom: "1rem" }} />}

              <div className="p-grid">
                <div className="p-col-6">
                  <div className="p-field">
                    <label htmlFor="apiUrl">API URL</label>
                    <Dropdown 
                      id="apiUrl" 
                      value={selectedApiUrl} 
                      onChange={(e) => { 
                        setSelectedApiUrl(e.value); 
                        // Updated port assignment logic
                        if (e.value === "http://127.0.0.1") {
                          setSelectedPort(8002);
                        } else if (e.value === "http://195.230.127.227") {
                          setSelectedPort(8281); // Default to FLUX-Dev.1/EDIT for remote
                        }
                      }} 
                      options={apiUrlOptions} 
                      placeholder="Select API URL" 
                      disabled={isLoading || isEditing} 
                    />
                  </div>
                </div>
                <div className="p-col-6">
                  <div className="p-field">
                    <label htmlFor="port">API Port</label>
                    <Dropdown 
                      id="port" 
                      value={selectedPort} 
                      onChange={(e) => setSelectedPort(e.value)} 
                      options={getPortOptions()} 
                      placeholder="Select API Port" 
                      disabled={isLoading || isEditing} 
                    />
                  </div>
                </div>
              </div>

              <TabView activeIndex={activeTabIndex} onTabChange={(e) => setActiveTabIndex(e.index)}>
                <TabPanel header="Upload & Edit">
                  <div className="p-field">
                    <label htmlFor="fileUpload">Upload Image</label>
                    <FileUpload
                      ref={fileUploadRef}
                      mode="basic"
                      name="image"
                      accept="image/*"
                      maxFileSize={10000000}
                      onSelect={handleFileUpload}
                      auto
                      chooseLabel="Choose Image"
                      className="p-button-outlined"
                    />
                  </div>

                  <div className="p-field">
                    <label htmlFor="editPrompt">Edit Prompt</label>
                    <InputTextarea 
                      id="editPrompt" 
                      value={editPrompt} 
                      onChange={(e) => setEditPrompt(e.target.value)} 
                      rows={3} 
                      autoResize 
                      placeholder="Describe how you want to modify the image..."
                      disabled={isEditing}
                    />
                  </div>

                  <div className="p-field">
                    <label htmlFor="strength">Edit Strength: {strength}</label>
                    <Slider 
                      id="strength" 
                      value={strength} 
                      onChange={(e) => setStrength(e.value)} 
                      min={0.1} 
                      max={1.0} 
                      step={0.05}
                      disabled={isEditing}
                    />
                    <small>Lower values = subtle changes, Higher values = major changes</small>
                  </div>

                  <div className="p-field">
                    {isEditing ? (
                      <div className="p-grid" style={{ margin: 0 }}>
                        <div className="p-col-8" style={{ paddingRight: "0.25rem" }}>
                          <Button 
                            label={`Editing... ${progress}%${remainingTime ? ` (ETA ${remainingTime})` : ""}`} 
                            icon="pi pi-spin pi-spinner" 
                            className="p-button-primary p-button-lg" 
                            disabled={true} 
                            style={{ width: "100%" }} 
                          />
                        </div>
                        <div className="p-col-4" style={{ paddingLeft: "0.25rem" }}>
                          <Button 
                            label="Cancel" 
                            icon="pi pi-times" 
                            className="p-button-danger p-button-lg" 
                            onClick={handleCancel} 
                            style={{ width: "100%" }} 
                          />
                        </div>
                      </div>
                    ) : (
                      <div className="p-grid" style={{ margin: 0 }}>
                        <div className="p-col-6" style={{ paddingRight: "0.25rem" }}>
                          <Button 
                            label="Edit (WS)" 
                            icon="pi pi-pencil" 
                            onClick={handleEditImage} 
                            className="p-button-warning p-button-lg" 
                            style={{ width: "100%" }}
                            disabled={!originalImage || !editPrompt.trim()}
                          />
                        </div>
                        {/* <div className="p-col-6" style={{ paddingLeft: "0.25rem" }}>
                          <Button 
                            label="Edit (Direct)" 
                            icon="pi pi-bolt" 
                            onClick={handleEditImageDirect} 
                            className="p-button-secondary p-button-lg" 
                            style={{ width: "100%" }}
                            disabled={!originalImage || !editPrompt.trim()}
                          />
                        </div> */}
                      </div>
                    )}
                  </div>
                </TabPanel>

                <TabPanel header="Generate New">
                  <div className="p-field">
                    <label htmlFor="prompt">Prompt *</label>
                    <InputTextarea 
                      id="prompt" 
                      value={prompt} 
                      onChange={(e) => setPrompt(e.target.value)} 
                      rows={4} 
                      autoResize 
                      placeholder="Describe the image you want to generate..."
                      disabled={isLoading}
                    />
                  </div>

                  <div className="p-field">
                    {isLoading ? (
                      <div className="p-grid" style={{ margin: 0 }}>
                        <div className="p-col-8" style={{ paddingRight: "0.25rem" }}>
                          <Button 
                            label={`Generating... ${progress}%${remainingTime ? ` (ETA ${remainingTime})` : ""}`} 
                            icon="pi pi-spin pi-spinner" 
                            className="p-button-primary p-button-lg" 
                            disabled={true} 
                            style={{ width: "100%" }} 
                          />
                        </div>
                        <div className="p-col-4" style={{ paddingLeft: "0.25rem" }}>
                          <Button 
                            label="Cancel" 
                            icon="pi pi-times" 
                            className="p-button-danger p-button-lg" 
                            onClick={handleCancel} 
                            style={{ width: "100%" }} 
                          />
                        </div>
                      </div>
                    ) : (
                      <Button 
                        label="Generate New Image" 
                        icon="pi pi-bolt" 
                        onClick={handleGenerate} 
                        className="p-button-primary p-button-lg" 
                        style={{ width: "100%" }}
                        disabled={!prompt.trim()}
                      />
                    )}
                  </div>
                </TabPanel>

                <TabPanel header="Settings">
                  <div className="p-field">
                    <label htmlFor="negativePrompt">Negative Prompt</label>
                    <InputTextarea 
                      id="negativePrompt" 
                      value={negativePrompt} 
                      onChange={(e) => setNegativePrompt(e.target.value)} 
                      rows={2} 
                      autoResize 
                      placeholder="What you don't want in the image..."
                      disabled={isLoading || isEditing}
                    />
                  </div>

                  <div className="p-grid">
                    <div className="p-col-6">
                      <div className="p-field">
                        <label htmlFor="numInferenceSteps">Inference Steps</label>
                        <InputNumber 
                          id="numInferenceSteps" 
                          value={numInferenceSteps} 
                          onValueChange={(e) => setNumInferenceSteps(e.value)} 
                          min={1} 
                          max={100} 
                          showButtons
                          disabled={isLoading || isEditing}
                        />
                      </div>
                    </div>
                    <div className="p-col-6">
                      <div className="p-field">
                        <label htmlFor="guidanceScale">Guidance Scale</label>
                        <InputNumber 
                          id="guidanceScale" 
                          value={guidanceScale} 
                          onValueChange={(e) => setGuidanceScale(e.value)} 
                          step={0.1} 
                          min={0.1} 
                          max={20} 
                          showButtons
                          disabled={isLoading || isEditing}
                        />
                      </div>
                    </div>
                  </div>

                  <div className="p-grid">
                    <div className="p-col-6">
                      <div className="p-field">
                        <label htmlFor="width">Width</label>
                        <InputNumber 
                          id="width" 
                          value={width} 
                          onValueChange={(e) => setWidth(e.value)} 
                          min={64} 
                          max={1024} 
                          step={64} 
                          showButtons
                          disabled={isLoading || isEditing}
                        />
                      </div>
                    </div>
                    <div className="p-col-6">
                      <div className="p-field">
                        <label htmlFor="height">Height</label>
                        <InputNumber 
                          id="height" 
                          value={height} 
                          onValueChange={(e) => setHeight(e.value)} 
                          min={64} 
                          max={1024} 
                          step={64} 
                          showButtons
                          disabled={isLoading || isEditing}
                        />
                      </div>
                    </div>
                  </div>

                  {(isLoading || isEditing) && (
                    <div className="p-field">
                      <label>Progress</label>
                      <ProgressBar 
                        value={progress} 
                        displayValueTemplate={(value) => (remainingTime ? `${value}% (ETA ${remainingTime})` : `${value}%`)} 
                        style={{ height: "1.5rem" }} 
                      />
                    </div>
                  )}
                </TabPanel>
              </TabView>
            </div>
          </Card>
        </div>
      </div>
    </div>
  );
};

export default T2ImgEdit;