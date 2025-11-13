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
import "primereact/resources/themes/lara-light-indigo/theme.css";
import "primereact/resources/primereact.min.css";
import "primeicons/primeicons.css";
import "./T2ImgChat.css";

const T2ImgChat = () => {
  const [prompt, setPrompt] = useState("");
  const [negativePrompt, setNegativePrompt] = useState("");
  const [numInferenceSteps, setNumInferenceSteps] = useState(20);
  const [guidanceScale, setGuidanceScale] = useState(7.5);
  const [width, setWidth] = useState(512);
  const [height, setHeight] = useState(512);
  const [fileName, setFileName] = useState("");
  const [image, setImage] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [remainingTime, setRemainingTime] = useState("");
  const [error, setError] = useState("");
  const [selectedPort, setSelectedPort] = useState(8000);
  const [selectedApiUrl, setSelectedApiUrl] = useState("http://127.0.0.1");

  const wsRef = useRef(null);
  const cancelledRef = useRef(false);

  const apiUrlOptions = [
    { label: "Local", value: "http://127.0.0.1" },
    { label: "Remote", value: "http://your-remote-server-ip" },
  ];

  const getPortOptions = () => {
    if (selectedApiUrl === "http://127.0.0.1") {
      return [
        { label: "8000 (Direct model)", value: 8000 },
        { label: "8001 (Model with LoRA)", value: 8001 },
      ];
    } else if (selectedApiUrl === "http://your-remote-server-ip") {
      return [{ label: "8280 (FLUX-Dev.1)", value: 8280 }];
    }
    return [];
  };

  useEffect(() => {
    return () => {
      cleanupWebSocket();
      if (image) {
        URL.revokeObjectURL(image);
      }
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
    setProgress(0);
    setRemainingTime("");
    setError("Generation cancelled by user");
  };

  const getApiUrl = (endpoint) => `${selectedApiUrl}:${selectedPort}${endpoint}`;
  const getWebSocketUrl = () => `${selectedApiUrl.replace("http", "ws")}:${selectedPort}/progress`;

  const handleGenerate = async () => {
    if (!prompt.trim()) {
      setError("Please enter a prompt");
      return;
    }

    setImage(null);
    setIsLoading(true);
    setProgress(0);
    setRemainingTime("");
    setError("");
    cancelledRef.current = false;

    try {
      cleanupWebSocket();
      const ws = new WebSocket(getWebSocketUrl());
      wsRef.current = ws;

      ws.onopen = () => {
        console.log(`WebSocket connected to ${getWebSocketUrl()}`);
        if (ws.readyState === WebSocket.OPEN && !cancelledRef.current) {
          const requestData = {
            prompt,
            negative_prompt: negativePrompt,
            num_inference_steps: numInferenceSteps,
            guidance_scale: guidanceScale,
            width,
            height,
          };
          if (selectedPort === 8001) requestData.lora_weight = 1.0;
          ws.send(JSON.stringify(requestData));
        }
      };

      ws.onmessage = (event) => {
        if (cancelledRef.current) return;

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
          } else {
            setRemainingTime("");
          }
        } else if (payload.type === "complete") {
          setProgress(100);
          setRemainingTime("00:00");
          cleanupWebSocket();
          fetchImage();
        } else if (payload.type === "error") {
          setError(`Error: ${payload.message || "unknown"}`);
          cleanupWebSocket();
          setIsLoading(false);
        } else if (payload.type === "text") {
          // legacy text messages
          if (typeof payload.message === "string" && payload.message.includes("Model loaded")) {
            // ignore
          } else if (payload.message && payload.message.startsWith("Progress:")) {
            const num = parseInt(payload.message.replace("Progress:", "").replace("%", "").split("|")[0].trim(), 10);
            setProgress(Number.isNaN(num) ? 0 : num);
          }
        }
      };

      ws.onerror = (err) => {
        if (cancelledRef.current) return;
        console.error("WebSocket error:", err);
        setError(`Failed to connect to ${getWebSocketUrl()}`);
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
      setError(`Failed to setup WebSocket connection on ${getWebSocketUrl()}`);
      cleanupWebSocket();
      setIsLoading(false);
    }
  };

  const fetchImage = async () => {
    if (cancelledRef.current) return;
    try {
      await new Promise((r) => setTimeout(r, 1000));
      if (cancelledRef.current) return;
      const requestData = {
        prompt,
        negative_prompt: negativePrompt,
        num_inference_steps: numInferenceSteps,
        guidance_scale: guidanceScale,
        width,
        height,
      };
      if (selectedPort === 8001) requestData.lora_weight = 1.0;
      const response = await axios.post(getApiUrl("/generate-image"), requestData, {
        responseType: "blob",
        timeout: 30000,
      });
      if (cancelledRef.current) return;
      if (image) URL.revokeObjectURL(image);
      const imageUrl = URL.createObjectURL(response.data);
      setImage(imageUrl);
      setProgress(100);
    } catch (err) {
      if (cancelledRef.current) return;
      console.error("Error fetching image:", err);
      setError(`Failed to fetch the generated image from ${getApiUrl("/generate-image")}`);
    } finally {
      if (!cancelledRef.current) setIsLoading(false);
    }
  };

  const handleSave = () => {
    if (!image) {
      setError("No image to save");
      return;
    }
    const finalFileName = fileName.trim() || "generated_image.png";
    const link = document.createElement("a");
    link.href = image;
    link.download = finalFileName.endsWith(".png") ? finalFileName : `${finalFileName}.png`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    setError("");
  };

  const getModelInfo = () => {
    const portInfo = getPortOptions().find((option) => option.value === selectedPort);
    return portInfo ? portInfo.label : `Port ${selectedPort}`;
  };

  const imageDisplay = () => (
    <div style={{ textAlign: "center", minHeight: "300px", display: "flex", alignItems: "center", justifyContent: "center" }}>
      {image ? (
        <img src={image} alt="Generated" style={{ maxWidth: "100%", maxHeight: "400px", borderRadius: "8px", boxShadow: "0 4px 6px rgba(0,0,0,0.1)" }} />
      ) : (
        <div style={{ color: "#6b7280", textAlign: "center" }}>
          <i className="pi pi-image" style={{ fontSize: "3rem", marginBottom: "1rem", display: "block" }}></i>
          <p>No image generated yet</p>
          <small>Click Generate to create an image</small>
        </div>
      )}
    </div>
  );

  return (
    <div className="p-grid p-justify-center p-align-start p-mt-4" style={{ padding: "1rem" }}>
      <div className="p-col-12 p-md-6 image-panel">
        <Card title="Generated Image" style={{ height: "100%" }}>
          {imageDisplay()}
          <Divider />
          <div className="p-fluid">
            <div className="p-field">
              <label htmlFor="fileName">File Name</label>
              <div className="p-inputgroup">
                <InputText id="fileName" value={fileName} onChange={(e) => setFileName(e.target.value)} placeholder="generated_image.png" />
                <Button icon="pi pi-download" label="Save" className="p-button-success" onClick={handleSave} disabled={!image} />
              </div>
            </div>
          </div>
        </Card>
      </div>

      <div className="p-col-12 p-md-6 controls-panel">
        <Card title={`Generate Image - ${getModelInfo()}`}>
          <div className="p-fluid">
            {error && <Message severity="error" text={error} style={{ marginBottom: "1rem" }} />}

            <div className="p-grid">
              <div className="p-col-6">
                <div className="p-field">
                  <label htmlFor="apiUrl">API URL</label>
                  <Dropdown id="apiUrl" value={selectedApiUrl} onChange={(e) => { setSelectedApiUrl(e.value); setSelectedPort(e.value === "http://127.0.0.1" ? 8000 : 8280); }} options={apiUrlOptions} placeholder="Select API URL" disabled={isLoading} />
                </div>
              </div>
              <div className="p-col-6">
                <div className="p-field">
                  <label htmlFor="port">API Port</label>
                  <Dropdown id="port" value={selectedPort} onChange={(e) => setSelectedPort(e.value)} options={getPortOptions()} placeholder="Select API Port" disabled={isLoading} />
                </div>
              </div>
            </div>

            <div className="p-field">
              <label htmlFor="prompt">Prompt *</label>
              <InputTextarea id="prompt" value={prompt} onChange={(e) => setPrompt(e.target.value)} rows={4} autoResize placeholder="Describe the image you want to generate..." />
            </div>

            <div className="p-field">
              <label htmlFor="negativePrompt">Negative Prompt</label>
              <InputTextarea id="negativePrompt" value={negativePrompt} onChange={(e) => setNegativePrompt(e.target.value)} rows={2} autoResize placeholder="What you don't want in the image..." />
            </div>

            <div className="p-grid">
              <div className="p-col-6">
                <div className="p-field">
                  <label htmlFor="numInferenceSteps">Inference Steps</label>
                  <InputNumber id="numInferenceSteps" value={numInferenceSteps} onValueChange={(e) => setNumInferenceSteps(e.value)} min={1} max={100} showButtons />
                </div>
              </div>
              <div className="p-col-6">
                <div className="p-field">
                  <label htmlFor="guidanceScale">Guidance Scale</label>
                  <InputNumber id="guidanceScale" value={guidanceScale} onValueChange={(e) => setGuidanceScale(e.value)} step={0.1} min={0.1} max={20} showButtons />
                </div>
              </div>
            </div>

            <div className="p-grid">
              <div className="p-col-6">
                <div className="p-field">
                  <label htmlFor="width">Width</label>
                  <InputNumber id="width" value={width} onValueChange={(e) => setWidth(e.value)} min={64} max={1024} step={64} showButtons />
                </div>
              </div>
              <div className="p-col-6">
                <div className="p-field">
                  <label htmlFor="height">Height</label>
                  <InputNumber id="height" value={height} onValueChange={(e) => setHeight(e.value)} min={64} max={1024} step={64} showButtons />
                </div>
              </div>
            </div>

            {isLoading && (
              <div className="p-field">
                <label>Generation Progress</label>
                <ProgressBar value={progress} displayValueTemplate={(value) => (remainingTime ? `${value}% (ETA ${remainingTime})` : `${value}%`)} style={{ height: "1.5rem" }} />
              </div>
            )}

            <div className="p-field">
              {isLoading ? (
                <div className="p-grid" style={{ margin: 0 }}>
                  <div className="p-col-8" style={{ paddingRight: "0.25rem" }}>
                    <Button label={`Generating... ${progress}%${remainingTime ? ` (ETA ${remainingTime})` : ""}`} icon="pi pi-spin pi-spinner" className="p-button-primary p-button-lg" disabled={true} style={{ width: "100%" }} />
                  </div>
                  <div className="p-col-4" style={{ paddingLeft: "0.25rem" }}>
                    <Button label="Cancel" icon="pi pi-times" className="p-button-danger p-button-lg" onClick={handleCancel} style={{ width: "100%" }} />
                  </div>
                </div>
              ) : (
                <Button label={`Generate Image (${getModelInfo()})`} icon="pi pi-bolt" onClick={handleGenerate} className="p-button-primary p-button-lg" style={{ width: "100%" }} />
              )}
            </div>
          </div>
        </Card>
      </div>
    </div>
  );
};

export default T2ImgChat;