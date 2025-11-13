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
import { TabView, TabPanel } from "primereact/tabview";
import { ToggleButton } from "primereact/togglebutton";
import { ColorPicker } from "primereact/colorpicker";
import { MultiSelect } from "primereact/multiselect";
import { Slider } from "primereact/slider";
import { RadioButton } from "primereact/radiobutton";
import "primereact/resources/themes/lara-light-indigo/theme.css";
import "primereact/resources/primereact.min.css";
import "primeicons/primeicons.css";
import "./DiagramChat.css";

const DiagramChat = () => {
  // Diagram states
  const [originalDiagram, setOriginalDiagram] = useState(null);
  const [editedDiagram, setEditedDiagram] = useState(null);
  const [activeDiagramType, setActiveDiagramType] = useState('original'); // 'original' or 'edited'
  const [diagramCode, setDiagramCode] = useState("");
  const [editedDiagramCode, setEditedDiagramCode] = useState("");
  
  // Input states
  const [scientificText, setScientificText] = useState("");
  const [editInstructions, setEditInstructions] = useState("");
  const [selectedDiagramType, setSelectedDiagramType] = useState("graphviz");  // Changed default to graphviz
  const [selectedDiagramStyle, setSelectedDiagramStyle] = useState("digraph");  // Changed default to digraph
  
  // Style customization states
  const [primaryColor, setPrimaryColor] = useState("#3498db");
  const [secondaryColor, setSecondaryColor] = useState("#2ecc71");
  const [backgroundColor, setBackgroundColor] = useState("#ffffff");
  const [fontFamily, setFontFamily] = useState("Arial");
  const [fontSize, setFontSize] = useState(12);
  const [lineWidth, setLineWidth] = useState(2);
  const [nodeShape, setNodeShape] = useState("rectangle");
  const [layoutDirection, setLayoutDirection] = useState("TB");
  const [showGrid, setShowGrid] = useState(false);
  const [showLegend, setShowLegend] = useState(true);
  
  // PNG export settings
  const [pngScale, setPngScale] = useState(3); // Scale factor for high-res PNG
  const [pngQuality, setPngQuality] = useState(0.95); // PNG quality
  
  // UI states
  const [fileName, setFileName] = useState("");
  const [isGenerating, setIsGenerating] = useState(false);
  const [isEditing, setIsEditing] = useState(false);
  const [progress, setProgress] = useState(0);
  const [remainingTime, setRemainingTime] = useState("");
  const [error, setError] = useState("");
  const [selectedPort, setSelectedPort] = useState(8003); // Port for diagram API
  const [selectedApiUrl, setSelectedApiUrl] = useState("http://127.0.0.1");
  const [activeTabIndex, setActiveTabIndex] = useState(0);
  const [selectedLLMModel, setSelectedLLMModel] = useState("codegen-2b"); // Fixed default to match API expectation

  const wsRef = useRef(null);
  const cancelledRef = useRef(false);
  const fileUploadRef = useRef(null);
  const clientTimeoutRef = useRef(null); // Add client timeout reference
  const canvasRef = useRef(null); // Reference for canvas element used in PNG conversion

  const apiUrlOptions = [
    { label: "Local", value: "http://127.0.0.1" },
    { label: "Remote", value: "http://195.230.127.227" },
  ];

  const getPortOptions = () => {
    if (selectedApiUrl === "http://127.0.0.1") {
      return [
        { label: "8003 (Diagram API)", value: 8003 },
        { label: "8002 (Image Edit API)", value: 8002 },
        { label: "8001 (T2I with LoRA)", value: 8001 },
        { label: "8000 (Direct T2I)", value: 8000 },
      ];
    } else if (selectedApiUrl === "http://195.230.127.227") {
      return [
        { label: "8283 (Diagram API)", value: 8283 },
        { label: "8281 (FLUX-Dev.1/EDIT)", value: 8281 },
        { label: "8280 (FLUX-Dev.1)", value: 8280 }
      ];
    }
    return [];
  };

  const diagramTypeOptions = [
    { label: "GraphViz", value: "graphviz", icon: "pi pi-share-alt" }
  ];

  const diagramStyleOptions = {
    graphviz: [
      { label: "Directed Graph", value: "digraph" },
      { label: "Hierarchical", value: "hierarchical" },
      { label: "Network", value: "network" },
      { label: "Process", value: "process" },
      { label: "Conceptual", value: "conceptual" }
    ]
  };

  // Updated LLM model options to match API expectations
  const llmModelOptions = [
    { label: "CodeGen 2B (Recommended)", value: "codegen-2b", icon: "pi pi-bolt" },
    { label: "CodeGen 350M (Fast)", value: "codegen-350m", icon: "pi pi-flash" },
    { label: "DeepSeek Coder (Advanced)", value: "deepseek-coder", icon: "pi pi-code" },
    { label: "DialoGPT Medium", value: "diaglo-gpt", icon: "pi pi-comments" },
    { label: "DistilGPT2 (Fallback)", value: "distilgpt2", icon: "pi pi-circle" },
    { label: "WizardCoder", value: "wizardcoder", icon: "pi pi-magic-wand" },
    { label: "SciGPT + Coder", value: "sci-coder", icon: "pi pi-chart-bar" }
  ];

  const fontOptions = [
    { label: "Arial", value: "Arial" },
    { label: "Times New Roman", value: "Times New Roman" },
    { label: "Helvetica", value: "Helvetica" },
    { label: "Roboto", value: "Roboto" },
    { label: "Open Sans", value: "Open Sans" }
  ];

  const nodeShapeOptions = [
    { label: "Rectangle", value: "rectangle" },
    { label: "Circle", value: "circle" },
    { label: "Ellipse", value: "ellipse" },
    { label: "Diamond", value: "diamond" },
    { label: "Hexagon", value: "hexagon" }
  ];

  const layoutOptions = [
    { label: "Top to Bottom", value: "TB" },
    { label: "Bottom to Top", value: "BT" },
    { label: "Left to Right", value: "LR" },
    { label: "Right to Left", value: "RL" }
  ];

  const pngScaleOptions = [
    { label: "1x (Standard)", value: 1 },
    { label: "2x (High)", value: 2 },
    { label: "3x (Very High)", value: 3 },
    { label: "4x (Ultra)", value: 4 },
    { label: "5x (Maximum)", value: 5 }
  ];

  useEffect(() => {
    return () => {
      cleanupWebSocket();
      if (originalDiagram) URL.revokeObjectURL(originalDiagram);
      if (editedDiagram) URL.revokeObjectURL(editedDiagram);
      if (clientTimeoutRef.current) {
        clearTimeout(clientTimeoutRef.current);
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
    // Clear client timeout
    if (clientTimeoutRef.current) {
      clearTimeout(clientTimeoutRef.current);
      clientTimeoutRef.current = null;
    }
  };

  const handleCancel = () => {
    cancelledRef.current = true;
    cleanupWebSocket();
    setIsGenerating(false);
    setIsEditing(false);
    setProgress(0);
    setRemainingTime("");
    setError("Operation cancelled by user");
  };

  const getApiUrl = (endpoint) => `${selectedApiUrl}:${selectedPort}${endpoint}`;
  const getWebSocketUrl = (wsEndpoint) => `${selectedApiUrl.replace("http", "ws")}:${selectedPort}${wsEndpoint}`;

  // Handle file upload (for text files or existing diagrams)
  const handleFileUpload = (event) => {
    const file = event.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (e) => {
        const content = e.target.result;
        if (file.type.startsWith('text/') || file.name.endsWith('.txt') || file.name.endsWith('.md')) {
          setScientificText(content);
          console.log("📄 File uploaded:", file.name, "Content length:", content.length);
        } else if (file.name.endsWith('.mmd') || file.name.endsWith('.dot') || file.name.endsWith('.puml')) {
          setDiagramCode(content);
          console.log("📊 Diagram file uploaded:", file.name);
        }
        setError("");
      };
      reader.readAsText(file);
    }
    fileUploadRef.current.clear();
  };

  // Convert SVG to high-resolution PNG
  const convertSvgToPng = async (svgUrl, scale = 3, quality = 0.95) => {
    return new Promise((resolve, reject) => {
      try {
        console.log(`🖼️ Converting SVG to PNG at ${scale}x scale...`);
        
        // Create a new image element
        const img = new Image();
        img.crossOrigin = 'anonymous';
        
        img.onload = () => {
          try {
            // Create canvas with scaled dimensions
            const canvas = document.createElement('canvas');
            const ctx = canvas.getContext('2d');
            
            // Set canvas size with scale factor
            canvas.width = img.naturalWidth * scale;
            canvas.height = img.naturalHeight * scale;
            
            // Set high-quality rendering
            ctx.imageSmoothingEnabled = true;
            ctx.imageSmoothingQuality = 'high';
            
            // Fill background with white (important for PNG transparency)
            ctx.fillStyle = backgroundColor || '#ffffff';
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            
            // Draw the scaled image
            ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
            
            // Convert to PNG blob
            canvas.toBlob((blob) => {
              if (blob) {
                console.log(`✅ PNG conversion successful: ${canvas.width}x${canvas.height}px`);
                resolve(blob);
              } else {
                reject(new Error('Failed to convert canvas to blob'));
              }
            }, 'image/png', quality);
            
          } catch (error) {
            console.error('❌ Canvas rendering error:', error);
            reject(error);
          }
        };
        
        img.onerror = (error) => {
          console.error('❌ Image loading error:', error);
          reject(new Error('Failed to load SVG image'));
        };
        
        // Load the SVG
        img.src = svgUrl;
        
      } catch (error) {
        console.error('❌ SVG to PNG conversion error:', error);
        reject(error);
      }
    });
  };

  // Generate diagram from scientific text using WebSocket
  const handleGenerateDiagram = async () => {
    if (!scientificText.trim()) {
      setError("Please enter scientific text or upload a document");
      return;
    }
  
    console.log("🔍 Generation Debug Info:");
    console.log("- Selected LLM Model:", selectedLLMModel);
    console.log("- API URL:", getWebSocketUrl("/generate-diagram"));
    console.log("- Scientific Text Length:", scientificText.length);
    console.log("- Diagram Type:", selectedDiagramType);
    console.log("- Diagram Style:", selectedDiagramStyle);
  
    setIsGenerating(true);
    setProgress(0);
    setRemainingTime("");
    setError("");
    cancelledRef.current = false;

    // Add client-side timeout (3 minutes)
    clientTimeoutRef.current = setTimeout(() => {
      if (wsRef.current && !cancelledRef.current) {
        console.log("⏰ Client timeout reached");
        setError("Generation timed out. The model might be overloaded or the text too complex. Try with simpler text or a smaller model.");
        handleCancel();
      }
    }, 180000); // 3 minutes
  
    try {
      cleanupWebSocket();
      const wsEndpoint = "/generate-diagram";
      const ws = new WebSocket(getWebSocketUrl(wsEndpoint));
      wsRef.current = ws;
  
      ws.onopen = () => {
        console.log(`✅ WebSocket connected to ${getWebSocketUrl(wsEndpoint)}`);
        if (ws.readyState === WebSocket.OPEN && !cancelledRef.current) {
          const requestData = {
            scientific_text: scientificText,
            diagram_type: selectedDiagramType,
            diagram_style: selectedDiagramStyle,
            llm_model: selectedLLMModel,  // This should now correctly pass the model
            style_preferences: {
              primary_color: primaryColor,
              secondary_color: secondaryColor,
              background_color: backgroundColor,
              font_family: fontFamily,
              font_size: fontSize,
              line_width: lineWidth,
              node_shape: nodeShape,
              layout_direction: layoutDirection,
              show_grid: showGrid,
              show_legend: showLegend
            }
          };
          
          console.log("📤 Sending WebSocket data:");
          console.log("   - Scientific text preview:", scientificText.substring(0, 100) + "...");
          console.log("   - LLM Model:", requestData.llm_model);
          console.log("   - Diagram config:", `${requestData.diagram_type}/${requestData.diagram_style}`);
          ws.send(JSON.stringify(requestData));
        }
      };
      
      ws.onmessage = (event) => {
        if (cancelledRef.current) return;
        // Clear timeout on any message received
        if (clientTimeoutRef.current) {
          clearTimeout(clientTimeoutRef.current);
          clientTimeoutRef.current = null;
        }
        handleWebSocketMessage(event, () => fetchGeneratedDiagram());
      };

      ws.onerror = (err) => {
        if (cancelledRef.current) return;
        console.error("❌ WebSocket error:", err);
        setError(`Failed to connect to ${getWebSocketUrl(wsEndpoint)}. Please check if the API server is running.`);
        cleanupWebSocket();
        setIsGenerating(false);
      };

      ws.onclose = (event) => {
        console.log("🔌 WebSocket closed:", event.code, event.reason);
        if (wsRef.current === ws) wsRef.current = null;
        // Clear timeout when connection closes
        if (clientTimeoutRef.current) {
          clearTimeout(clientTimeoutRef.current);
          clientTimeoutRef.current = null;
        }
      };
    } catch (err) {
      if (cancelledRef.current) return;
      console.error("❌ WebSocket setup error:", err);
      setError(`Failed to setup WebSocket connection: ${err.message}`);
      cleanupWebSocket();
      setIsGenerating(false);
    }
  };

  // Edit existing diagram using WebSocket
  const handleEditDiagram = async () => {
    if (!diagramCode.trim() && !originalDiagram) {
      setError("Please generate a diagram first or provide diagram code");
      return;
    }
    if (!editInstructions.trim()) {
      setError("Please enter edit instructions");
      return;
    }

    console.log("✏️ Edit Debug Info:");
    console.log("- Edit instructions:", editInstructions);
    console.log("- Current diagram code length:", diagramCode.length);
    console.log("- Selected LLM Model:", selectedLLMModel);

    setIsEditing(true);
    setProgress(0);
    setRemainingTime("");
    setError("");
    cancelledRef.current = false;

    // Add client-side timeout for editing too
    clientTimeoutRef.current = setTimeout(() => {
      if (wsRef.current && !cancelledRef.current) {
        console.log("⏰ Edit timeout reached");
        setError("Edit operation timed out. Try with simpler instructions.");
        handleCancel();
      }
    }, 120000); // 2 minutes for editing

    try {
      cleanupWebSocket();
      const wsEndpoint = "/edit-diagram";
      const ws = new WebSocket(getWebSocketUrl(wsEndpoint));
      wsRef.current = ws;

      ws.onopen = () => {
        console.log(`✅ Edit WebSocket connected to ${getWebSocketUrl(wsEndpoint)}`);
        if (ws.readyState === WebSocket.OPEN && !cancelledRef.current) {
          const requestData = {
            current_diagram_code: diagramCode,
            edit_instructions: editInstructions,
            diagram_type: selectedDiagramType,
            llm_model: selectedLLMModel,
            style_preferences: {
              primary_color: primaryColor,
              secondary_color: secondaryColor,
              background_color: backgroundColor,
              font_family: fontFamily,
              font_size: fontSize,
              line_width: lineWidth,
              node_shape: nodeShape,
              layout_direction: layoutDirection,
              show_grid: showGrid,
              show_legend: showLegend
            }
          };
          console.log("📤 Sending edit request:", requestData);
          ws.send(JSON.stringify(requestData));
        }
      };

      ws.onmessage = (event) => {
        if (cancelledRef.current) return;
        // Clear timeout on message received
        if (clientTimeoutRef.current) {
          clearTimeout(clientTimeoutRef.current);
          clientTimeoutRef.current = null;
        }
        handleWebSocketMessage(event, () => fetchEditedDiagram());
      };

      ws.onerror = (err) => {
        if (cancelledRef.current) return;
        console.error("❌ Edit WebSocket error:", err);
        setError(`Failed to connect to ${getWebSocketUrl(wsEndpoint)}`);
        cleanupWebSocket();
        setIsEditing(false);
      };

      ws.onclose = (event) => {
        console.log("🔌 Edit WebSocket closed:", event.code, event.reason);
        if (wsRef.current === ws) wsRef.current = null;
        if (clientTimeoutRef.current) {
          clearTimeout(clientTimeoutRef.current);
          clientTimeoutRef.current = null;
        }
      };
    } catch (err) {
      if (cancelledRef.current) return;
      console.error("❌ Edit WebSocket setup error:", err);
      setError(`Failed to setup edit WebSocket connection: ${err.message}`);
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

    console.log("📥 WebSocket message:", payload);

    if (payload.type === "progress") {
      const p = Number(payload.progress) || 0;
      setProgress(p);
      console.log(`📊 Progress: ${p}% - ${payload.message || ''}`);
      
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
      console.log("✅ Generation/Edit completed successfully");
      setProgress(100);
      setRemainingTime("00:00");
      cleanupWebSocket();
      onComplete();
    } else if (payload.type === "error") {
      console.error("❌ Server error:", payload.message);
      setError(`Server Error: ${payload.message || "Unknown error occurred"}`);
      cleanupWebSocket();
      setIsGenerating(false);
      setIsEditing(false);
    } else if (payload.type === "text") {
      console.log("📝 Status update:", payload.message);
    }
  };

  // Fetch generated diagram
  const fetchGeneratedDiagram = async () => {
    if (cancelledRef.current) return;
    try {
      console.log("📊 Fetching generated diagram...");
      await new Promise((r) => setTimeout(r, 1000)); // Brief delay
      if (cancelledRef.current) return;
      
      const response = await axios.post(getApiUrl("/get-diagram"), {
        diagram_type: selectedDiagramType
      }, {
        timeout: 30000,
      });
      
      if (cancelledRef.current) return;
      
      console.log("📥 Diagram response received:", {
        hasCode: !!response.data.diagram_code,
        hasImage: !!response.data.rendered_image,
        codeLength: response.data.diagram_code?.length || 0
      });
      
      // Handle both image and code responses
      if (response.data.diagram_code) {
        setDiagramCode(response.data.diagram_code);
        console.log("💾 Diagram code saved");
      }
      
      if (response.data.rendered_image) {
        // Convert base64 to blob if needed
        const imageBlob = await fetch(`data:image/svg+xml;base64,${response.data.rendered_image}`).then(r => r.blob());
        if (originalDiagram) URL.revokeObjectURL(originalDiagram);
        const imageUrl = URL.createObjectURL(imageBlob);
        setOriginalDiagram(imageUrl);
        setActiveDiagramType('original');
        console.log("🖼️ Diagram image rendered and saved");
      }
      
      setProgress(100);
      console.log("✅ Diagram fetch completed successfully");
    } catch (err) {
      if (cancelledRef.current) return;
      console.error("❌ Error fetching diagram:", err);
      setError(`Failed to fetch the generated diagram: ${err.response?.data?.detail || err.message}`);
    } finally {
      if (!cancelledRef.current) setIsGenerating(false);
    }
  };

  // Fetch edited diagram
  const fetchEditedDiagram = async () => {
    if (cancelledRef.current) return;
    try {
      console.log("📊 Fetching edited diagram...");
      await new Promise((r) => setTimeout(r, 1000));
      if (cancelledRef.current) return;
      
      const response = await axios.post(getApiUrl("/get-diagram"), {
        diagram_type: selectedDiagramType,
        is_edited: true
      }, {
        timeout: 30000,
      });
      
      if (cancelledRef.current) return;
      
      console.log("📥 Edited diagram response received");
      
      if (response.data.diagram_code) {
        setEditedDiagramCode(response.data.diagram_code);
        console.log("💾 Edited diagram code saved");
      }
      
      if (response.data.rendered_image) {
        const imageBlob = await fetch(`data:image/svg+xml;base64,${response.data.rendered_image}`).then(r => r.blob());
        if (editedDiagram) URL.revokeObjectURL(editedDiagram);
        const imageUrl = URL.createObjectURL(imageBlob);
        setEditedDiagram(imageUrl);
        setActiveDiagramType('edited');
        console.log("🖼️ Edited diagram image rendered and saved");
      }
      
      setProgress(100);
      console.log("✅ Edited diagram fetch completed successfully");
    } catch (err) {
      if (cancelledRef.current) return;
      console.error("❌ Error fetching edited diagram:", err);
      setError(`Failed to fetch the edited diagram: ${err.response?.data?.detail || err.message}`);
    } finally {
      if (!cancelledRef.current) setIsEditing(false);
    }
  };

  // Direct generation (without WebSocket) - Updated for better debugging
  const handleGenerateDirect = async () => {
    if (!scientificText.trim()) {
      setError("Please enter scientific text");
      return;
    }

    console.log("⚡ Direct generation started");
    console.log("- Model:", selectedLLMModel);
    console.log("- Text length:", scientificText.length);

    setIsGenerating(true);
    setProgress(0);
    setError("");
    cancelledRef.current = false;

    try {
      const requestData = {
        scientific_text: scientificText,
        diagram_type: selectedDiagramType,
        diagram_style: selectedDiagramStyle,
        llm_model: selectedLLMModel,
        style_preferences: {
          primary_color: primaryColor,
          secondary_color: secondaryColor,
          background_color: backgroundColor,
          font_family: fontFamily,
          font_size: fontSize,
          line_width: lineWidth,
          node_shape: nodeShape,
          layout_direction: layoutDirection,
          show_grid: showGrid,
          show_legend: showLegend
        }
      };

      console.log("📤 Direct API request data:", requestData);
      setProgress(30);

      const response = await axios.post(getApiUrl("/generate-diagram-direct"), requestData, {
        timeout: 120000, // 2 minutes
      });

      if (cancelledRef.current) return;
      
      console.log("📥 Direct API response received");
      setProgress(70);
      
      if (response.data.diagram_code) {
        setDiagramCode(response.data.diagram_code);
        console.log("💾 Direct: Diagram code saved");
      }
      
      if (response.data.rendered_image) {
        const imageBlob = await fetch(`data:image/svg+xml;base64,${response.data.rendered_image}`).then(r => r.blob());
        if (originalDiagram) URL.revokeObjectURL(originalDiagram);
        const imageUrl = URL.createObjectURL(imageBlob);
        setOriginalDiagram(imageUrl);
        setActiveDiagramType('original');
        console.log("🖼️ Direct: Diagram image rendered");
      }
      
      setProgress(100);
      console.log("✅ Direct generation completed successfully");
    } catch (err) {
      if (cancelledRef.current) return;
      console.error("❌ Direct generation error:", err);
      setError(`Failed to generate diagram: ${err.response?.data?.detail || err.message}`);
    } finally {
      if (!cancelledRef.current) setIsGenerating(false);
    }
  };

  // Enhanced save diagram function with PNG support
  const handleSave = async (format = 'svg') => {
    const diagramToSave = activeDiagramType === 'edited' ? editedDiagram : originalDiagram;
    const codeToSave = activeDiagramType === 'edited' ? editedDiagramCode : diagramCode;
    
    if (!diagramToSave && !codeToSave) {
      setError("No diagram to save");
      return;
    }
    
    const finalFileName = fileName.trim() || `${activeDiagramType}_diagram`;
    
    try {
      if (format === 'code' && codeToSave) {
        // Save diagram code
        const blob = new Blob([codeToSave], { type: 'text/plain' });
        const link = document.createElement("a");
        link.href = URL.createObjectURL(blob);
        link.download = `${finalFileName}.${selectedDiagramType}`;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        URL.revokeObjectURL(link.href);
        console.log("💾 Diagram code saved:", `${finalFileName}.${selectedDiagramType}`);
        
      } else if (format === 'png' && diagramToSave) {
        // Convert SVG to high-resolution PNG
        console.log(`🖼️ Converting to PNG at ${pngScale}x scale...`);
        setError(""); // Clear any previous errors
        
        try {
          const pngBlob = await convertSvgToPng(diagramToSave, pngScale, pngQuality);
          
          const link = document.createElement("a");
          link.href = URL.createObjectURL(pngBlob);
          link.download = `${finalFileName}_${pngScale}x.png`;
          document.body.appendChild(link);
          link.click();
          document.body.removeChild(link);
          URL.revokeObjectURL(link.href);
          
          console.log(`✅ High-res PNG saved: ${finalFileName}_${pngScale}x.png`);
          
        } catch (pngError) {
          console.error("❌ PNG conversion failed:", pngError);
          setError(`Failed to convert to PNG: ${pngError.message}. Try using SVG format instead.`);
          return;
        }
        
      } else if (diagramToSave) {
        // Save as SVG (original format)
        const link = document.createElement("a");
        link.href = diagramToSave;
        link.download = `${finalFileName}.${format}`;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        console.log("💾 Diagram image saved:", `${finalFileName}.${format}`);
      }
      
      setError("");
      
    } catch (err) {
      console.error("❌ Save error:", err);
      setError(`Failed to save diagram: ${err.message}`);
    }
  };

  // Use edited diagram as new original
  const handleUseAsOriginal = () => {
    if (!editedDiagram && !editedDiagramCode) return;
    
    console.log("🔄 Using edited diagram as new original");
    
    if (originalDiagram) URL.revokeObjectURL(originalDiagram);
    setOriginalDiagram(editedDiagram);
    setDiagramCode(editedDiagramCode);
    setEditedDiagram(null);
    setEditedDiagramCode("");
    setActiveDiagramType('original');
    setEditInstructions("");
  };

  const getModelInfo = () => {
    const portInfo = getPortOptions().find((option) => option.value === selectedPort);
    const modelInfo = llmModelOptions.find((option) => option.value === selectedLLMModel);
    return portInfo ? `${portInfo.label} (${modelInfo?.label || selectedLLMModel})` : `Port ${selectedPort} (${modelInfo?.label || selectedLLMModel})`;
  };

  const getCurrentDiagram = () => {
    return activeDiagramType === 'edited' ? editedDiagram : originalDiagram;
  };

  const getCurrentCode = () => {
    return activeDiagramType === 'edited' ? editedDiagramCode : diagramCode;
  };

  const diagramDisplay = () => (
    <div className="diagram-display-container">
      {getCurrentDiagram() ? (
        <div style={{ textAlign: "center" }}>
          <div className="diagram-preview" style={{ 
            maxWidth: "100%", 
            maxHeight: "400px", 
            overflow: "auto",
            border: "1px solid #ddd",
            borderRadius: "8px",
            backgroundColor: backgroundColor,
            padding: "1rem"
          }}>
            <img 
              src={getCurrentDiagram()} 
              alt={activeDiagramType} 
              style={{ maxWidth: "100%", height: "auto" }} 
            />
          </div>
          <div className="diagram-type-buttons" style={{ marginTop: "1rem" }}>
            {originalDiagram && (
              <Button 
                label="Original" 
                className={`p-button-sm ${activeDiagramType === 'original' ? 'p-button-primary' : 'p-button-outlined'}`}
                onClick={() => setActiveDiagramType('original')}
                style={{ marginRight: "0.5rem" }}
              />
            )}
            {editedDiagram && (
              <Button 
                label="Edited" 
                className={`p-button-sm ${activeDiagramType === 'edited' ? 'p-button-primary' : 'p-button-outlined'}`}
                onClick={() => setActiveDiagramType('edited')}
              />
            )}
          </div>
        </div>
      ) : (
        <div className="empty-diagram-state" style={{ textAlign: "center", padding: "2rem", color: "#666" }}>
          <i className="pi pi-sitemap" style={{ fontSize: "3rem", color: "#ccc", marginBottom: "1rem" }}></i>
          <p style={{ margin: "0.5rem 0", fontSize: "1.1rem" }}>No diagram yet</p>
          <small>Enter scientific text and generate a diagram</small>
        </div>
      )}
    </div>
  );

  const codeDisplay = () => (
    <div className="code-display-container">
      {getCurrentCode() ? (
        <div style={{ textAlign: "left" }}>
          <div className="p-field">
            <label>Diagram Code ({selectedDiagramType.toUpperCase()})</label>
            <InputTextarea
              value={getCurrentCode()}
              onChange={(e) => {
                if (activeDiagramType === 'edited') {
                  setEditedDiagramCode(e.target.value);
                } else {
                  setDiagramCode(e.target.value);
                }
              }}
              rows={15}
              style={{ 
                fontFamily: "monospace", 
                fontSize: "12px",
                backgroundColor: "#f8f9fa"
              }}
            />
          </div>
        </div>
      ) : (
        <div className="empty-code-state" style={{ textAlign: "center", padding: "2rem", color: "#666" }}>
          <i className="pi pi-code" style={{ fontSize: "3rem", color: "#ccc", marginBottom: "1rem" }}></i>
          <p style={{ margin: "0.5rem 0", fontSize: "1.1rem" }}>No code yet</p>
          <small>Generate a diagram to see the code</small>
        </div>
      )}
    </div>
  );

  return (
    <div className="diagram-chat-container">
      <div className="p-grid p-justify-center p-align-start p-mt-4" style={{ padding: "1rem" }}>
        <div className="p-col-12 p-md-6 diagram-panel">
          <Card title="Scientific Diagram Generator" style={{ height: "100%" }}>
            <TabView>
              <TabPanel header="Preview" leftIcon="pi pi-eye">
                {diagramDisplay()}
              </TabPanel>
              <TabPanel header="Code" leftIcon="pi pi-code">
                {codeDisplay()}
              </TabPanel>
            </TabView>
            
            <Divider />
            
            {editedDiagram && (
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
                <InputText 
                  id="fileName" 
                  value={fileName} 
                  onChange={(e) => setFileName(e.target.value)} 
                  placeholder={`${activeDiagramType}_diagram`} 
                  style={{ marginBottom: "0.5rem" }}
                />
              </div>

              {/* PNG Export Settings */}
              <div className="p-field">
                <label>PNG Export Settings</label>
                <div className="p-grid" style={{ margin: 0 }}>
                  <div className="p-col-8" style={{ paddingRight: "0.25rem" }}>
                    <label htmlFor="pngScale" style={{ fontSize: "0.9rem" }}>Scale: {pngScale}x</label>
                    <Dropdown 
                      id="pngScale" 
                      value={pngScale} 
                      onChange={(e) => setPngScale(e.value)} 
                      options={pngScaleOptions} 
                      placeholder="Select Scale" 
                    />
                  </div>
                  <div className="p-col-4" style={{ paddingLeft: "0.25rem" }}>
                    <label htmlFor="pngQuality" style={{ fontSize: "0.9rem" }}>Quality: {Math.round(pngQuality * 100)}%</label>
                    <Slider 
                      id="pngQuality" 
                      value={pngQuality} 
                      onChange={(e) => setPngQuality(e.value)} 
                      min={0.5} 
                      max={1.0} 
                      step={0.05}
                    />
                  </div>
                </div>
              </div>

              {/* Download Buttons */}
              <div className="p-field">
                <div className="p-grid" style={{ margin: 0 }}>
                  <div className="p-col-4" style={{ paddingRight: "0.125rem" }}>
                    <Button 
                      icon="pi pi-download" 
                      label="SVG" 
                      className="p-button-success p-button-sm" 
                      onClick={() => handleSave('svg')} 
                      disabled={!getCurrentDiagram()} 
                      style={{ width: "100%" }}
                    />
                  </div>
                  <div className="p-col-4" style={{ padding: "0 0.125rem" }}>
                    <Button 
                      icon="pi pi-image" 
                      label={`PNG ${pngScale}x`} 
                      className="p-button-info p-button-sm" 
                      onClick={() => handleSave('png')} 
                      disabled={!getCurrentDiagram()} 
                      style={{ width: "100%" }}
                    />
                  </div>
                  <div className="p-col-4" style={{ paddingLeft: "0.125rem" }}>
                    <Button 
                      icon="pi pi-file" 
                      label="Code" 
                      className="p-button-secondary p-button-sm" 
                      onClick={() => handleSave('code')} 
                      disabled={!getCurrentCode()} 
                      style={{ width: "100%" }}
                    />
                  </div>
                </div>
              </div>
            </div>
          </Card>
        </div>

        <div className="p-col-12 p-md-6 controls-panel">
          <Card title={`GraphViz Generator - ${getModelInfo()}`}>
            <div className="p-fluid">
              {error && <Message severity="error" text={error} style={{ marginBottom: "1rem" }} />}

              <div className="p-grid">
                <div className="p-col-4">
                  <div className="p-field">
                    <label htmlFor="apiUrl">API URL</label>
                    <Dropdown 
                      id="apiUrl" 
                      value={selectedApiUrl} 
                      onChange={(e) => { 
                        console.log("🔗 API URL changed to:", e.value);
                        setSelectedApiUrl(e.value); 
                        if (e.value === "http://127.0.0.1") {
                          setSelectedPort(8003);
                        } else if (e.value === "http://195.230.127.227") {
                          setSelectedPort(8283);
                        }
                      }} 
                      options={apiUrlOptions} 
                      placeholder="Select API URL" 
                      disabled={isGenerating || isEditing} 
                    />
                  </div>
                </div>
                <div className="p-col-4">
                  <div className="p-field">
                    <label htmlFor="port">API Port</label>
                    <Dropdown 
                      id="port" 
                      value={selectedPort} 
                      onChange={(e) => {
                        console.log("🔌 API Port changed to:", e.value);
                        setSelectedPort(e.value);
                      }} 
                      options={getPortOptions()} 
                      placeholder="Select API Port" 
                      disabled={isGenerating || isEditing} 
                    />
                  </div>
                </div>
                <div className="p-col-4">
                  <div className="p-field">
                    <label htmlFor="llmModel">LLM Model</label>
                    <Dropdown 
                      id="llmModel" 
                      value={selectedLLMModel} 
                      onChange={(e) => {
                        console.log("🤖 LLM Model changed to:", e.value);
                        setSelectedLLMModel(e.value);
                      }} 
                      options={llmModelOptions} 
                      placeholder="Select LLM Model" 
                      disabled={isGenerating || isEditing} 
                    />
                  </div>
                </div>
              </div>

              <TabView activeIndex={activeTabIndex} onTabChange={(e) => setActiveTabIndex(e.index)}>
                <TabPanel header="Generate" leftIcon="pi pi-plus">
                  <div className="p-field">
                    <label htmlFor="fileUpload">Upload Document (Optional)</label>
                    <FileUpload
                      ref={fileUploadRef}
                      mode="basic"
                      name="document"
                      accept=".txt,.md,.doc,.docx,.pdf"
                      maxFileSize={5000000}
                      onSelect={handleFileUpload}
                      auto
                      chooseLabel="Choose Document"
                      className="p-button-outlined"
                    />
                  </div>

                  <div className="p-field">
                    <label htmlFor="scientificText">Scientific Text *</label>
                    <InputTextarea 
                      id="scientificText" 
                      value={scientificText} 
                      onChange={(e) => setScientificText(e.target.value)} 
                      rows={6} 
                      autoResize 
                      placeholder="Enter detailed scientific text: research methodology, process flow, experimental procedure, data analysis steps, etc. Be specific about the relationships and sequence of steps for better diagram generation."
                      disabled={isGenerating}
                    />
                    <small className="p-text-secondary">
                      💡 Tip: More detailed text with clear process steps produces better diagrams
                    </small>
                  </div>

                  <div className="p-grid">
                    <div className="p-col-6">
                      <div className="p-field">
                        <label htmlFor="diagramType">Diagram Type</label>
                        <Dropdown 
                          id="diagramType" 
                          value={selectedDiagramType} 
                          onChange={(e) => {
                            console.log("📊 Diagram type changed to:", e.value);
                            setSelectedDiagramType(e.value);
                          }} 
                          options={diagramTypeOptions} 
                          placeholder="Select Diagram Type" 
                          disabled={isGenerating}
                        />
                      </div>
                    </div>
                    <div className="p-col-6">
                      <div className="p-field">
                        <label htmlFor="diagramStyle">Diagram Style</label>
                        <Dropdown 
                          id="diagramStyle" 
                          value={selectedDiagramStyle} 
                          onChange={(e) => {
                            console.log("🎨 Diagram style changed to:", e.value);
                            setSelectedDiagramStyle(e.value);
                          }} 
                          options={diagramStyleOptions[selectedDiagramType] || []} 
                          placeholder="Select Style" 
                          disabled={isGenerating}
                        />
                      </div>
                    </div>
                  </div>

                  <div className="p-field">
                    {isGenerating ? (
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
                      <div className="p-grid" style={{ margin: 0 }}>
                        <div className="p-col-6" style={{ paddingRight: "0.25rem" }}>
                          <Button 
                            label="Generate Diagram" 
                            icon="pi pi-sitemap" 
                            onClick={handleGenerateDiagram} 
                            className="p-button-success p-button-lg" 
                            style={{ width: "100%" }}
                            disabled={!scientificText.trim()}
                          />
                        </div>
                        <div className="p-col-6" style={{ paddingLeft: "0.25rem" }}>
                          <Button 
                            label="Direct" 
                            icon="pi pi-flash" 
                            onClick={handleGenerateDirect} 
                            className="p-button-secondary p-button-lg" 
                            style={{ width: "100%" }}
                            disabled={!scientificText.trim()}
                          />
                        </div>
                      </div>
                    )}
                  </div>
                </TabPanel>

                <TabPanel header="Edit" leftIcon="pi pi-pencil">
                  <div className="p-field">
                    <label htmlFor="editInstructions">Edit Instructions *</label>
                    <InputTextarea 
                      id="editInstructions" 
                      value={editInstructions} 
                      onChange={(e) => setEditInstructions(e.target.value)} 
                      rows={4} 
                      autoResize 
                      placeholder="Be specific about changes: 'Change all boxes to circles', 'Add a decision node after step 2', 'Change colors to blue theme', 'Add more detail to the analysis phase'..."
                      disabled={isEditing}
                    />
                    <small className="p-text-secondary">
                      💡 Tip: Be specific about what you want to change for better results
                    </small>
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
                      <Button 
                        label="Edit Diagram" 
                        icon="pi pi-pencil" 
                        onClick={handleEditDiagram} 
                        className="p-button-warning p-button-lg" 
                        style={{ width: "100%" }}
                        disabled={(!diagramCode && !originalDiagram) || !editInstructions.trim()}
                      />
                    )}
                  </div>
                </TabPanel>

                <TabPanel header="Style" leftIcon="pi pi-palette">
                  <div className="p-grid">
                    <div className="p-col-4">
                      <div className="p-field">
                        <label>Primary Color</label>
                        <ColorPicker 
                          value={primaryColor.replace('#', '')} 
                          onChange={(e) => setPrimaryColor(`#${e.value}`)}
                          style={{ width: "100%" }}
                        />
                      </div>
                    </div>
                    <div className="p-col-4">
                      <div className="p-field">
                        <label>Secondary Color</label>
                        <ColorPicker 
                          value={secondaryColor.replace('#', '')} 
                          onChange={(e) => setSecondaryColor(`#${e.value}`)}
                          style={{ width: "100%" }}
                        />
                      </div>
                    </div>
                    <div className="p-col-4">
                      <div className="p-field">
                        <label>Background</label>
                        <ColorPicker 
                          value={backgroundColor.replace('#', '')} 
                          onChange={(e) => setBackgroundColor(`#${e.value}`)}
                          style={{ width: "100%" }}
                        />
                      </div>
                    </div>
                  </div>

                  <div className="p-grid">
                    <div className="p-col-6">
                      <div className="p-field">
                        <label htmlFor="fontFamily">Font Family</label>
                        <Dropdown 
                          id="fontFamily" 
                          value={fontFamily} 
                          onChange={(e) => setFontFamily(e.value)} 
                          options={fontOptions} 
                          placeholder="Select Font" 
                        />
                      </div>
                    </div>
                    <div className="p-col-6">
                      <div className="p-field">
                        <label htmlFor="fontSize">Font Size: {fontSize}px</label>
                        <Slider 
                          id="fontSize" 
                          value={fontSize} 
                          onChange={(e) => setFontSize(e.value)} 
                          min={8} 
                          max={24} 
                          step={1}
                        />
                      </div>
                    </div>
                  </div>

                  <div className="p-grid">
                    <div className="p-col-6">
                      <div className="p-field">
                        <label htmlFor="nodeShape">Node Shape</label>
                        <Dropdown 
                          id="nodeShape" 
                          value={nodeShape} 
                          onChange={(e) => setNodeShape(e.value)} 
                          options={nodeShapeOptions} 
                          placeholder="Select Shape" 
                        />
                      </div>
                    </div>
                    <div className="p-col-6">
                      <div className="p-field">
                        <label htmlFor="layoutDirection">Layout</label>
                        <Dropdown 
                          id="layoutDirection" 
                          value={layoutDirection} 
                          onChange={(e) => setLayoutDirection(e.value)} 
                          options={layoutOptions} 
                          placeholder="Select Layout" 
                        />
                      </div>
                    </div>
                  </div>

                  <div className="p-grid">
                    <div className="p-col-4">
                      <div className="p-field">
                        <label htmlFor="lineWidth">Line Width: {lineWidth}px</label>
                        <Slider 
                          id="lineWidth" 
                          value={lineWidth} 
                          onChange={(e) => setLineWidth(e.value)} 
                          min={1} 
                          max={5} 
                          step={0.5}
                        />
                      </div>
                    </div>
                    <div className="p-col-4">
                      <div className="p-field-checkbox">
                        <ToggleButton
                          checked={showGrid} 
                          onChange={(e) => setShowGrid(e.value)} 
                          onLabel="Grid ON" 
                          offLabel="Grid OFF"
                          onIcon="pi pi-th-large" 
                          offIcon="pi pi-th-large"
                        />
                      </div>
                    </div>
                    <div className="p-col-4">
                      <div className="p-field-checkbox">
                        <ToggleButton
                          checked={showLegend} 
                          onChange={(e) => setShowLegend(e.value)} 
                          onLabel="Legend ON" 
                          offLabel="Legend OFF"
                          onIcon="pi pi-list" 
                          offIcon="pi pi-list"
                        />
                      </div>
                    </div>
                  </div>
                </TabPanel>

                <TabPanel header="Settings" leftIcon="pi pi-cog">
                  {/* Progress bar with integrated cancel button */}
                  {(isGenerating || isEditing) && (
                    <div className="p-field">
                      <div className="p-grid" style={{ margin: 0, marginBottom: "1rem" }}>
                        <div className="p-col-8" style={{ paddingRight: "0.25rem" }}>
                          <label>Progress</label>
                          <ProgressBar 
                            value={progress} 
                            displayValueTemplate={(value) => (remainingTime ? `${value}% (ETA ${remainingTime})` : `${value}%`)} 
                            style={{ height: "1.5rem" }} 
                          />
                        </div>
                        <div className="p-col-4" style={{ paddingLeft: "0.25rem" }}>
                          <label>&nbsp;</label>
                          <Button 
                            label="Cancel" 
                            icon="pi pi-times" 
                            className="p-button-danger p-button-sm" 
                            onClick={handleCancel} 
                            style={{ width: "100%", height: "1.5rem", fontSize: "0.8rem" }} 
                          />
                        </div>
                      </div>
                    </div>
                  )}

                  <div className="p-field">
                    <label>Diagram Information</label>
                    <div style={{ backgroundColor: "#f8f9fa", padding: "1rem", borderRadius: "4px", fontSize: "0.9rem" }}>
                      <p style={{ margin: "0.25rem 0" }}><strong>Type:</strong> {selectedDiagramType.toUpperCase()}</p>
                      <p style={{ margin: "0.25rem 0" }}><strong>Style:</strong> {selectedDiagramStyle}</p>
                      <p style={{ margin: "0.25rem 0" }}><strong>Model:</strong> {selectedLLMModel}</p>
                      <p style={{ margin: "0.25rem 0" }}><strong>API:</strong> {selectedApiUrl}:{selectedPort}</p>
                      <p style={{ margin: "0.25rem 0" }}><strong>Status:</strong> {isGenerating ? "🔄 Generating..." : isEditing ? "✏️ Editing..." : "✅ Ready"}</p>
                      <p style={{ margin: "0.25rem 0" }}><strong>Text Length:</strong> {scientificText.length} chars</p>
                      <p style={{ margin: "0.25rem 0" }}><strong>PNG Export:</strong> {pngScale}x scale, {Math.round(pngQuality * 100)}% quality</p>
                    </div>
                  </div>

                  <div className="p-field">
                    <label>Quick Actions</label>
                    <div className="p-grid" style={{ margin: 0 }}>
                      <div className="p-col-6" style={{ paddingRight: "0.25rem" }}>
                        <Button 
                          label="Clear Text" 
                          icon="pi pi-trash" 
                          className="p-button-outlined p-button-sm" 
                          onClick={() => {
                            setScientificText("");
                            console.log("🗑️ Scientific text cleared");
                          }}
                          style={{ width: "100%" }}
                        />
                      </div>
                      <div className="p-col-6" style={{ paddingLeft: "0.25rem" }}>
                        <Button 
                          label="Reset All" 
                          icon="pi pi-refresh" 
                          className="p-button-outlined p-button-sm" 
                          onClick={() => {
                            setScientificText("");
                            setEditInstructions("");
                            setDiagramCode("");
                            setEditedDiagramCode("");
                            if (originalDiagram) URL.revokeObjectURL(originalDiagram);
                            if (editedDiagram) URL.revokeObjectURL(editedDiagram);
                            setOriginalDiagram(null);
                            setEditedDiagram(null);
                            setActiveDiagramType('original');
                            setError("");
                            console.log("🔄 All data reset");
                          }}
                          style={{ width: "100%" }}
                        />
                      </div>
                    </div>
                  </div>
                </TabPanel>
              </TabView>
            </div>
          </Card>
        </div>
      </div>
      
      {/* Hidden canvas for PNG conversion */}
      <canvas ref={canvasRef} style={{ display: 'none' }} />
    </div>
  );
};

export default DiagramChat;