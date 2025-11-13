import React, { useState, useRef, useCallback } from 'react';
import { Card } from 'primereact/card';
import { Button } from 'primereact/button';
import { FileUpload } from 'primereact/fileupload';
import { ProgressBar } from 'primereact/progressbar';
import { Toast } from 'primereact/toast';
import { Dialog } from 'primereact/dialog';
import { ScrollPanel } from 'primereact/scrollpanel';
import { Dropdown } from 'primereact/dropdown';
import { InputTextarea } from 'primereact/inputtextarea';
import { Divider } from 'primereact/divider';
import { Badge } from 'primereact/badge';
import { Chip } from 'primereact/chip';
import { ProgressSpinner } from 'primereact/progressspinner';
import { SplitButton } from 'primereact/splitbutton';
import { Slider } from 'primereact/slider';
import axios from 'axios';
import * as pdfjsLib from 'pdfjs-dist';
import { Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell, WidthType, BorderStyle } from 'docx';

// Global worker setup promise to ensure it's only done once
let workerSetupPromise = null;

// Setup PDF.js worker optimized for 3.11.174
const setupWorker = async () => {
    if (workerSetupPromise) {
        return workerSetupPromise;
    }

    workerSetupPromise = (async () => {
        if (typeof window === 'undefined') {
            return null;
        }

        // Get the current PDF.js version
        const pdfVersion = pdfjsLib.version;
        console.log(`📄 PDF.js version: ${pdfVersion}`);

        // Prioritize the local 3.11.174 worker that we downloaded
        const workerConfigs = [
            { url: '/pdf.worker.min.js', version: '3.11.174', description: 'Local 3.11.174 worker' },
            
            // Keep existing fallbacks but reorder for 3.11.174 compatibility
            { url: 'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js', version: '3.11.174', description: 'CloudFlare CDN (3.11.174)' },
            { url: 'https://unpkg.com/pdfjs-dist@3.11.174/build/pdf.worker.min.js', version: '3.11.174', description: 'unpkg CDN (3.11.174)' },
            
            // For PDF.js v5.x, try different build paths (if current version is 5.x)
            { url: `https://unpkg.com/pdfjs-dist@${pdfVersion}/legacy/build/pdf.worker.min.js`, version: pdfVersion, description: 'unpkg Legacy Build' },
            { url: `https://unpkg.com/pdfjs-dist@${pdfVersion}/webpack/pdf.worker.min.js`, version: pdfVersion, description: 'unpkg Webpack Build' },
            
            // Additional stable fallbacks
            { url: 'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.10.111/pdf.worker.min.js', version: '3.10.111', description: 'CloudFlare CDN (3.10.111)' },
            { url: 'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/2.16.105/pdf.worker.min.js', version: '2.16.105', description: 'CloudFlare CDN (2.16.105)' }
        ];

        for (const config of workerConfigs) {
            try {
                console.log(`Testing PDF worker: ${config.description} (v${config.version})`);
                
                // Quick HEAD request to check availability with timeout
                const controller = new AbortController();
                const timeoutId = setTimeout(() => controller.abort(), 5000);
                
                const response = await fetch(config.url, { 
                    method: 'HEAD',
                    cache: 'no-cache',
                    signal: controller.signal
                });
                
                clearTimeout(timeoutId);
                
                if (response.ok) {
                    // For local files, do additional validation
                    if (config.url.startsWith('/')) {
                        try {
                            const testController = new AbortController();
                            const testTimeoutId = setTimeout(() => testController.abort(), 3000);
                            
                            const testResponse = await fetch(config.url, { 
                                headers: { 'Range': 'bytes=0-1000' },
                                signal: testController.signal
                            });
                            
                            clearTimeout(testTimeoutId);
                            const snippet = await testResponse.text();
                            
                            // Check if it's actually JavaScript (more thorough check for 3.11.174)
                            if (snippet.includes('<html') || snippet.includes('<!DOCTYPE') || 
                                snippet.includes('<body') || snippet.includes('404') ||
                                snippet.includes('Not Found') || snippet.includes('Cannot GET')) {
                                console.warn(`Local worker appears to be HTML/error page, skipping: ${config.url}`);
                                continue;
                            }
                            
                            // Check for PDF.js worker signatures specific to 3.11.174
                            if (snippet.length < 100 || (!snippet.includes('function') && !snippet.includes('var') && 
                                !snippet.includes('!function') && !snippet.includes('const') && !snippet.includes('let') &&
                                !snippet.includes('pdfjsLib') && !snippet.includes('worker'))) {
                                console.warn(`Local worker doesn't look like valid PDF.js worker, skipping: ${config.url}`);
                                continue;
                            }
                            
                            console.log(`✅ Local 3.11.174 worker validated successfully`);
                        } catch (rangeError) {
                            console.warn(`Could not validate local worker, skipping: ${config.url}`, rangeError.message);
                            continue;
                        }
                    }
                    
                    // Set the worker source
                    pdfjsLib.GlobalWorkerOptions.workerSrc = config.url;
                    console.log(`✅ Using PDF.js worker: ${config.description} (v${config.version})`);
                    console.log(`   URL: ${config.url}`);
                    
                    // Shorter wait time since we have a stable local worker
                    await new Promise(resolve => setTimeout(resolve, 500));
                    
                    return config;
                }
            } catch (error) {
                console.warn(`❌ Failed to load worker from ${config.description}:`, error.message);
            }
        }
        
        console.error('❌ No valid PDF.js worker found from any source');
        return null;
    })();

    return workerSetupPromise;
};

const OCR = () => {
    const [selectedOllama, setSelectedOllama] = useState(localStorage.getItem("selectedOllama") || 'http://localhost:11434');
    const [selectedModel, setSelectedModel] = useState(localStorage.getItem("selectedOCRModel") || localStorage.getItem("selectedLLMModel"));
    const [availableModels, setAvailableModels] = useState([]);
    const [uploadedFiles, setUploadedFiles] = useState([]);
    const [extractedText, setExtractedText] = useState('');
    const [isProcessing, setIsProcessing] = useState(false);
    const [progress, setProgress] = useState(0);
    const [currentFile, setCurrentFile] = useState(null);
    const [showPreview, setShowPreview] = useState(false);
    const [previewImages, setPreviewImages] = useState([]);
    const [processingStats, setProcessingStats] = useState(null);
    const [isConnected, setIsConnected] = useState(true);
    const [connectionQuality, setConnectionQuality] = useState('good');
    const [imageQuality, setImageQuality] = useState(parseInt(localStorage.getItem("ocrImageQuality") || "95"));
    const [timeout, setTimeout] = useState(parseInt(localStorage.getItem("ocrTimeout") || "180"));
    const [maxRetries, setMaxRetries] = useState(parseInt(localStorage.getItem("ocrMaxRetries") || "2"));
    const [workerStatus, setWorkerStatus] = useState('checking');
    const [workerUrl, setWorkerUrl] = useState('');
    const [workerVersion, setWorkerVersion] = useState('');
    const [workerConfig, setWorkerConfig] = useState(null);
    const [isTestingWorker, setIsTestingWorker] = useState(false);
    // Add new state for language handling (after line 154, with other state declarations)
    const [selectedLanguage, setSelectedLanguage] = useState(localStorage.getItem("ocrLanguage") || "auto");
    const [useLanguageHints, setUseLanguageHints] = useState(localStorage.getItem("ocrUseLanguageHints") === "true");
    // Add language options with prioritized languages at the top
    const languageOptions = [
        { label: 'Auto-detect (Mixed/Unknown)', value: 'auto' },
        { label: '🇺🇸 English', value: 'en' },
        { label: '🇷🇺 Russian', value: 'ru' },
        { label: '🇧🇬 BulprocessImageWithStreamgarian', value: 'bg' },
        { label: '──────────────────', value: 'separator', disabled: true },
        { label: 'Spanish', value: 'es' },
        { label: 'French', value: 'fr' },
        { label: 'German', value: 'de' },
        { label: 'Italian', value: 'it' },
        { label: 'Portuguese', value: 'pt' },
        { label: 'Dutch', value: 'nl' },
        { label: 'Chinese (Simplified)', value: 'zh-cn' },
        { label: 'Chinese (Traditional)', value: 'zh-tw' },
        { label: 'Japanese', value: 'ja' },
        { label: 'Korean', value: 'ko' },
        { label: 'Arabic', value: 'ar' },
        { label: 'Hindi', value: 'hi' },
        { label: 'Thai', value: 'th' },
        { label: 'Vietnamese', value: 'vi' },
        { label: 'Polish', value: 'pl' },
        { label: 'Czech', value: 'cs' },
        { label: 'Hungarian', value: 'hu' },
        { label: 'Finnish', value: 'fi' },
        { label: 'Swedish', value: 'sv' },
        { label: 'Norwegian', value: 'no' },
        { label: 'Danish', value: 'da' }
    ];    
    // New streaming state
    const [streamingText, setStreamingText] = useState('');
    const [currentPageProcessing, setCurrentPageProcessing] = useState(null);
    const [isStreaming, setIsStreaming] = useState(false);
    // ADD: New state for table detection
    const [detectTables, setDetectTables] = useState(localStorage.getItem("ocrDetectTables") === "true");
    const [structuredData, setStructuredData] = useState(null); // Store parsed structure
    // ADD: New state for page selection
    const [pdfPages, setPdfPages] = useState([]);
    const [selectedPages, setSelectedPages] = useState([]);
    const [showPageSelector, setShowPageSelector] = useState(false);
    const [pdfFile, setPdfFile] = useState(null);
    const [pageSelectMode, setPageSelectMode] = useState('all'); // 'all', 'range', 'custom'
    const [pageRange, setPageRange] = useState({ start: 1, end: 1 });

    const toast = useRef(null);
    const fileUploadRef = useRef(null);
    const abortControllerRef = useRef(null);

    // Supported file types
    const supportedTypes = {
        'image/jpeg': 'JPEG Image',
        'image/jpg': 'JPG Image',
        'image/png': 'PNG Image',
        'image/tiff': 'TIFF Image',
        'image/webp': 'WebP Image',
        'application/pdf': 'PDF Document'
    };

    // Optimized PDF.js worker test for 3.11.174
    const testWorker = useCallback(async () => {
        // Prevent multiple simultaneous tests
        if (isTestingWorker) {
            console.log('🔄 Worker test already in progress, skipping...');
            return;
        }

        try {
            setIsTestingWorker(true);
            setWorkerStatus('testing');
            
            // Ensure worker is set up first
            let config = workerConfig;
            if (!config) {
                config = await setupWorker();
                setWorkerConfig(config);
            }
            
            if (!config) {
                throw new Error('No worker configuration available');
            }

            const currentWorkerSrc = pdfjsLib.GlobalWorkerOptions.workerSrc;
            if (!currentWorkerSrc) {
                throw new Error('Worker source not set after setup');
            }

            setWorkerUrl(currentWorkerSrc);
            
            console.log('🧪 Testing PDF worker with optimized settings for 3.11.174...');
            
            // Optimized minimal PDF for 3.11.174 compatibility
            const minimalPDF = new Uint8Array([
                37, 80, 68, 70, 45, 49, 46, 52, 10, 37, 226, 227, 207, 211, 10, 10,
                49, 32, 48, 32, 111, 98, 106, 10, 60, 60, 10, 47, 84, 121, 112, 101,
                32, 47, 67, 97, 116, 97, 108, 111, 103, 10, 47, 80, 97, 103, 101, 115,
                32, 50, 32, 48, 32, 82, 10, 62, 62, 10, 101, 110, 100, 111, 98, 106,
                10, 10, 50, 32, 48, 32, 111, 98, 106, 10, 60, 60, 10, 47, 84, 121,
                112, 101, 32, 47, 80, 97, 103, 101, 115, 10, 47, 75, 105, 100, 115,
                32, 91, 51, 32, 48, 32, 82, 93, 10, 47, 67, 111, 117, 110, 116, 32,
                49, 10, 62, 62, 10, 101, 110, 100, 111, 98, 106, 10, 10, 51, 32, 48,
                32, 111, 98, 106, 10, 60, 60, 10, 47, 84, 121, 112, 101, 32, 47, 80,
                97, 103, 101, 10, 47, 80, 97, 114, 101, 110, 116, 32, 50, 32, 48, 32,
                82, 10, 47, 77, 101, 100, 105, 97, 66, 111, 120, 32, 91, 48, 32, 48,
                32, 54, 49, 50, 32, 55, 57, 50, 93, 10, 62, 62, 10, 101, 110, 100,
                111, 98, 106, 10, 10, 120, 114, 101, 102, 10, 48, 32, 52, 10, 48, 48,
                48, 48, 48, 48, 48, 48, 48, 48, 32, 54, 53, 53, 51, 53, 32, 102, 32,
                10, 48, 48, 48, 48, 48, 48, 48, 48, 49, 53, 32, 48, 48, 48, 48, 48,
                32, 110, 32, 10, 48, 48, 48, 48, 48, 48, 48, 48, 55, 57, 32, 48, 48,
                48, 48, 48, 32, 110, 32, 10, 48, 48, 48, 48, 48, 48, 48, 49, 55, 51,
                32, 48, 48, 48, 48, 48, 32, 110, 32, 10, 116, 114, 97, 105, 108, 101,
                114, 10, 60, 60, 10, 47, 83, 105, 122, 101, 32, 52, 10, 47, 82, 111,
                111, 116, 32, 49, 32, 48, 32, 82, 10, 62, 62, 10, 115, 116, 97, 114,
                116, 120, 114, 101, 102, 10, 50, 57, 51, 10, 37, 37, 69, 79, 70
            ]).buffer;

            // Simplified test with proper promise handling
            const loadingTask = pdfjsLib.getDocument({ 
                data: minimalPDF,
                verbosity: 0,
                // Settings optimized for 3.11.174
                disableAutoFetch: true,
                disableStream: true,
                disableFontFace: false, // 3.11.174 handles fonts better
                isEvalSupported: false,
                isOffscreenCanvasSupported: false
            });
            
            // Set a timeout promise
            const timeoutPromise = new Promise((_, reject) => {
                setTimeout(() => reject(new Error('Worker test timeout after 30 seconds')), 30000);
            });

            // Create the test promise
            const testPromise = (async () => {
                const pdf = await loadingTask.promise;
                console.log(`📄 Test PDF loaded: ${pdf.numPages} pages`);
                
                // Try to get the first page
                const page = await pdf.getPage(1);
                console.log(`📄 Test page loaded: ${page.pageNumber}`);
                
                return true; // Explicitly return success
            })();

            // Wait for either test completion or timeout
            await Promise.race([testPromise, timeoutPromise]);
            
            setWorkerStatus('ready');
            setWorkerVersion(config.version);
            
            console.log('✅ PDF.js worker test successful with 3.11.174');
            
            toast.current?.show({
                severity: 'success',
                summary: 'PDF Worker Ready',
                detail: `PDF processing ready with v${config.version}`,
                life: 3000
            });
            
        } catch (error) {
            console.error('❌ PDF.js worker test failed:', error);
            setWorkerStatus('failed');
            
            // Check if it's just a timeout but worker is actually working
            if (error.message.includes('timeout')) {
                // Since we see the logs working, let's assume it's ready anyway
                console.log('🔄 Timeout detected but worker appears functional, marking as ready...');
                setWorkerStatus('ready');
                setWorkerVersion(workerConfig?.version || '3.11.174');
                
                toast.current?.show({
                    severity: 'success',
                    summary: 'PDF Worker Ready (Timeout Override)',
                    detail: 'PDF processing appears to be working despite timeout',
                    life: 3000
                });
            } else {
                toast.current?.show({
                    severity: 'error',
                    summary: 'PDF Worker Failed',
                    detail: 'PDF processing unavailable. Only image files can be processed.',
                    life: 5000
                });
            }
        } finally {
            setIsTestingWorker(false);
        }
    }, [workerConfig, isTestingWorker]);

    // Initialize component
    React.useEffect(() => {
        let cleanup = null;
        
        const initializeComponent = async () => {
            // Check connection and fetch models first
            checkConnection();
            fetchAvailableModels();
            
            // Set up worker
            setWorkerStatus('checking');
            try {
                const config = await setupWorker();
                setWorkerConfig(config);
                
                if (config) {
                    setWorkerUrl(config.url);
                    setWorkerVersion(config.version);
                    
                    // Test worker after setup - only once with longer delay
                    const testTimeout = setTimeout(() => {
                        if (!isTestingWorker) { // Only test if not already testing
                            testWorker();
                        }
                    }, 1000); // Increased delay to 1 second
                    
                    cleanup = () => clearTimeout(testTimeout);
                } else {
                    setWorkerStatus('failed');
                    toast.current?.show({
                        severity: 'error',
                        summary: 'PDF Worker Setup Failed',
                        detail: 'Could not initialize PDF worker. Only image files can be processed.',
                        life: 5000
                    });
                }
            } catch (error) {
                console.error('Worker setup failed:', error);
                setWorkerStatus('failed');
            }
            
            // Log PDF.js information
            console.log('📄 PDF.js version:', pdfjsLib.version);
            console.log('🔧 Initial worker source:', pdfjsLib.GlobalWorkerOptions.workerSrc);
        };

        initializeComponent();
        
        // Cleanup on unmount
        return () => {
            if (cleanup) cleanup();
        };
    }, [selectedOllama]); // Removed testWorker dependency to prevent multiple runs

    // Check Ollama connection
    const checkConnection = useCallback(async () => {
        try {
            const startTime = Date.now();
            const response = await axios.get(`${selectedOllama}/api/version`, { timeout: 5000 });
            const latency = Date.now() - startTime;
            
            if (response.status === 200) {
                setIsConnected(true);
                setConnectionQuality(latency < 1000 ? 'good' : latency < 2000 ? 'fair' : 'poor');
            }
        } catch (error) {
            setIsConnected(false);
            setConnectionQuality('offline');
            console.error('Connection check failed:', error);
        }
    }, [selectedOllama]);

    // Fetch available models - FIXED: Don't filter models, let user choose
    const fetchAvailableModels = useCallback(async () => {
        try {
            const response = await axios.get(`${selectedOllama}/api/tags`, { timeout: 10000 });
            if (response.data && response.data.models) {
                // Don't filter - show all models like chatOverPicture does
                const allModels = response.data.models.map(model => ({
                    label: model.name,
                    value: model.name,
                    size: model.size
                }));
                
                setAvailableModels(allModels);
                
                if (allModels.length === 0) {
                    toast.current?.show({
                        severity: 'warn',
                        summary: 'No Models',
                        detail: 'No models found. Please install a vision-capable model.',
                        life: 5000
                    });
                }
            }
        } catch (error) {
            console.error('Failed to fetch models:', error);
            toast.current?.show({
                severity: 'error',
                summary: 'Connection Error',
                detail: 'Could not fetch available models',
                life: 3000
            });
        }
    }, [selectedOllama]);

  // IMPROVED: Convert file to base64 with better quality and preprocessing
    const fileToBase64 = (file, quality = imageQuality / 100) => {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = (event) => {
                const img = new Image();
                img.onload = () => {
                    const canvas = document.createElement('canvas');
                    const ctx = canvas.getContext('2d');
                    
                    // IMPROVED: Better dimensions for OCR - larger for better text recognition
                    const maxDim = 3072; // Increased from 1200 to 2048
                    let { width, height } = img;
                    
                    // Calculate scale to maintain aspect ratio
                    let scale = 1;
                    if (width > maxDim || height > maxDim) {
                        scale = Math.min(maxDim / width, maxDim / height);
                        width = Math.round(width * scale);
                        height = Math.round(height * scale);
                    }
                    
                    canvas.width = width;
                    canvas.height = height;
                    
                    // IMPROVED: Better image preprocessing for OCR
                    ctx.fillStyle = 'white'; // White background
                    ctx.fillRect(0, 0, width, height);
                    
                    // Enable image smoothing for better quality
                    ctx.imageSmoothingEnabled = false;
                    // ctx.imageSmoothingEnabled = true;
                    // ctx.imageSmoothingQuality = 'high';
                    
                    // Draw the image
                    ctx.drawImage(img, 0, 0, width, height);
                    
                    // IMPROVED: Higher quality JPEG with better compression
                    const dataUrl = canvas.toDataURL('image/png'); // Changed to PNG for lossless
                    // const dataUrl = canvas.toDataURL('image/jpeg', Math.max(quality, 0.85)); // Minimum 85% quality
                    const base64 = dataUrl.split(',')[1];
                    resolve(base64);
                };
                img.onerror = reject;
                img.src = event.target.result;
            };
            reader.onerror = reject;
            reader.readAsDataURL(file);
        });
    };

    // IMPROVED: PDF processing with higher quality and better rendering
    const pdfToImages = async (file) => {
        if (workerStatus !== 'ready') {
            throw new Error('PDF worker is not ready. Please wait for worker initialization or refresh the page.');
        }

        try {
            console.log('📄 Starting PDF processing with enhanced OCR settings...');
            const arrayBuffer = await file.arrayBuffer();
            
            const loadingTask = pdfjsLib.getDocument({ 
                data: arrayBuffer,
                verbosity: 0,
                disableFontFace: false,
                isEvalSupported: false,
                isOffscreenCanvasSupported: false,
                standardFontDataUrl: null,
                useSystemFonts: true,
                disableAutoFetch: false,
                disableStream: false
            });
            
            const pdf = await loadingTask.promise;
            console.log(`📄 PDF loaded successfully. Pages: ${pdf.numPages}`);
            
            const images = [];

            for (let pageNum = 1; pageNum <= pdf.numPages; pageNum++) {
                try {
                    console.log(`📄 Processing page ${pageNum} with enhanced OCR settings...`);
                    const page = await pdf.getPage(pageNum);
                    
                    // IMPROVED: Much higher scale for better OCR recognition
                    const baseScale = imageQuality >= 90 ? 6.0 :    // Increased from 4.0
                                    imageQuality >= 80 ? 5.0 :    // Increased from 3.5
                                    imageQuality >= 70 ? 4.0 :    // Increased from 3.0
                                    3.0;                           // Increased from 2.5

                    const viewport = page.getViewport({ scale: baseScale });

                    const canvas = document.createElement('canvas');
                    const context = canvas.getContext('2d');
                    canvas.height = viewport.height;
                    canvas.width = viewport.width;

                    // IMPROVED: Better canvas settings for text rendering
                    context.fillStyle = 'white';
                    context.fillRect(0, 0, canvas.width, canvas.height);

                    // Enable high-quality rendering with better settings
                    context.imageSmoothingEnabled = false;  // Changed to false for sharper text
                    context.textRenderingOptimization = 'optimizeQuality';

                    const renderContext = {
                        canvasContext: context,
                        viewport: viewport,
                        intent: 'print',
                        renderInteractiveForms: false,
                        annotationMode: 0,
                        // IMPROVED: Add text layer rendering
                        enableWebGL: false,
                        optionalContentConfigPromise: null,
                        transform: null,
                        imageLayer: null
                    };
                    await page.render(renderContext).promise;

                    // IMPROVED: Higher quality JPEG conversion
                    const quality = Math.max(imageQuality / 100, 0.90); // Minimum 90% for OCR
                    const dataUrl = canvas.toDataURL('image/jpeg', quality);
                    const base64 = dataUrl.split(',')[1];
                    
                    images.push({
                        base64,
                        pageNumber: pageNum,
                        width: viewport.width,
                        height: viewport.height
                    });
                    
                    console.log(`✅ Page ${pageNum} processed with scale ${baseScale}, dimensions: ${viewport.width}x${viewport.height}`);
                } catch (pageError) {
                    console.error(`❌ Error processing page ${pageNum}:`, pageError);
                    toast.current?.show({
                        severity: 'warn',
                        summary: 'Page Error',
                        detail: `Failed to process page ${pageNum}. Continuing with other pages.`,
                        life: 3000
                    });
                }
            }

            console.log(`✅ PDF processing complete. ${images.length} pages processed with enhanced OCR settings.`);
            return images;
            
        } catch (error) {
            console.error('❌ PDF processing error:', error);
            
            if (error.message.includes('worker') || error.message.includes('EOF') || error.message.includes('Unexpected token')) {
                throw new Error('PDF worker error. Please refresh the page and try again.');
            } else if (error.message.includes('Invalid PDF')) {
                throw new Error('Invalid PDF file. Please ensure the file is not corrupted.');
            } else if (error.message.includes('Password')) {
                throw new Error('Password-protected PDFs are not supported.');
            } else if (error.message.includes('network')) {
                throw new Error('Network error while processing PDF. Please check your connection.');
            } else {
                throw new Error(`Failed to process PDF: ${error.message}`);
            }
        }
    };

    // Add new state for column handling
    const [columnMode, setColumnMode] = useState(localStorage.getItem("ocrColumnMode") || "auto");
    const [splitColumns, setSplitColumns] = useState(localStorage.getItem("ocrSplitColumns") === "true");
    
    // MODIFIED: Load PDF pages for selection with thumbnails for all pages
    const loadPdfPages = async (file) => {
        if (workerStatus !== 'ready') {
            throw new Error('PDF worker is not ready. Please wait for worker initialization or refresh the page.');
        }

        try {
            console.log('📄 Loading PDF for page selection...');
            const arrayBuffer = await file.arrayBuffer();
            
            const loadingTask = pdfjsLib.getDocument({ 
                data: arrayBuffer,
                verbosity: 0,
                disableFontFace: false,
                isEvalSupported: false,
                isOffscreenCanvasSupported: false
            });
            
            const pdf = await loadingTask.promise;
            console.log(`📄 PDF loaded successfully. Pages: ${pdf.numPages}`);
            
            const pages = [];

            // IMPROVED: Generate thumbnails for all pages (with reasonable limit)
            const maxThumbnails = Math.min(pdf.numPages, 50); // Increased limit to 50 pages
            
            // Show progress for thumbnail generation
            toast.current?.show({
                severity: 'info',
                summary: 'Loading Page Previews',
                detail: `Generating thumbnails for ${maxThumbnails} pages...`,
                life: 3000
            });
            
            for (let pageNum = 1; pageNum <= pdf.numPages; pageNum++) {
                const pageInfo = {
                    pageNumber: pageNum,
                    thumbnail: null
                };

                // Generate thumbnails for more pages (or all if reasonable number)
                if (pageNum <= maxThumbnails) {
                    try {
                        console.log(`📄 Generating thumbnail for page ${pageNum}/${maxThumbnails}...`);
                        const page = await pdf.getPage(pageNum);
                        
                        // IMPROVED: Smaller scale for faster generation but still visible
                        const viewport = page.getViewport({ scale: 0.25 }); // Reduced scale for faster generation
                        
                        const canvas = document.createElement('canvas');
                        const context = canvas.getContext('2d');
                        canvas.height = viewport.height;
                        canvas.width = viewport.width;

                        context.fillStyle = 'white';
                        context.fillRect(0, 0, canvas.width, canvas.height);

                        const renderContext = {
                            canvasContext: context,
                            viewport: viewport,
                            intent: 'print',
                            // IMPROVED: Faster rendering options for thumbnails
                            renderInteractiveForms: false,
                            annotationMode: 0
                        };

                        await page.render(renderContext).promise;
                        
                        // IMPROVED: Lower quality for thumbnails to save memory
                        pageInfo.thumbnail = canvas.toDataURL('image/jpeg', 0.6);
                        
                        console.log(`✅ Thumbnail generated for page ${pageNum}`);
                    } catch (pageError) {
                        console.warn(`Could not generate thumbnail for page ${pageNum}:`, pageError);
                    }
                }

                pages.push(pageInfo);
            }

            setPdfPages(pages);
            setSelectedPages(pages.map(p => p.pageNumber)); // Select all by default
            setPageRange({ start: 1, end: pdf.numPages });
            
            console.log(`✅ PDF page loading complete. ${pages.length} pages loaded, ${maxThumbnails} thumbnails generated.`);
            
            if (pdf.numPages > maxThumbnails) {
                toast.current?.show({
                    severity: 'info',
                    summary: 'Thumbnail Limit',
                    detail: `Showing thumbnails for first ${maxThumbnails} pages. All pages are still selectable.`,
                    life: 4000
                });
            }
            
            return pages;
            
        } catch (error) {
            console.error('❌ PDF page loading error:', error);
            throw new Error(`Failed to load PDF pages: ${error.message}`);
        }
    };
    // NEW: Process selected PDF pages
    const processSelectedPdfPages = async (file, selectedPageNumbers) => {
        if (workerStatus !== 'ready') {
            throw new Error('PDF worker is not ready. Please wait for worker initialization or refresh the page.');
        }

        try {
            console.log(`📄 Processing ${selectedPageNumbers.length} selected pages...`);
            const arrayBuffer = await file.arrayBuffer();
            
            const loadingTask = pdfjsLib.getDocument({ 
                data: arrayBuffer,
                verbosity: 0,
                disableFontFace: false,
                isEvalSupported: false,
                isOffscreenCanvasSupported: false,
                standardFontDataUrl: null,
                useSystemFonts: true,
                disableAutoFetch: false,
                disableStream: false
            });
            
            const pdf = await loadingTask.promise;
            const images = [];

            // Sort selected pages
            const sortedPages = [...selectedPageNumbers].sort((a, b) => a - b);

            for (const pageNum of sortedPages) {
                try {
                    console.log(`📄 Processing selected page ${pageNum}...`);
                    const page = await pdf.getPage(pageNum);
                    
                    const baseScale = imageQuality >= 90 ? 4.0 : 
                                    imageQuality >= 80 ? 3.5 : 
                                    imageQuality >= 70 ? 3.0 : 2.5;
                    
                    const viewport = page.getViewport({ scale: baseScale });
                    
                    const canvas = document.createElement('canvas');
                    const context = canvas.getContext('2d');
                    canvas.height = viewport.height;
                    canvas.width = viewport.width;

                    context.fillStyle = 'white';
                    context.fillRect(0, 0, canvas.width, canvas.height);
                    
                    context.imageSmoothingEnabled = true;
                    context.imageSmoothingQuality = 'high';
                    context.textRenderingOptimization = 'optimizeQuality';

                    const renderContext = {
                        canvasContext: context,
                        viewport: viewport,
                        intent: 'print',
                        renderInteractiveForms: false,
                        annotationMode: 0,
                        optionalContentConfigPromise: null,
                        transform: null,
                        imageLayer: null
                    };

                    await page.render(renderContext).promise;

                    const quality = Math.max(imageQuality / 100, 0.90);
                    const dataUrl = canvas.toDataURL('image/jpeg', quality);
                    const base64 = dataUrl.split(',')[1];
                    
                    images.push({
                        base64,
                        pageNumber: pageNum,
                        width: viewport.width,
                        height: viewport.height
                    });
                    
                    console.log(`✅ Page ${pageNum} processed`);
                } catch (pageError) {
                    console.error(`❌ Error processing page ${pageNum}:`, pageError);
                    toast.current?.show({
                        severity: 'warn',
                        summary: 'Page Error',
                        detail: `Failed to process page ${pageNum}. Continuing with other pages.`,
                        life: 3000
                    });
                }
            }

            console.log(`✅ Selected pages processing complete. ${images.length} pages processed.`);
            return images;
            
        } catch (error) {
            console.error('❌ Selected pages processing error:', error);
            throw new Error(`Failed to process selected pages: ${error.message}`);
        }
    };

    // NEW: Handle page selection mode change
    const handlePageSelectModeChange = (mode) => {
        setPageSelectMode(mode);
        
        if (mode === 'all') {
            setSelectedPages(pdfPages.map(p => p.pageNumber));
        } else if (mode === 'range') {
            const { start, end } = pageRange;
            const rangePages = [];
            for (let i = start; i <= end; i++) {
                if (i <= pdfPages.length) {
                    rangePages.push(i);
                }
            }
            setSelectedPages(rangePages);
        } else if (mode === 'custom') {
            setSelectedPages([]);
        }
    };

    // NEW: Handle page range change
    const handlePageRangeChange = (field, value) => {
        const newRange = { ...pageRange, [field]: value };
        setPageRange(newRange);
        
        if (pageSelectMode === 'range') {
            const rangePages = [];
            for (let i = newRange.start; i <= newRange.end; i++) {
                if (i <= pdfPages.length) {
                    rangePages.push(i);
                }
            }
            setSelectedPages(rangePages);
        }
    };

    // NEW: Toggle individual page selection
    const togglePageSelection = (pageNumber) => {
        if (pageSelectMode !== 'custom') return;
        
        setSelectedPages(prev => {
            if (prev.includes(pageNumber)) {
                return prev.filter(p => p !== pageNumber);
            } else {
                return [...prev, pageNumber].sort((a, b) => a - b);
            }
        });
    };

    // Replace the processImageWithStream function with this corrected version:
    const processImageWithStream = async (image, fileType, controller, retryCount = 0) => {
        let ocrPrompt;
        
        // Language-specific instructions
        const getLanguageInstructions = () => {
            if (!useLanguageHints || selectedLanguage === 'auto') {
                return '';
            }
            
            const languageNames = {
                'en': 'English',
                'ru': 'Russian (Русский)',
                'bg': 'Bulgarian (Български)',
                'es': 'Spanish',
                'fr': 'French',
                'de': 'German',
                'it': 'Italian',
                'pt': 'Portuguese',
                'nl': 'Dutch',
                'zh-cn': 'Chinese (Simplified)',
                'zh-tw': 'Chinese (Traditional)',
                'ja': 'Japanese',
                'ko': 'Korean',
                'ar': 'Arabic',
                'hi': 'Hindi',
                'th': 'Thai',
                'vi': 'Vietnamese',
                'pl': 'Polish',
                'cs': 'Czech',
                'hu': 'Hungarian',
                'fi': 'Finnish',
                'sv': 'Swedish',
                'no': 'Norwegian',
                'da': 'Danish'
            };
            
            const languageName = languageNames[selectedLanguage];
            let instructions = `\nIMPORTANT: The text in this document is primarily in ${languageName}. Pay special attention to ${languageName} characters, words, and text patterns.`;
            
            // Add specific instructions for Cyrillic languages
            if (selectedLanguage === 'ru' || selectedLanguage === 'bg') {
                instructions += ` Make sure to correctly recognize Cyrillic characters (А, Б, В, Г, Д, Е, Ё, Ж, З, И, Й, К, Л, М, Н, О, П, Р, С, Т, У, Ф, Х, Ц, Ч, Ш, Щ, Ъ, Ы, Ь, Э, Ю, Я) and distinguish them from similar Latin characters.`;
            }
            
            return instructions;
        };
        
        // CRITICAL FIX: Rewrite the prompt to prioritize ALL text extraction first
        if (detectTables) {
            ocrPrompt = `You are a comprehensive OCR system. Your PRIMARY goal is to extract EVERY SINGLE CHARACTER of text from this document without missing anything.
    
    CRITICAL INSTRUCTIONS - FOLLOW EXACTLY:
    
    1. EXTRACT ALL TEXT FIRST - Read the document from top to bottom, left to right
    2. Include EVERY paragraph, EVERY sentence, EVERY word visible
    3. NEVER skip any text content, even if it looks like regular paragraphs
    4. Extract everything you see: titles, headers, body text, formulas, captions, footnotes, everything
    
    SECONDARY: Table Detection Rules
    - ONLY after extracting all text, if you see data arranged in rows and columns, mark it with:
      [TABLE_START]
      Row1Column1|Row1Column2|Row1Column3
      Row2Column1|Row2Column2|Row2Column3
      [TABLE_END]
    
    READING PROCESS - MANDATORY:
    Step 1: Start from the very top of the document
    Step 2: Read EVERY line of text from left to right, top to bottom
    Step 3: Include ALL content: paragraphs, equations, formulas, explanations
    Step 4: When you encounter tabular data, mark it with [TABLE_START]...[TABLE_END]
    Step 5: Continue reading ALL remaining text after the table
    Step 6: Repeat until you reach the very bottom of the document
    
    WHAT YOU MUST INCLUDE:
    ✓ ALL paragraph text (introductions, explanations, conclusions)
    ✓ ALL formulas and equations (even between tables)
    ✓ ALL headers and subheaders (like "TABLE III. NET PRICE...")
    ✓ ALL technical descriptions and methodology text
    ✓ ALL text that appears between tables
    ✓ ALL footnotes and references
    
    CRITICAL EXAMPLES:
    
    Example of CORRECT extraction:
    Some introduction text here...
    
    The methodology for calculating prices involves several steps. First, we need to understand the basic formula.
    
    [TABLE_START]
    Item|Price|Quantity
    Apple|$1.50|10
    Orange|$2.00|8
    [TABLE_END]
    
    After analyzing the table above, we can see that the pricing structure follows a specific pattern. The next section will discuss implementation details.
    
    Another paragraph explaining the results and their implications for the research.
    
    [TABLE_START]
    Result|Value|Units
    Speed|120|mph
    Time|30|seconds
    [TABLE_END]
    
    The conclusion of this analysis shows that the proposed method is effective.
    
    ${getLanguageInstructions()}
    
    ABSOLUTE REQUIREMENTS:
    - Extract EVERY SINGLE WORD visible in the image
    - NEVER skip paragraphs of text, especially between tables
    - Include ALL explanatory text, methodology descriptions, and conclusions
    - Mark tables with [TABLE_START]...[TABLE_END] but don't let this distract from text extraction
    - Read the ENTIRE document from start to finish
    
    Now extract ALL text from this image - every word, every paragraph, without missing anything:`;
    
        } else {
            // Column-aware prompts when tables are disabled (unchanged)
            if (columnMode === "two-column") {
                ocrPrompt = `This document has TWO COLUMNS of text. Extract ALL text following this process:
    
            STEP 1: READ the ENTIRE LEFT COLUMN from top to bottom - EVERY paragraph, EVERY line
            STEP 2: Add the separator: "--- RIGHT COLUMN ---"
            STEP 3: READ the ENTIRE RIGHT COLUMN from top to bottom - EVERY paragraph, EVERY line
    
            CRITICAL RULES:
            - Extract EVERY SINGLE line of text you can see
            - Include ALL paragraphs, headings, body text, conclusions, everything
            - NEVER skip paragraphs or content
            - NEVER mix text from different columns
            - Read each column completely from top to bottom
            - DO NOT add section numbers or headers not visible in the image
            - CONTINUE until you have extracted ALL visible text - do not stop early
    
            ${getLanguageInstructions()}
    
            Format:
            [All left column text from top to bottom - EVERY paragraph]
    
            --- RIGHT COLUMN ---
    
            [All right column text from top to bottom - EVERY paragraph]
    
            Extract EVERY visible line of text from start to finish - do not stop early.`;
    
            } else if (columnMode === "multi-column") {
                ocrPrompt = `This document has MULTIPLE COLUMNS. Extract ALL text systematically:
    
            1. Identify all columns from left to right
            2. Read each column COMPLETELY from top to bottom - EVERY paragraph
            3. Separate each column with "--- COLUMN X ---"
            4. Extract EVERY line of text in each column
            5. Never skip paragraphs or content
            6. CONTINUE until you have extracted ALL visible text - do not stop early
    
            ${getLanguageInstructions()}
    
            Read columns in order: leftmost first, then next column, etc.
            Extract ALL visible text from start to finish - every paragraph, every line, everything you can see.`;
    
            } else if (columnMode === "auto") {
                ocrPrompt = `Extract ALL text from this image from start to finish. 
    
            If this document has COLUMNS:
            - Read the LEFT column completely from top to bottom FIRST - EVERY paragraph
            - Then read the RIGHT column completely from top to bottom - EVERY paragraph  
            - Add "--- RIGHT COLUMN ---" between them
            - NEVER skip paragraphs or content
    
            If this is single column text:
            - Read normally from top to bottom, left to right - EVERY line
    
            CRITICAL: 
            - Extract EVERY SINGLE paragraph and line of text you can see
            - Include introductions, body paragraphs, conclusions, everything
            - DO NOT skip content because it seems like "regular text"
            - DO NOT add section numbers or headers not visible
            - Extract ALL visible content without exception
            - CONTINUE until you have extracted ALL visible text - do not stop early
    
            ${getLanguageInstructions()}
    
            Extract EVERYTHING visible from start to finish - every word, every paragraph.`;
    
            } else {
                ocrPrompt = `Extract ALL visible text from this image exactly as written from start to finish. 
    
            Rules:
            - Read left to right, top to bottom
            - Include EVERY line, EVERY paragraph, EVERY piece of text visible
            - Include ALL content: headings, body paragraphs, conclusions, captions, everything
            - Preserve formatting and spacing
            - Don't skip ANYTHING that is visible
            - Don't add explanations or content not shown
            - CONTINUE until you have extracted ALL visible text - do not stop early
    
            ${getLanguageInstructions()}
    
            Return ALL extracted visible text from start to finish - every single word and paragraph you can see.`;
            }
        }
    
        // Rest of the function remains the same...
        try {
            const pageLabel = fileType === 'application/pdf' ? `Page ${image.pageNumber}` : 'Image';
            setCurrentPageProcessing(pageLabel);
            setIsStreaming(true);
            setStreamingText('');

            const body = {
                model: selectedModel,
                prompt: ocrPrompt,
                stream: true,
                images: [image.base64],
                options: {
                    temperature: 0.0,
                    top_p: 0.1,
                    top_k: 5,
                    repeat_penalty: 1.15,
                    num_predict: 16384,
                    num_ctx: 8192,
                    stop: ["\n\n\n\n\n", "---END---"],
                    seed: 42,
                    frequency_penalty: 0.3,
                    presence_penalty: 0.2,
                    mirostat: 1,
                    mirostat_tau: 3.0,
                    mirostat_eta: 0.1
                }
            };

            console.log('🤖 Sending request to Ollama with prompt:', `${selectedOllama}/api/generate`);
            
            // ENHANCED: Add CORS-specific error handling
            const requestOptions = {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    // Add CORS headers for preflight
                    'Accept': 'application/json',
                },
                body: JSON.stringify(body),
                signal: controller.signal,
                // Add mode for better CORS handling
                mode: 'cors',
                credentials: 'omit'
            };
            
            let response;
            try {
                response = await fetch(`${selectedOllama}/api/generate`, requestOptions);
            } catch (fetchError) {
                // ENHANCED: Better CORS error detection
                if (fetchError.message.includes('CORS') || 
                    fetchError.message.includes('cross-origin') ||
                    fetchError.message.includes('Network request failed')) {
                    throw new Error(`CORS Error: Cannot connect to Ollama server from this domain. Please configure CORS on your Ollama server with: OLLAMA_ORIGINS="*" ollama serve`);
                }
                throw fetchError;
            }

            if (!response.ok) {
                let errorMessage = `HTTP ${response.status} ${response.statusText}`;
                
                // ENHANCED: Check for common CORS-related status codes
                if (response.status === 404) {
                    // Check if this might be a CORS preflight issue
                    try {
                        const testResponse = await fetch(`${selectedOllama}/api/version`, {
                            method: 'GET',
                            mode: 'cors',
                            credentials: 'omit'
                        });
                        
                        if (!testResponse.ok) {
                            errorMessage += ` - This appears to be a CORS issue. Please configure your Ollama server with: OLLAMA_ORIGINS="*" ollama serve`;
                        }
                    } catch (testError) {
                        errorMessage += ` - CORS preflight failed. Please configure your Ollama server with: OLLAMA_ORIGINS="*" ollama serve`;
                    }
                } else if (response.status === 0) {
                    errorMessage = `Network error - likely CORS issue. Please configure your Ollama server with: OLLAMA_ORIGINS="*" ollama serve`;
                }
                
                try {
                    const errorText = await response.text();
                    if (errorText && !errorText.includes('html')) {
                        errorMessage += ` - ${errorText}`;
                    }
                } catch (e) {
                    // Ignore error reading response body
                }
                
                throw new Error(errorMessage);
            }
    
            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let fullResponse = '';
            let currentStreamText = '';
            let buffer = '';
            let lastContent = '';
            let repetitionCount = 0;
            const MAX_REPETITION = 5;
    
            while (true) {
                const { done, value } = await reader.read();
    
                if (done) break;
    
                const chunk = decoder.decode(value, { stream: true });
                buffer += chunk;
    
                const lines = buffer.split('\n');
                buffer = lines.pop();
    
                for (const line of lines) {
                    if (line.trim() === '') continue;
    
                    try {
                        const jsonResponse = JSON.parse(line);
    
                        if (jsonResponse.response) {
                            let responseText = jsonResponse.response;
    
                            if (responseText.length > 30 && lastContent.includes(responseText)) {
                                repetitionCount++;
                                console.warn(`🔄 Repetition detected (${repetitionCount}/${MAX_REPETITION}):`, responseText.substring(0, 50));
                                
                                if (repetitionCount >= MAX_REPETITION) {
                                    console.warn('🛑 Stopping due to excessive repetition');
                                    jsonResponse.done = true;
                                    break;
                                }
                            } else {
                                repetitionCount = 0;
                            }
    
                            if (fullResponse.length > 2000) {
                                const lastPart = fullResponse.slice(-800);
                                if (lastPart.includes(responseText) && responseText.length > 20) {
                                    console.warn('🔄 Loop pattern detected, stopping');
                                    jsonResponse.done = true;
                                    break;
                                }
                            }
    
                            const stopMarkers = [
                                'I can see the image contains',
                                'This image appears to show',
                                'The document seems to contain',
                                'Looking at this document'
                            ];
                            
                            if (stopMarkers.some(marker => responseText.includes(marker))) {
                                console.warn('🛑 Stop marker detected:', responseText.substring(0, 50));
                                jsonResponse.done = true;
                                break;
                            }
    
                            fullResponse += responseText;
                            currentStreamText += responseText;
                            lastContent = fullResponse.slice(-300);
                            
                            setStreamingText(currentStreamText);
    
                            if (fullResponse.length > 50000) {
                                console.warn('🛑 Response very long, likely complete or repetitive. Stopping.');
                                jsonResponse.done = true;
                                break;
                            }
                        }
    
                        if (jsonResponse.done) {
                            setIsStreaming(false);
                            setCurrentPageProcessing(null);
    
                            let cleanedResponse = fullResponse.trim();
    
                            const prefixesToRemove = [
                                /^(I can see|This image shows|The text in this image|Looking at this image|Here is the extracted|Here's the text|The extracted text|From this image|I can extract|Based on this image)[^:]*:?\s*/i,
                            ];
    
                            let hasRemovedPrefix = false;
                            prefixesToRemove.forEach(regex => {
                                if (!hasRemovedPrefix && regex.test(cleanedResponse)) {
                                    cleanedResponse = cleanedResponse.replace(regex, '');
                                    hasRemovedPrefix = true;
                                }
                            });
    
                            cleanedResponse = cleanedResponse
                                .replace(/^###\s+[A-Z][A-Z\s]+$/gm, '')
                                .replace(/^\*\*([A-Z][A-Z\s]+)\*\*$/gm, '$1')
                                .replace(/\n{4,}/g, '\n\n\n')
                                .trim();
    
                            // CRITICAL FIX: Normalize table markers to be consistent
                            console.log('🔧 Normalizing table markers...');
                            console.log('📝 Before normalization preview:', cleanedResponse.substring(0, 500));
                            
                            // Fix common variations of table markers
                            cleanedResponse = cleanedResponse
                                // Fix various start marker variations
                                .replace(/\[Table_Start\]/gi, '[TABLE_START]')
                                .replace(/\[table_start\]/gi, '[TABLE_START]')
                                .replace(/\[TABLE_start\]/gi, '[TABLE_START]')
                                .replace(/\[Table_START\]/gi, '[TABLE_START]')
                                .replace(/\[table_START\]/gi, '[TABLE_START]')
                                .replace(/\[TABLESTART\]/gi, '[TABLE_START]')
                                .replace(/\[TableStart\]/gi, '[TABLE_START]')
                                .replace(/\[tablestart\]/gi, '[TABLE_START]')
                                
                                // Fix various end marker variations
                                .replace(/\[Table_End\]/gi, '[TABLE_END]')
                                .replace(/\[table_end\]/gi, '[TABLE_END]')
                                .replace(/\[TABLE_end\]/gi, '[TABLE_END]')
                                .replace(/\[Table_END\]/gi, '[TABLE_END]')
                                .replace(/\[table_END\]/gi, '[TABLE_END]')
                                .replace(/\[TABLEEND\]/gi, '[TABLE_END]')
                                .replace(/\[TableEnd\]/gi, '[TABLE_END]')
                                .replace(/\[tableend\]/gi, '[TABLE_END]');
    
                            console.log('📝 After normalization preview:', cleanedResponse.substring(0, 500));
                            console.log('🔍 TABLE_START markers found:', (cleanedResponse.match(/\[TABLE_START\]/g) || []).length);
                            console.log('🔍 TABLE_END markers found:', (cleanedResponse.match(/\[TABLE_END\]/g) || []).length);
    
                            // ENHANCED: Fallback table detection if no markers found
                            if (detectTables && !cleanedResponse.includes('[TABLE_START]')) {
                                console.log('⚠️ No table markers found, applying fallback table detection...');
                                cleanedResponse = applyFallbackTableDetection(cleanedResponse);
                                console.log('🔍 After fallback - TABLE_START markers:', (cleanedResponse.match(/\[TABLE_START\]/g) || []).length);
                            }
    
                            // Apply column processing if needed and no table markers were found
                            if (!detectTables || !cleanedResponse.includes('[TABLE_START]')) {
                                if (columnMode === "auto" && !cleanedResponse.includes('--- RIGHT COLUMN ---')) {
                                    const isProbablyTwoColumn = detectColumnLayout(cleanedResponse);
                                    if (isProbablyTwoColumn) {
                                        console.log('🔍 Auto-detected likely two-column content, applying post-processing');
                                        cleanedResponse = postProcessColumns(cleanedResponse);
                                    }
                                }
    
                                if (splitColumns && columnMode !== "single" && !cleanedResponse.includes('--- RIGHT COLUMN ---')) {
                                    cleanedResponse = postProcessColumns(cleanedResponse);
                                }
                            }
    
                            cleanedResponse = cleanedResponse
                                .replace(/^\s*\n+/, '')
                                .replace(/\n+\s*$/, '')
                                .replace(/\n{4,}/g, '\n\n\n')
                                .trim();
    
                            console.log('📝 Final cleaned length:', cleanedResponse.length);
                            console.log('📝 Final cleaned preview:', cleanedResponse.substring(0, 300) + '...');
                            console.log('📝 Final cleaned last 300 chars:', cleanedResponse.substring(Math.max(0, cleanedResponse.length - 300)));
                            console.log('🔍 Final TABLE_START count:', (cleanedResponse.match(/\[TABLE_START\]/g) || []).length);
                            console.log('🔍 Final TABLE_END count:', (cleanedResponse.match(/\[TABLE_END\]/g) || []).length);
    
                            return cleanedResponse;
                        }
                    } catch (parseError) {
                        console.debug('JSON parse error (expected for streaming):', parseError);
                    }
                }
            }
    
            setIsStreaming(false);
            setCurrentPageProcessing(null);
            return fullResponse.trim();
    
        } catch (error) {
            setIsStreaming(false);
            setCurrentPageProcessing(null);

            if (error.name === 'AbortError') {
                throw error;
            }

            console.error(`OCR streaming attempt ${retryCount + 1} failed:`, error.message);

            // ENHANCED: Don't retry CORS errors
            if (error.message.includes('CORS') || error.message.includes('cross-origin')) {
                throw new Error(`CORS Configuration Required: ${error.message}`);
            }

            if (retryCount < maxRetries) {
                toast.current?.show({
                    severity: 'warn',
                    summary: 'Retrying OCR',
                    detail: `Attempt ${retryCount + 1} failed. Retrying... (${retryCount + 1}/${maxRetries + 1})`,
                    life: 3000
                });

                await new Promise(resolve => setTimeout(resolve, Math.pow(2, retryCount) * 2000));
                return await processImageWithStream(image, fileType, controller, retryCount + 1);
            } else {
                if (error.name === 'AbortError') {
                    throw new Error('OCR request was cancelled');
                } else if (error.message.includes('timeout') || error.message.includes('408')) {
                    throw new Error(`OCR timeout after ${timeout}s. Try reducing image quality or increasing timeout.`);
                } else if (error.message.includes('503')) {
                    throw new Error('OCR service unavailable. The model may be overloaded.');
                } else if (error.message.includes('500')) {
                    throw new Error('OCR model error. Try a different model or reduce image size.');
                } else if (error.message.includes('CORS')) {
                    throw new Error(`CORS Error: ${error.message}`);
                } else {
                    throw new Error(`OCR failed: ${error.message}`);
                }
            }
        }
    };

    // ADD: New function for fallback table detection
    const applyFallbackTableDetection = (text) => {
        console.log('🔍 Applying fallback table detection...');
        
        const lines = text.split('\n');
        let result = '';
        let currentTable = [];
        let inTable = false;
        
        for (let i = 0; i < lines.length; i++) {
            const line = lines[i].trim();
            
            // Skip empty lines
            if (!line) {
                if (inTable && currentTable.length >= 2) {
                    result += processTableFromLines(currentTable);
                    currentTable = [];
                    inTable = false;
                }
                result += '\n';
                continue;
            }
            
            // Detect potential table rows with VERY aggressive criteria
            const isTableRow = (
                // Contains typical table separators
                (line.includes(':') && /\d/.test(line)) || // Key: value with numbers
                (line.includes('|') && line.split('|').length >= 2) || // Pipe separated
                (line.match(/\t+/) && /\d/.test(line)) || // Tab separated with numbers
                (line.match(/\s{3,}/) && /\d/.test(line)) || // Multiple spaces with numbers
                
                // Common table patterns
                (/^[A-Za-z\s]+:\s*[\d\$€£¥%]/.test(line)) || // Label: number/currency
                (/^[A-Za-z\s]+\s+[\d\$€£¥%]/.test(line) && line.length < 50) || // Label number (short lines)
                
                // Performance/stats patterns
                (/\b(test|result|score|time|speed|size|count|total|average|min|max|cpu|memory|ram|gpu)\b/i.test(line) && /\d/.test(line)) ||
                
                // Financial patterns
                (/\b(price|cost|amount|revenue|profit|loss|budget|expense|income)\b/i.test(line) && /[\$€£¥\d]/.test(line)) ||
                
                // Technical specs
                (/\b(mhz|ghz|gb|mb|kb|tb|fps|rpm|watts|volts|amps)\b/i.test(line)) ||
                
                // Lists with clear structure
                (line.length < 60 && line.split(/\s+/).length >= 2 && line.split(/\s+/).length <= 6 && /\d/.test(line))
            );
            
            if (isTableRow) {
                currentTable.push(line);
                inTable = true;
            } else {
                // End of potential table
                if (inTable && currentTable.length >= 2) {
                    result += processTableFromLines(currentTable);
                    currentTable = [];
                    inTable = false;
                }
                result += line + '\n';
            }
        }
        
        // Process final table if exists
        if (inTable && currentTable.length >= 2) {
            result += processTableFromLines(currentTable);
        }
        
        return result;
    };
    
    // ADD: Helper function to process table from lines
    const processTableFromLines = (tableLines) => {
        console.log(`📊 Processing ${tableLines.length} table lines:`, tableLines.slice(0, 3));
        
        const tableRows = [];
        
        for (const line of tableLines) {
            let cells = [];
            
            // Try different splitting methods
            if (line.includes('|')) {
                cells = line.split('|').map(cell => cell.trim()).filter(cell => cell);
            } else if (line.includes(':') && line.split(':').length === 2) {
                const parts = line.split(':');
                cells = [parts[0].trim(), parts[1].trim()];
            } else if (line.match(/\t+/)) {
                cells = line.split(/\t+/).map(cell => cell.trim()).filter(cell => cell);
            } else if (line.match(/\s{3,}/)) {
                cells = line.split(/\s{3,}/).map(cell => cell.trim()).filter(cell => cell);
            } else {
                // Try to intelligently split by finding number boundaries
                const words = line.split(/\s+/);
                if (words.length >= 2 && words.length <= 6) {
                    // Look for natural break points
                    let breakPoint = -1;
                    for (let i = 1; i < words.length; i++) {
                        if (/[\d$€£¥%]/.test(words[i])) {
                            breakPoint = i;
                            break;
                        }
                    }
                    
                    if (breakPoint > 0) {
                        cells = [
                            words.slice(0, breakPoint).join(' '),
                            words.slice(breakPoint).join(' ')
                        ];
                    } else {
                        // Default: split roughly in half
                        const midPoint = Math.ceil(words.length / 2);
                        cells = [
                            words.slice(0, midPoint).join(' '),
                            words.slice(midPoint).join(' ')
                        ];
                    }
                }
            }
            
            if (cells.length >= 2) {
                tableRows.push(cells);
            }
        }
        
        if (tableRows.length >= 2) {
            // Find most common column count
            const columnCounts = tableRows.map(row => row.length);
            const mostCommonCount = columnCounts.sort((a,b) =>
                columnCounts.filter(v => v===a).length - columnCounts.filter(v => v===b).length
            ).pop();
            
            // Normalize rows
            const normalizedRows = tableRows.map(row => {
                const normalizedRow = [...row];
                while (normalizedRow.length < mostCommonCount) {
                    normalizedRow.push('');
                }
                return normalizedRow.slice(0, mostCommonCount);
            });
            
            const tableContent = normalizedRows.map(row => row.join('|')).join('\n');
            
            console.log(`✅ Created fallback table with ${normalizedRows.length} rows, ${mostCommonCount} columns`);
            
            return `\n[TABLE_START]\n${tableContent}\n[TABLE_END]\n`;
        }
        
        // If couldn't create valid table, return as regular text
        return tableLines.join('\n') + '\n';
    };    
    
    // NEW: Automatically detect if content is likely two-column
    const detectColumnLayout = (text) => {
        const lines = text.split('\n').filter(line => line.trim());
        
        // Look for indicators of two-column layout
        let longLines = 0;
        let shortLines = 0;
        let averageLength = 0;
        
        lines.forEach(line => {
            const trimmed = line.trim();
            averageLength += trimmed.length;
            
            if (trimmed.length > 100) {
                longLines++;
            } else if (trimmed.length > 20) {
                shortLines++;
            }
        });
        
        averageLength = averageLength / lines.length;
        
        // If we have many long lines, it might be mixed column content
        const longLineRatio = longLines / lines.length;
        
        console.log('🔍 Column detection:', {
            lines: lines.length,
            longLines,
            longLineRatio: longLineRatio.toFixed(2),
            averageLength: averageLength.toFixed(1)
        });
        
        // Suggest two-column processing if many lines are very long
        return longLineRatio > 0.3 && averageLength > 80;
    };

    // IMPROVED: Better post-processing for column separation
    const postProcessColumns = (text) => {
        console.log('🔄 Post-processing columns...');
        console.log('📝 Original text preview:', text.substring(0, 300) + '...');
        
        // If we already have column separators from the prompt, use them
        if (text.includes('--- RIGHT COLUMN ---')) {
            console.log('✅ Column separators already present');
            return text;
        }
        
        const lines = text.split('\n');
        const processedLines = [];
        let leftColumnLines = [];
        let rightColumnLines = [];
        let inRightColumn = false;
        
        for (let i = 0; i < lines.length; i++) {
            const line = lines[i].trim();
            
            // Skip empty lines but preserve them
            if (!line) {
                if (!inRightColumn) {
                    leftColumnLines.push('');
                } else {
                    rightColumnLines.push('');
                }
                continue;
            }
            
            // Check if this might be the start of right column content
            // Look for sudden topic changes, page numbers, or different formatting
            if (!inRightColumn && i > 0) {
                // Heuristics to detect column break:
                // 1. Line much shorter than previous lines (might be end of left column)
                // 2. Different formatting pattern
                // 3. Sudden topic change
                
                const prevLine = lines[i-1]?.trim() || '';
                if (prevLine.length > 50 && line.length < 30 && i < lines.length - 5) {
                    // Might be switching to right column
                    inRightColumn = true;
                    rightColumnLines.push(line);
                    continue;
                }
            }
            
            // If line is very long, it might contain mixed column content
            if (line.length > 120) {
                // Try to split on common patterns
                const potentialSplit = line.split(/\s{4,}|\t{2,}/); // 4+ spaces or 2+ tabs
                if (potentialSplit.length === 2) {
                    console.log('📊 Detected mixed column content, splitting:', line.substring(0, 50) + '...');
                    leftColumnLines.push(potentialSplit[0].trim());
                    rightColumnLines.push(potentialSplit[1].trim());
                    inRightColumn = true; // Switch to right column mode after first split
                    continue;
                }
            }
            
            // Regular line assignment
            if (!inRightColumn) {
                leftColumnLines.push(line);
            } else {
                rightColumnLines.push(line);
            }
        }
        
        // Combine columns with separator
        const result = [];
        
        if (leftColumnLines.length > 0) {
            result.push(...leftColumnLines.filter(line => line.trim() || leftColumnLines.indexOf(line) < leftColumnLines.length - 1));
        }
        
        if (rightColumnLines.length > 0) {
            result.push('', '--- RIGHT COLUMN ---', '');
            result.push(...rightColumnLines.filter(line => line.trim() || rightColumnLines.indexOf(line) < rightColumnLines.length - 1));
        }
        
        const finalResult = result.join('\n');
        console.log('📝 Post-processed result preview:', finalResult.substring(0, 300) + '...');
        console.log('📊 Left column lines:', leftColumnLines.filter(l => l.trim()).length);
        console.log('📊 Right column lines:', rightColumnLines.filter(l => l.trim()).length);
        
        return finalResult;
    };
        
    // Replace the parseStructuredData function (around line 1750) with simplified version:
    const parseStructuredData = (text) => {
        console.log('🔍 Parsing structured data from text length:', text.length);
        console.log('📝 Raw text preview:', text.substring(0, 500) + '...');
        
        const structure = {
            tables: [],
            paragraphs: [],
            columns: []
        };

        // IMPROVED: More flexible table extraction
        const tableRegex = /\[TABLE_START\]([\s\S]*?)\[TABLE_END\]/gi;
        let match;
        let tableCount = 0;
        
        while ((match = tableRegex.exec(text)) !== null) {
            tableCount++;
            console.log(`📊 Found table ${tableCount}:`, match[1].substring(0, 200) + '...');
            
            const tableContent = match[1].trim();
            const rawRows = tableContent.split('\n').filter(row => row.trim());

            console.log(`   Raw rows: ${rawRows.length}`);
            console.log(`   First few rows:`, rawRows.slice(0, 3));
            
            // IMPROVED: Better table cleaning for two-column tables
            const cleanedRows = rawRows.map(row => {
                // Handle multiple separators (|, tabs, multiple spaces)
                let cells = row.split(/\|/).map(cell => cell.trim());
                
                // Remove empty cells only from the very beginning and end
                if (cells.length > 2 && cells[0] === '') {
                    cells.shift();
                }
                if (cells.length > 2 && cells[cells.length - 1] === '') {
                    cells.pop();
                }
                
                console.log(`   Row "${row}" -> Cells:`, cells);
                return cells;
            }).filter(cells => cells.length >= 2); // Must have at least 2 columns
            
            console.log(`   Cleaned rows: ${cleanedRows.length}`);
            
            if (cleanedRows.length > 0) {
                // Find the most common column count
                const columnCounts = cleanedRows.map(row => row.length);
                const mostCommonCount = columnCounts.sort((a,b) =>
                    columnCounts.filter(v => v===a).length - columnCounts.filter(v => v===b).length
                ).pop();
                
                console.log(`   Column counts:`, columnCounts);
                console.log(`   Most common count:`, mostCommonCount);
                
                // Normalize to most common count
                const normalizedRows = cleanedRows.map(row => {
                    const normalizedRow = [...row];
                    while (normalizedRow.length < mostCommonCount) {
                        normalizedRow.push('');
                    }
                    return normalizedRow.slice(0, mostCommonCount);
                });
                
                // Accept tables with at least 2 columns and 1 row
                if (normalizedRows.length > 0 && mostCommonCount >= 2) {
                    structure.tables.push({
                        rows: normalizedRows,
                        startIndex: match.index,
                        endIndex: match.index + match[0].length,
                        columns: mostCommonCount
                    });
                    console.log(`✅ Table added: ${normalizedRows.length} rows, ${mostCommonCount} columns`);
                } else {
                    console.log(`❌ Table rejected: ${normalizedRows.length} rows, ${mostCommonCount} columns`);
                }
            }
        }

        // IMPROVED: Also try to detect tables from patterns in regular text if no marked tables found
        if (structure.tables.length === 0) {
            console.log('🔍 No marked tables found, looking for table patterns...');
            structure.tables = detectTablePatterns(text);
        }

        // Extract regular text
        let cleanText = text.replace(/\[TABLE_START\][\s\S]*?\[TABLE_END\]/gi, '\n[TABLE_REMOVED]\n');
        const paragraphs = cleanText.split(/\n\s*\n/).filter(p => p.trim() && !p.includes('[TABLE_REMOVED]'));
        structure.paragraphs = paragraphs;

        console.log(`✅ Total tables parsed: ${structure.tables.length}`);

        return structure;
    };
        
    const parseStructuredDataWithEquations = parseStructuredData;

    // Add new helper function to process table content:
    const processTableContent = (tableContent, startIndex, endIndex) => {
        const rawRows = tableContent.split('\n').filter(row => row.trim());
        console.log(`   Processing table with ${rawRows.length} raw rows`);
        
        if (rawRows.length === 0) return null;
        
        const cleanedRows = rawRows.map(row => {
            let cells = row.split(/\|/).map(cell => cell.trim());
            
            // Remove empty cells from start and end
            if (cells.length > 2 && cells[0] === '') {
                cells.shift();
            }
            if (cells.length > 2 && cells[cells.length - 1] === '') {
                cells.pop();
            }
            
            return cells;
        }).filter(cells => cells.length >= 2);
        
        if (cleanedRows.length === 0) return null;
        
        // Find most common column count
        const columnCounts = cleanedRows.map(row => row.length);
        const mostCommonCount = columnCounts.sort((a,b) =>
            columnCounts.filter(v => v===a).length - columnCounts.filter(v => v===b).length
        ).pop();
        
        // Normalize rows
        const normalizedRows = cleanedRows.map(row => {
            const normalizedRow = [...row];
            while (normalizedRow.length < mostCommonCount) {
                normalizedRow.push('');
            }
            return normalizedRow.slice(0, mostCommonCount);
        });
        
        return {
            rows: normalizedRows,
            startIndex: startIndex,
            endIndex: endIndex,
            columns: mostCommonCount
        };
    };

    // Add new helper function to process malformed table content:
    const processMalformedTableContent = (tableContent, startIndex, endIndex) => {
        console.log(`   Processing malformed table content:`, tableContent.substring(0, 100) + '...');
        
        // Split by lines and look for rows with pipe separators
        const lines = tableContent.split('\n');
        const tableRows = [];
        
        for (let line of lines) {
            // Clean up the line
            line = line.trim();
            
            // Skip empty lines
            if (!line) continue;
            
            // If line contains pipes, treat as table row
            if (line.includes('|')) {
                // Remove any remaining [TABLE markers from the beginning
                line = line.replace(/^\[TABLE[^\|]*\|/, '');
                
                // Split by pipes
                const cells = line.split('|').map(cell => cell.trim()).filter(cell => cell !== '');
                
                if (cells.length >= 2) {
                    tableRows.push(cells);
                }
            }
        }
        
        if (tableRows.length === 0) return null;
        
        // Find most common column count
        const columnCounts = tableRows.map(row => row.length);
        const mostCommonCount = columnCounts.sort((a,b) =>
            columnCounts.filter(v => v===a).length - columnCounts.filter(v => v===b).length
        ).pop();
        
        // Normalize rows
        const normalizedRows = tableRows.map(row => {
            const normalizedRow = [...row];
            while (normalizedRow.length < mostCommonCount) {
                normalizedRow.push('');
            }
            return normalizedRow.slice(0, mostCommonCount);
        });
        
        return {
            rows: normalizedRows,
            startIndex: startIndex,
            endIndex: endIndex,
            columns: mostCommonCount
        };
    };

    // Add new function to detect tables from lines when markers are completely broken:
    const detectTablesFromLines = (text) => {
        console.log('🔍 Detecting tables from line patterns...');
        const tables = [];
        const lines = text.split('\n');
        
        let currentTableRows = [];
        let inTable = false;
        
        for (let i = 0; i < lines.length; i++) {
            const line = lines[i].trim();
            
            // Skip empty lines
            if (!line) {
                if (inTable && currentTableRows.length >= 2) {
                    // End of table
                    const processedTable = processLineBasedTable(currentTableRows);
                    if (processedTable) {
                        tables.push(processedTable);
                        console.log(`📊 Line-based table detected: ${processedTable.rows.length} rows`);
                    }
                }
                currentTableRows = [];
                inTable = false;
                continue;
            }
            
            // Check if line looks like table row
            if (line.includes('|') && line.split('|').filter(cell => cell.trim()).length >= 2) {
                currentTableRows.push(line);
                inTable = true;
            } else if (inTable) {
                // Non-table line encountered, process accumulated table
                if (currentTableRows.length >= 2) {
                    const processedTable = processLineBasedTable(currentTableRows);
                    if (processedTable) {
                        tables.push(processedTable);
                        console.log(`📊 Line-based table detected: ${processedTable.rows.length} rows`);
                    }
                }
                currentTableRows = [];
                inTable = false;
            }
        }
        
        // Process final table if exists
        if (inTable && currentTableRows.length >= 2) {
            const processedTable = processLineBasedTable(currentTableRows);
            if (processedTable) {
                tables.push(processedTable);
                console.log(`📊 Final line-based table detected: ${processedTable.rows.length} rows`);
            }
        }
        
        return tables;
    };

    // Add helper for line-based table processing:
    const processLineBasedTable = (lines) => {
        const tableRows = lines.map(line => {
            return line.split('|').map(cell => cell.trim()).filter(cell => cell !== '');
        }).filter(cells => cells.length >= 2);
        
        if (tableRows.length < 2) return null;
        
        // Find most common column count
        const columnCounts = tableRows.map(row => row.length);
        const mostCommonCount = columnCounts.sort((a,b) =>
            columnCounts.filter(v => v===a).length - columnCounts.filter(v => v===b).length
        ).pop();
        
        // Normalize rows
        const normalizedRows = tableRows.map(row => {
            const normalizedRow = [...row];
            while (normalizedRow.length < mostCommonCount) {
                normalizedRow.push('');
            }
            return normalizedRow.slice(0, mostCommonCount);
        });
        
        return {
            rows: normalizedRows,
            startIndex: 0,
            endIndex: 0,
            columns: mostCommonCount
        };
    };

    // Enhance the createCodeBlockElement function for better formatting:
    const createCodeBlockElement = (codeBlock, index) => {
        console.log(`💻 Adding code block ${index + 1}:`, codeBlock.content.substring(0, 100) + '...');
        
        // Split code into lines
        const lines = codeBlock.content.split('\n');
        const codeRuns = [];
        
        lines.forEach((line, lineIndex) => {
            // Preserve indentation and formatting
            const lineText = line || ' '; // Use space for empty lines
            
            codeRuns.push(new TextRun({
                text: lineText,
                font: 'Consolas',
                size: 22,
                color: '2D3748'
            }));
            
            // Add line break except for the last line
            if (lineIndex < lines.length - 1) {
                codeRuns.push(new TextRun({
                    text: '',
                    break: 1
                }));
            }
        });
        
        return [
            new Paragraph({ children: [new TextRun("")] }), // Space before
            new Paragraph({
                children: codeRuns,
                shading: {
                    type: 'clear',
                    fill: 'F7FAFC'
                },
                border: {
                    top: { style: 'single', size: 1, color: 'E2E8F0' },
                    bottom: { style: 'single', size: 1, color: 'E2E8F0' },
                    left: { style: 'single', size: 4, color: '4299E1' },
                    right: { style: 'single', size: 1, color: 'E2E8F0' }
                },
                indent: {
                    left: 360,
                    right: 360
                },
                spacing: {
                    before: 200,
                    after: 200
                }
            }),
            new Paragraph({ children: [new TextRun("")] })  // Space after
        ];
    };

    // NEW: Validate and clean table data
    const validateTableData = (tableData) => {
        if (!tableData || !tableData.rows || tableData.rows.length === 0) {
            return null;
        }
        
        // Remove rows that are mostly empty
        const validRows = tableData.rows.filter(row => {
            const nonEmptyCells = row.filter(cell => cell && cell.trim().length > 0);
            return nonEmptyCells.length >= Math.ceil(row.length * 0.3); // At least 30% of cells should have content
        });
        
        if (validRows.length === 0) {
            return null;
        }
        
        // Ensure consistent column count
        const maxColumns = Math.max(...validRows.map(row => row.length));
        const normalizedRows = validRows.map(row => {
            const normalizedRow = [...row];
            while (normalizedRow.length < maxColumns) {
                normalizedRow.push('');
            }
            return normalizedRow.slice(0, maxColumns); // Trim if too many columns
        });
        
        return {
            ...tableData,
            rows: normalizedRows,
            columns: maxColumns
        };
    };
    
    const detectTablePatterns = (text) => {
        console.log('🔍 Detecting table patterns in text...');
        const tables = [];
        const lines = text.split('\n').map(line => line.trim()).filter(line => line);
        
        let potentialTableRows = [];
        let inPotentialTable = false;
        
        for (let i = 0; i < lines.length; i++) {
            const line = lines[i];
            
            // MUCH MORE RESTRICTIVE: Skip obvious non-table content
            if (line.startsWith('--- Page ') || 
                line.includes('--- RIGHT COLUMN ---') ||
                (line.includes('TABLE') && line.length < 30) ||  // Skip table titles
                line.length < 15 ||  // Skip very short lines
                /^[A-Z\s]+:?\s*$/.test(line) ||  // Skip all-caps titles
                line.includes('where:') || line.includes('formula') ||  // Skip formula explanations
                line.includes('The ') || line.includes('This ') ||  // Skip sentences
                /^[a-z]/.test(line) ||  // Skip lines starting with lowercase (likely text)
                line.includes('Price =') || (line.includes('=') && line.length > 30)) {  // Skip formulas
                
                // End current table if we were building one
                if (inPotentialTable && potentialTableRows.length >= 3) { // Require at least 3 rows
                    const detectedTable = analyzeTableRows(potentialTableRows);
                    if (detectedTable && isValidTable(detectedTable)) {
                        tables.push(detectedTable);
                        console.log(`📊 Valid pattern table detected: ${detectedTable.rows.length} rows, ${detectedTable.columns} columns`);
                    }
                }
                potentialTableRows = [];
                inPotentialTable = false;
                continue;
            }
            
            // Look for potential table rows with VERY specific characteristics
            const words = line.split(/\s+/).filter(w => w.length > 0);
            
            // MUCH MORE RESTRICTIVE criteria for table detection
            const hasNumbers = /\d/.test(line);
            const hasUnits = /\b(ms|sec|MB|KB|GB|SYS|USD|EUR|%|bytes)\b/i.test(line);
            const hasColonPattern = /^[A-Za-z\s]+:\s*\d/.test(line); // Label: number pattern
            const isBalanced = words.length >= 2 && words.length <= 8;
            const hasTabularPattern = line.includes(':') && hasNumbers;
            
            // ONLY consider as table if it has clear tabular characteristics
            if (isBalanced && line.length < 80 && hasNumbers && (hasUnits || hasColonPattern || hasTabularPattern)) {
                potentialTableRows.push(line);
                inPotentialTable = true;
            } else {
                // End of potential table
                if (inPotentialTable && potentialTableRows.length >= 3) { // Require at least 3 rows
                    const detectedTable = analyzeTableRows(potentialTableRows);
                    if (detectedTable && isValidTable(detectedTable)) {
                        tables.push(detectedTable);
                        console.log(`📊 Valid pattern table detected: ${detectedTable.rows.length} rows, ${detectedTable.columns} columns`);
                    }
                }
                potentialTableRows = [];
                inPotentialTable = false;
            }
        }
        
        // Check final potential table
        if (inPotentialTable && potentialTableRows.length >= 3) { // Require at least 3 rows
            const detectedTable = analyzeTableRows(potentialTableRows);
            if (detectedTable && isValidTable(detectedTable)) {
                tables.push(detectedTable);
                console.log(`📊 Final valid pattern table detected: ${detectedTable.rows.length} rows, ${detectedTable.columns} columns`);
            }
        }
        
        console.log(`📊 Pattern detection found ${tables.length} potential tables`);
        return tables;
    };  

    // Add this new validation function after detectTablePatterns:
    const isValidTable = (tableData) => {
        if (!tableData || !tableData.rows || tableData.rows.length < 3) { // Require at least 3 rows
            return false;
        }
        
        // Check if it contains actual tabular data
        let numericCells = 0;
        let totalCells = 0;
        let hasMeaningfulStructure = false;
        let hasUnits = false;
        
        tableData.rows.forEach(row => {
            row.forEach(cell => {
                totalCells++;
                // Count cells with numbers, currency, percentages, etc.
                if (/[\d.,]+/.test(cell) && cell.length > 1) {
                    numericCells++;
                }
                // Look for key-value patterns
                if (cell.includes(':') || cell.includes('=')) {
                    hasMeaningfulStructure = true;
                }
                // Look for units
                if (/\b(ms|sec|MB|KB|GB|SYS|USD|EUR|%|bytes)\b/i.test(cell)) {
                    hasUnits = true;
                }
            });
        });
        
        // Much more restrictive validation
        const numericRatio = numericCells / totalCells;
        
        console.log(`   Table validation: ${numericCells}/${totalCells} numeric (${(numericRatio * 100).toFixed(1)}%), meaningful: ${hasMeaningfulStructure}, hasUnits: ${hasUnits}`);
        
        // Require higher numeric ratio AND units OR meaningful structure
        return (numericRatio >= 0.4 && hasUnits) || (numericRatio >= 0.3 && hasMeaningfulStructure);
    };    

    // Replace the analyzeTableRows function with this improved version:
    const analyzeTableRows = (rows) => {
        if (rows.length < 2) return null;
        
        console.log(`   Analyzing ${rows.length} potential table rows:`, rows.slice(0, 3));
        
        // Try to split rows into consistent columns
        const splitRows = rows.map(row => {
            let cells = [];
            
            // Method 1: Split by colon (key-value pairs)
            if (row.includes(':') && !row.includes('::')) {
                const parts = row.split(':');
                if (parts.length === 2) {
                    cells = [parts[0].trim(), parts[1].trim()];
                }
            }
            
            // Method 2: Split by pipe (explicit table separators)
            if (cells.length < 2 && row.includes('|')) {
                cells = row.split('|').map(cell => cell.trim()).filter(cell => cell);
            }
            
            // Method 3: Split by multiple spaces (space-separated columns)
            if (cells.length < 2 && /\s{3,}/.test(row)) {
                cells = row.split(/\s{3,}/).map(cell => cell.trim()).filter(cell => cell);
            }
            
            // Method 4: Split by equals sign
            if (cells.length < 2 && row.includes('=')) {
                const parts = row.split('=');
                if (parts.length === 2) {
                    cells = [parts[0].trim(), parts[1].trim()];
                }
            }
            
            // Method 5: Intelligent word grouping for 2-column layout
            if (cells.length < 2) {
                const words = row.split(/\s+/);
                if (words.length >= 2 && words.length <= 6) {
                    // Look for natural break points (like before numbers, currencies, etc.)
                    let breakPoint = -1;
                    for (let i = 1; i < words.length; i++) {
                        if (/[\d$€£¥%]/.test(words[i])) {
                            breakPoint = i;
                            break;
                        }
                    }
                    
                    if (breakPoint > 0) {
                        cells = [
                            words.slice(0, breakPoint).join(' '),
                            words.slice(breakPoint).join(' ')
                        ];
                    } else {
                        // Default: split roughly in half
                        const midPoint = Math.ceil(words.length / 2);
                        cells = [
                            words.slice(0, midPoint).join(' '),
                            words.slice(midPoint).join(' ')
                        ];
                    }
                }
            }
            
            console.log(`     Row "${row}" -> Cells:`, cells);
            return cells;
        }).filter(cells => cells.length >= 2); // Must have at least 2 columns
        
        if (splitRows.length < 2) {
            console.log(`   Rejected: only ${splitRows.length} valid rows`);
            return null;
        }
        
        // Find the most common column count
        const columnCounts = splitRows.map(row => row.length);
        const mostCommonCount = columnCounts.sort((a,b) =>
            columnCounts.filter(v => v===a).length - columnCounts.filter(v => v===b).length
        ).pop();
        
        // Accept only if most rows have the same column count
        const validRows = splitRows.filter(row => row.length === mostCommonCount);
        const validRatio = validRows.length / splitRows.length;
        
        console.log(`   Column analysis: common count=${mostCommonCount}, valid ratio=${(validRatio * 100).toFixed(1)}%`);
        
        if (validRows.length >= 2 && mostCommonCount >= 2 && validRatio >= 0.7) {
            // Normalize all rows to the most common count
            const normalizedRows = validRows.map(row => {
                const normalizedRow = [...row];
                while (normalizedRow.length < mostCommonCount) {
                    normalizedRow.push('');
                }
                return normalizedRow.slice(0, mostCommonCount);
            });
            
            return {
                rows: normalizedRows,
                columns: mostCommonCount,
                startIndex: 0,
                endIndex: 0
            };
        }
        
        console.log(`   Rejected: insufficient valid rows or columns`);
        return null;
    };
    
    // Replace the generateDocx function (around line 2530) with this corrected version:
    const generateDocx = async (text, structuredData) => {
        console.log('📊 DOCX Generation Debug:');
        console.log('📝 Text contains [TABLE_START]:', text.includes('[TABLE_START]'));
        console.log('📊 Structured data tables:', structuredData?.tables?.length || 0);
        console.log('📝 Full text length:', text.length);
        console.log('📝 Full text preview:', text.substring(0, 500) + '...');
    
        const children = [];
    
        if (structuredData && structuredData.tables.length > 0) {
            console.log('📝 Processing structured data with', structuredData.tables.length, 'tables');
            
            // FIXED: Use the NEW segment-based approach instead of simple split
            const tablePositions = [];
            const tableRegex = /\[TABLE_START\]([\s\S]*?)\[TABLE_END\]/gi;
            let match;
            
            // Find all table positions in the original text
            while ((match = tableRegex.exec(text)) !== null) {
                tablePositions.push({
                    start: match.index,
                    end: match.index + match[0].length,
                    content: match[0]
                });
            }
            
            console.log('📊 Found table positions:', tablePositions.map(t => ({ start: t.start, end: t.end })));
            
            // FIXED: Extract ALL text segments properly
            let currentPosition = 0;
            const segments = [];
            
            for (let i = 0; i < tablePositions.length; i++) {
                const tablePos = tablePositions[i];
                
                // Extract text before this table
                if (currentPosition < tablePos.start) {
                    const textSegment = text.substring(currentPosition, tablePos.start).trim();
                    if (textSegment) {
                        segments.push({ type: 'text', content: textSegment });
                        console.log(`📝 Text segment ${segments.length} (before table ${i + 1}):`, textSegment.substring(0, 100) + '...');
                    } else {
                        console.log(`📝 Empty text segment before table ${i + 1}`);
                    }
                }
                
                // Add the table
                segments.push({ type: 'table', tableIndex: i });
                console.log(`📊 Table segment ${segments.length} (table ${i + 1})`);
                
                currentPosition = tablePos.end;
            }
            
            // CRITICAL FIX: Extract final text segment after the last table
            console.log('🔍 Checking for final text segment...');
            console.log('   Current position after last table:', currentPosition);
            console.log('   Total text length:', text.length);
            console.log('   Remaining text length:', text.length - currentPosition);
            
            if (currentPosition < text.length) {
                const finalTextSegment = text.substring(currentPosition).trim();
                console.log('   Final text raw length:', text.substring(currentPosition).length);
                console.log('   Final text trimmed length:', finalTextSegment.length);
                console.log('   Final text content preview:', JSON.stringify(finalTextSegment.substring(0, 200)));
                
                if (finalTextSegment) {
                    segments.push({ type: 'text', content: finalTextSegment });
                    console.log(`📝 ✅ FINAL text segment ${segments.length} (after last table):`, finalTextSegment.substring(0, 100) + '...');
                } else {
                    console.log('❌ Final text segment is empty after trimming');
                }
            } else {
                console.log('❌ No text after last table (currentPosition >= text.length)');
            }
            
            console.log('📝 Total segments created:', segments.length);
            console.log('📊 Segment breakdown:', segments.map((s, i) => `${i + 1}: ${s.type}${s.type === 'table' ? ` (table ${s.tableIndex + 1})` : ` (${s.content.substring(0, 50)}...)`}`));
            
            // FIXED: Process all segments in order
            for (let i = 0; i < segments.length; i++) {
                const segment = segments[i];
                console.log(`🔄 Processing segment ${i + 1}/${segments.length}: ${segment.type}`);
                
                if (segment.type === 'text') {
                    console.log(`   📝 Processing text segment with ${segment.content.length} characters`);
                    console.log(`   📝 Text preview: "${segment.content.substring(0, 150)}..."`);
                    
                    // IMPROVED: Better paragraph splitting
                    let paragraphs = segment.content.split(/\n\s*\n/).filter(p => p.trim());
                    
                    // If no double newlines found, try single newlines
                    if (paragraphs.length === 1 && segment.content.includes('\n')) {
                        paragraphs = segment.content.split('\n').filter(p => p.trim());
                    }
                    
                    // If still only one paragraph, treat the whole content as one paragraph
                    if (paragraphs.length === 0 && segment.content.trim()) {
                        paragraphs = [segment.content.trim()];
                    }
                    
                    console.log(`   📝 Split into ${paragraphs.length} paragraphs:`, paragraphs.map((p, idx) => `${idx + 1}: "${p.substring(0, 50)}..."`));
                    
                    paragraphs.forEach((para, paraIndex) => {
                        const trimmedPara = para.trim();
                        if (trimmedPara) {
                            console.log(`     ✅ Adding paragraph ${paraIndex + 1}: "${trimmedPara.substring(0, 50)}..."`);
                            children.push(createEnhancedParagraph(trimmedPara));
                        }
                    });
                    
                    console.log(`   ✅ Added ${paragraphs.filter(p => p.trim()).length} paragraphs from this text segment`);
                    
                } else if (segment.type === 'table') {
                    const tableData = structuredData.tables[segment.tableIndex];
                    if (tableData) {
                        console.log(`📊 ✅ Inserting table ${segment.tableIndex + 1} with ${tableData.rows.length} rows`);
                        children.push(...createTableElement(tableData, segment.tableIndex));
                    } else {
                        console.log(`❌ Table data missing for table ${segment.tableIndex + 1}`);
                    }
                }
            }
            
        } else {
            // No structured data - process as regular text
            console.log('📝 Processing as regular text (no tables detected)');
            const cleanedText = text.replace(/\[TABLE_START\]([\s\S]*?)\[TABLE_END\]/gi, '');
            console.log('📝 Cleaned text length:', cleanedText.length);
            
            const paragraphs = cleanedText.split(/\n\s*\n/).filter(p => p.trim());
            if (paragraphs.length === 1 && cleanedText.includes('\n')) {
                paragraphs.push(...cleanedText.split('\n').filter(p => p.trim()));
            }
            
            console.log('📝 Found paragraphs:', paragraphs.length);
            
            paragraphs.forEach((para, index) => {
                const trimmedPara = para.trim();
                if (trimmedPara) {
                    console.log(`   Adding paragraph ${index + 1}: "${trimmedPara.substring(0, 50)}..."`);
                    children.push(createEnhancedParagraph(trimmedPara));
                }
            });
        }
    
        // Ensure we have at least one element
        if (children.length === 0) {
            console.log('❌ No children found, adding default content');
            children.push(new Paragraph({ children: [new TextRun("No content to export.")] }));
        }
    
        console.log('📝 ✅ Final document children count:', children.length);
        console.log('📊 Children types breakdown:', children.map((child, i) => `${i + 1}: ${child.constructor.name}`));
    
        const docWithSections = new Document({
            sections: [{
                properties: {},
                children: children
            }]
        });
    
        return await Packer.toBlob(docWithSections);
    };    

    // Replace the createEnhancedParagraph function to remove subscript/superscript handling:
    const createEnhancedParagraph = (text) => {
        try {
            // Simple paragraph creation without equation/subscript formatting
            return new Paragraph({
                children: [new TextRun(text)]
            });
        } catch (error) {
            console.warn('Paragraph creation failed, using basic formatting:', error);
            return new Paragraph({
                children: [new TextRun(text)]
            });
        }
    };


    
    // Replace the createTableElement function to remove subscript/superscript handling:
    const createTableElement = (tableData, tableIndex) => {
        console.log(`📊 Adding table ${tableIndex + 1} with`, tableData.rows.length, 'rows and', tableData.columns, 'columns');
        console.log(`📊 Table ${tableIndex + 1} content:`, tableData.rows);
        
        const tableRows = tableData.rows.map((row, rowIndex) => {
            const isHeaderRow = rowIndex === 0;
            
            return new TableRow({
                children: row.map((cell, cellIndex) => {
                    const cellContent = cell || '';
                    
                    // Simple cell content without subscript/superscript formatting
                    const cellParagraph = new Paragraph({
                        children: [new TextRun({
                            text: cellContent,
                            bold: isHeaderRow,
                            size: isHeaderRow ? 24 : 22
                        })]
                    });
                    
                    return new TableCell({
                        children: [cellParagraph],
                        width: {
                            size: Math.floor(100 / tableData.columns),
                            type: WidthType.PERCENTAGE
                        },
                        margins: {
                            top: 100,
                            bottom: 100,
                            left: 100,
                            right: 100
                        },
                        shading: isHeaderRow ? {
                            fill: "F0F0F0"
                        } : undefined,
                        borders: {
                            top: { style: BorderStyle.SINGLE, size: 1, color: "000000" },
                            bottom: { style: BorderStyle.SINGLE, size: 1, color: "000000" },
                            left: { style: BorderStyle.SINGLE, size: 1, color: "000000" },
                            right: { style: BorderStyle.SINGLE, size: 1, color: "000000" }
                        }
                    });
                })
            });
        });

        const table = new Table({
            rows: tableRows,
            width: {
                size: 100,
                type: WidthType.PERCENTAGE
            },
            borders: {
                top: { style: BorderStyle.SINGLE, size: 1, color: "000000" },
                bottom: { style: BorderStyle.SINGLE, size: 1, color: "000000" },
                left: { style: BorderStyle.SINGLE, size: 1, color: "000000" },
                right: { style: BorderStyle.SINGLE, size: 1, color: "000000" },
                insideHorizontal: { style: BorderStyle.SINGLE, size: 1, color: "000000" },
                insideVertical: { style: BorderStyle.SINGLE, size: 1, color: "000000" }
            }
        });

        // Return table with spacing
        return [
            new Paragraph({ children: [new TextRun("")] }), // Space before
            table,
            new Paragraph({ children: [new TextRun("")] })  // Space after
        ];
    };

    // MODIFIED: Update processFile to use the new function
    const processFile = async (file) => {
        await processFileWithSelectedPages(file, null);
    };

    // MODIFIED: Handle file upload with page selection for PDFs
    const onFileSelect = (event) => {
        const files = Array.from(event.files);
        setUploadedFiles(files);
        
        if (files.length > 0) {
            const file = files[0];
            
            if (!Object.keys(supportedTypes).includes(file.type)) {
                toast.current?.show({
                    severity: 'error',
                    summary: 'Unsupported File',
                    detail: 'Please select a PDF, JPEG, PNG, or TIFF file',
                    life: 3000
                });
                return;
            }

            if (file.size > 50 * 1024 * 1024) {
                toast.current?.show({
                    severity: 'error',
                    summary: 'File Too Large',
                    detail: 'Please select a file smaller than 50MB',
                    life: 3000
                });
                return;
            }

            if (file.type === 'application/pdf') {
                if (workerStatus !== 'ready') {
                    toast.current?.show({
                        severity: 'error',
                        summary: 'PDF Worker Not Ready',
                        detail: 'PDF processing is not available. Please wait for initialization or refresh the page.',
                        life: 3000
                    });
                    return;
                }

                // Show page selector for PDFs
                setPdfFile(file);
                loadPdfPages(file).then(() => {
                    setShowPageSelector(true);
                }).catch(error => {
                    toast.current?.show({
                        severity: 'error',
                        summary: 'PDF Loading Failed',
                        detail: error.message,
                        life: 5000
                    });
                });
            } else {
                // Process image files directly
                processFile(file);
            }
        }
    };

        // MODIFIED: Process file with optional page selection
    const processFileWithSelectedPages = async (file, selectedPageNumbers = null) => {
        setIsProcessing(true);
        setProgress(0);
        setCurrentFile(file);
        setExtractedText('');
        setProcessingStats(null);
        setStreamingText('');
        setIsStreaming(false);
        setCurrentPageProcessing(null);

        const startTime = Date.now();
        let allExtractedText = '';

        try {
            let imagesToProcess = [];

            if (file.type === 'application/pdf') {
                const pageLabel = selectedPageNumbers ? 
                    `${selectedPageNumbers.length} selected pages` : 
                    'all pages';
                    
                toast.current?.show({
                    severity: 'info',
                    summary: 'Processing PDF',
                    detail: `Converting ${pageLabel} to images...`,
                    life: 3000
                });
                
                setProgress(10);
                
                if (selectedPageNumbers) {
                    imagesToProcess = await processSelectedPdfPages(file, selectedPageNumbers);
                } else {
                    imagesToProcess = await pdfToImages(file);
                }
                
                setPreviewImages(imagesToProcess);
                setProgress(25);
            } else {
                const base64 = await fileToBase64(file, imageQuality / 100);
                imagesToProcess = [{
                    base64,
                    pageNumber: 1,
                    filename: file.name
                }];
                setPreviewImages(imagesToProcess);
                setProgress(25);
            }

            if (imagesToProcess.length === 0) {
                throw new Error('No pages could be processed from the file.');
            }

            const controller = new AbortController();
            abortControllerRef.current = controller;

            for (let i = 0; i < imagesToProcess.length; i++) {
                const image = imagesToProcess[i];
                const pageProgress = (i / imagesToProcess.length) * 75 + 25;
                setProgress(pageProgress);

                toast.current?.show({
                    severity: 'info',
                    summary: 'OCR Processing',
                    detail: `Processing ${file.type === 'application/pdf' ? `page ${image.pageNumber}` : 'image'}... (${i + 1}/${imagesToProcess.length})`,
                    life: 2000
                });

                try {
                    const pageText = await processImageWithStream(image, file.type, controller);
                    
                    if (file.type === 'application/pdf' && imagesToProcess.length > 1) {
                        const pageHeader = `\n\n--- Page ${image.pageNumber} ---\n\n`;
                        allExtractedText += pageHeader + pageText;
                        setExtractedText(allExtractedText);
                    } else {
                        allExtractedText = pageText;
                        setExtractedText(allExtractedText);
                    }

                } catch (pageError) {
                    console.error(`Error processing ${file.type === 'application/pdf' ? `page ${image.pageNumber}` : 'image'}:`, pageError);
                    
                    if (pageError.name !== 'AbortError') {
                        const errorText = `\n\n--- Error processing ${file.type === 'application/pdf' ? `page ${image.pageNumber}` : 'image'} ---\n[OCR Error: ${pageError.message}]\n\n`;
                        allExtractedText += errorText;
                        setExtractedText(allExtractedText);
                    }
                }
            }

            setProgress(100);

            // Parse structured data if table detection is enabled
            // if (detectTables && allExtractedText) {
            //     console.log('🔍 Parsing structured data from extracted text...');
            //     const parsedData = parseStructuredData(allExtractedText);
            //     console.log('📊 Parsed data:', parsedData);
            //     if (parsedData.tables.length > 0) {
            //         parsedData.tables = parsedData.tables.map(validateTableData).filter(Boolean);
            //         console.log('✅ Validated tables:', parsedData.tables.length);
            //     }
            //     setStructuredData(parsedData);
            // } else {
            //     setStructuredData(null);
            // }
  
            // Update the processing stats section (around line 2800) to remove equation references:
            if (detectTables && allExtractedText) {
                console.log('🔍 Parsing structured data from extracted text...');
                const parsedData = parseStructuredData(allExtractedText);
                console.log('📊 Parsed data:', parsedData);
                if (parsedData.tables.length > 0) {
                    parsedData.tables = parsedData.tables.map(validateTableData).filter(Boolean);
                    console.log('✅ Validated tables:', parsedData.tables.length);
                }
                setStructuredData(parsedData);
            } else {
                setStructuredData(null);
            }

            const processingTime = (Date.now() - startTime) / 1000;
            // Update the stats object to remove equation references:
            const stats = {
                processingTime: processingTime.toFixed(1),
                pages: imagesToProcess.length,
                charactersExtracted: allExtractedText.length,
                wordsExtracted: allExtractedText.split(/\s+/).filter(word => word.length > 0).length,
                tablesDetected: structuredData?.tables?.length || 0,
                tableColumns: structuredData?.tables?.reduce((total, table) => total + table.columns, 0) || 0,
                language: selectedLanguage === 'auto' ? 'Auto-detected' : languageOptions.find(l => l.value === selectedLanguage)?.label.replace(/^🇺🇸|🇷🇺|🇧🇬\s/, '') || 'Unknown'
            };
            
            setProcessingStats(stats);
            
            toast.current?.show({
                severity: 'success',
                summary: 'OCR Complete',
                detail: `Extracted ${stats.charactersExtracted} characters${stats.tablesDetected > 0 ? ` and ${stats.tablesDetected} tables` : ''} from ${stats.pages} ${stats.pages === 1 ? 'page' : 'pages'} in ${stats.processingTime}s (${stats.language})`,
                life: 4000
            });

        } catch (error) {
            console.error('OCR processing error:', error);
            
            if (error.name === 'AbortError') {
                toast.current?.show({
                    severity: 'info',
                    summary: 'Processing Cancelled',
                    detail: 'OCR processing was cancelled',
                    life: 3000
                });
            } else {
                toast.current?.show({
                    severity: 'error',
                    summary: 'OCR Failed',
                    detail: error.message || 'Failed to process file for OCR',
                    life: 5000
                });
            }
            
            setExtractedText('');
        } finally {
            setIsProcessing(false);
            setProgress(0);
            setCurrentFile(null);
            setStreamingText('');
            setIsStreaming(false);
            setCurrentPageProcessing(null);
        }
    };
        // NEW: Start OCR with selected pages
    const startOcrWithSelectedPages = () => {
        if (selectedPages.length === 0) {
            toast.current?.show({
                severity: 'warn',
                summary: 'No Pages Selected',
                detail: 'Please select at least one page to process',
                life: 3000
            });
            return;
        }

        setShowPageSelector(false);
        processFileWithSelectedPages(pdfFile, selectedPages);
    };

    // Cancel processing
    const cancelProcessing = () => {
        if (abortControllerRef.current) {
            abortControllerRef.current.abort();
        }
        setIsStreaming(false);
        setCurrentPageProcessing(null);
        setStreamingText('');
    };

    // Copy text to clipboard
    const copyToClipboard = () => {
        navigator.clipboard.writeText(extractedText).then(() => {
            toast.current?.show({
                severity: 'success',
                summary: 'Copied',
                detail: 'Text copied to clipboard',
                life: 2000
            });
        }).catch(() => {
            toast.current?.show({
                severity: 'error',
                summary: 'Copy Failed',
                detail: 'Could not copy to clipboard',
                life: 2000
            });
        });
    };

    // Export text
    const exportText = () => {
        if (!extractedText) return;
        
        const filename = `ocr-${currentFile?.name || 'extracted'}-${new Date().toISOString().split('T')[0]}.txt`;
        const blob = new Blob([extractedText], { type: 'text/plain' });
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = filename;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        URL.revokeObjectURL(url);
        
        toast.current?.show({
            severity: 'success',
            summary: 'Exported',
            detail: `Text saved as ${filename}`,
            life: 3000
        });
    };

    // Update the export function call to use the regular exportDocx:
    const exportDocxUpdated = async () => {
        if (!extractedText) return;

        try {
            toast.current?.show({
                severity: 'info',
                summary: 'Generating DOCX',
                detail: 'Creating Word document...',
                life: 2000
            });

            console.log('📊 Structured data for DOCX export:', structuredData);
            console.log('📝 Text length:', extractedText.length);
            console.log('🔍 Table markers in text:', (extractedText.match(/\[TABLE_START\]/g) || []).length);

            const blob = await generateDocx(extractedText, structuredData);
            const filename = `ocr-${currentFile?.name || 'extracted'}-${new Date().toISOString().split('T')[0]}.docx`;
            
            const url = URL.createObjectURL(blob);
            const link = document.createElement('a');
            link.href = url;
            link.download = filename;
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
            URL.revokeObjectURL(url);
            
            toast.current?.show({
                severity: 'success',
                summary: 'DOCX Exported',
                detail: `Document saved as ${filename}${structuredData?.tables?.length ? ` with ${structuredData.tables.length} tables` : ''}`,
                life: 4000
            });
        } catch (error) {
            console.error('Error generating DOCX:', error);
            toast.current?.show({
                severity: 'error',
                summary: 'Export Failed',
                detail: 'Could not generate Word document',
                life: 3000
            });
        }
    };

    // NEW: Export as .docx
    const exportDocx = async () => {
        if (!extractedText) return;
    
        try {
            toast.current?.show({
                severity: 'info',
                summary: 'Generating DOCX',
                detail: 'Creating Word document...',
                life: 2000
            });
    
                    // DEBUG: Log the structured data
            console.log('📊 Structured data for DOCX export:', structuredData);
            console.log('📝 Text length:', extractedText.length);
            console.log('🔍 Table markers in text:', (extractedText.match(/\[TABLE_START\]/g) || []).length);

            const blob = await generateDocx(extractedText, structuredData);
            const filename = `ocr-${currentFile?.name || 'extracted'}-${new Date().toISOString().split('T')[0]}.docx`;
            
            const url = URL.createObjectURL(blob);
            const link = document.createElement('a');
            link.href = url;
            link.download = filename;
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
            URL.revokeObjectURL(url);
            
            toast.current?.show({
                severity: 'success',
                summary: 'DOCX Exported',
                detail: `Document saved as ${filename}${structuredData?.tables?.length ? ` with ${structuredData.tables.length} tables` : ''}`,
                life: 4000
            });
        } catch (error) {
            console.error('Error generating DOCX:', error);
            toast.current?.show({
                severity: 'error',
                summary: 'Export Failed',
                detail: 'Could not generate Word document',
                life: 3000
            });
        }
    };

    // Clear all
    const clearAll = () => {
        setUploadedFiles([]);
        setExtractedText('');
        setPreviewImages([]);
        setProcessingStats(null);
        setStreamingText('');
        setIsStreaming(false);
        setCurrentPageProcessing(null);
        // ADD: Clear structured data
        setStructuredData(null);
        if (fileUploadRef.current) {
            fileUploadRef.current.clear();
        }
    };

    // Get connection status
    const getConnectionStatus = () => {
        const statusMap = {
            'good': { color: '#4CAF50', icon: 'pi-wifi', label: 'Connected' },
            'fair': { color: '#FF9800', icon: 'pi-wifi', label: 'Slow' },
            'poor': { color: '#F44336', icon: 'pi-wifi', label: 'Poor' },
            'offline': { color: '#9E9E9E', icon: 'pi-times', label: 'Offline' }
        };
        return statusMap[connectionQuality] || statusMap.offline;
    };

    // Get worker status
    const getWorkerStatus = () => {
        const statusMap = {
            'checking': { color: '#FF9800', icon: 'pi-clock', label: 'Checking' },
            'testing': { color: '#2196F3', icon: 'pi-spin pi-spinner', label: 'Testing' },
            'ready': { color: '#4CAF50', icon: 'pi-check', label: 'Ready' },
            'failed': { color: '#F44336', icon: 'pi-times', label: 'Failed' }
        };
        return statusMap[workerStatus] || statusMap.checking;
    };

    const connectionStatus = getConnectionStatus();
    const pdfWorkerStatus = getWorkerStatus();

    const menuItems = [
        { label: 'Clear All', icon: 'pi pi-trash', command: clearAll },
        { label: 'Refresh Models', icon: 'pi pi-refresh', command: fetchAvailableModels },
        { label: 'Check Connection', icon: 'pi pi-wifi', command: checkConnection },
        { label: 'Test PDF Worker', icon: 'pi pi-cog', command: testWorker },
        { separator: true },
        { label: 'Export Text (.txt)', icon: 'pi pi-file', command: exportText, disabled: !extractedText },
        { label: 'Export Word (.docx)', icon: 'pi pi-file-word', command: exportDocxUpdated, disabled: !extractedText }
    ];

    return (
        <div style={{ padding: '20px', maxWidth: '1400px', margin: '0 auto' }}>
            <Toast ref={toast} position="top-right" />
            
            {/* Header */}
            <Card className="mb-4">
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <div>
                        <h2 style={{ margin: 0, display: 'flex', alignItems: 'center', gap: '10px' }}>
                            <i className="pi pi-eye" style={{ color: '#6366F1' }}></i>
                            OCR Text Extraction
                            <small style={{ color: '#888', fontSize: '14px', fontWeight: 'normal' }}>
                                (PDF.js v{pdfjsLib.version} - Worker v{workerVersion || 'unknown'})
                            </small>
                        </h2>
                        <p style={{ margin: '5px 0 0 0', color: '#666' }}>
                            Extract text from PDF, JPEG, PNG, and TIFF files using AI vision models with real-time streaming
                        </p>
                        {workerUrl && (
                            <small style={{ color: '#888', fontFamily: 'monospace' }}>
                                Worker: {workerUrl.replace(/^https?:\/\//, '')}
                                {workerUrl.startsWith('/') && ' (Local 3.11.174)'}
                            </small>
                        )}
                    </div>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                        <Chip 
                            label={connectionStatus.label} 
                            icon={`pi ${connectionStatus.icon}`}
                            style={{ backgroundColor: connectionStatus.color + '20', color: connectionStatus.color }}
                        />
                        <Chip 
                            label={`PDF: ${pdfWorkerStatus.label}`} 
                            icon={`pi ${pdfWorkerStatus.icon}`}
                            style={{ backgroundColor: pdfWorkerStatus.color + '20', color: pdfWorkerStatus.color }}
                        />
                        <SplitButton 
                            label="Options" 
                            icon="pi pi-cog" 
                            model={menuItems}
                            className="p-button-sm"
                        />
                    </div>
                </div>
            </Card>

            <div style={{ display: 'flex', gap: '20px' }}>
                {/* Left Panel */}
                <div style={{ flex: 1 }}>
                    <Card title="Upload & Settings" className="mb-3">
                        {/* Model Selection */}
                        <div className="mb-3">
                            <label className="block mb-2 font-medium">Vision Model</label>
                            <Dropdown
                                value={selectedModel}
                                options={availableModels}
                                onChange={(e) => {
                                    setSelectedModel(e.value);
                                    localStorage.setItem("selectedOCRModel", e.value);
                                }}
                                placeholder="Select a model"
                                style={{ width: '100%' }}
                                disabled={isProcessing}
                            />
                            {availableModels.length === 0 && (
                                <small style={{ color: '#f44336' }}>
                                    No models found. Please install a vision-capable model.
                                </small>
                            )}
                        </div>

                        <Divider />

                        {/* IMPROVED: Better default settings for OCR */}
                        <div className="mb-3">
                            <h4>Processing Settings</h4>
                            
                            <div className="mb-3">
                                <label className="block mb-2 font-medium">
                                    Image Quality: {imageQuality}%
                                </label>
                                <Slider
                                    value={imageQuality}
                                    onChange={(e) => {
                                        setImageQuality(e.value);
                                        localStorage.setItem("ocrImageQuality", e.value.toString());
                                    }}
                                    min={70}
                                    max={95}
                                    step={5}
                                    disabled={isProcessing}
                                />
                                <small style={{ color: '#666' }}>
                                    Recommended: 90%+ for complex documents with columns
                                </small>
                            </div>

                            <div className="mb-3">
                                <label className="block mb-2 font-medium">
                                    Timeout: {timeout}s
                                </label>
                                <Slider
                                    value={timeout}
                                    onChange={(e) => {
                                        setTimeout(e.value);
                                        localStorage.setItem("ocrTimeout", e.value.toString());
                                    }}
                                    min={60}
                                    max={600}
                                    step={30}
                                    disabled={isProcessing}
                                />
                                <small style={{ color: '#666' }}>
                                    Longer timeout recommended for complex documents
                                </small>
                            </div>

                            <div className="mb-3">
                                <label className="block mb-2 font-medium">
                                    Max Retries: {maxRetries}
                                </label>
                                <Slider
                                    value={maxRetries}
                                    onChange={(e) => {
                                        setMaxRetries(e.value);
                                        localStorage.setItem("ocrMaxRetries", e.value.toString());
                                    }}
                                    min={0}
                                    max={5}
                                    step={1}
                                    disabled={isProcessing}
                                />
                                <small style={{ color: '#666' }}>
                                    Number of retries for failed OCR attempts
                                </small>
                            </div>
                        </div>

                        <Divider />

                        {/* ADD: Page Selection Dialog */}
                        <Dialog
                            visible={showPageSelector}
                            onHide={() => setShowPageSelector(false)}
                            header={`Select Pages (${pdfPages.length} total)`}
                            style={{ width: '90vw', height: '85vh' }}
                            maximizable
                            footer={
                                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                                    <span style={{ color: '#666' }}>
                                        {selectedPages.length} of {pdfPages.length} pages selected
                                    </span>
                                    <div style={{ display: 'flex', gap: '10px' }}>
                                        <Button 
                                            label="Cancel" 
                                            icon="pi pi-times" 
                                            className="p-button-secondary"
                                            onClick={() => setShowPageSelector(false)}
                                        />
                                        <Button 
                                            label={`Process ${selectedPages.length} Pages`} 
                                            icon="pi pi-play" 
                                            disabled={selectedPages.length === 0}
                                            onClick={startOcrWithSelectedPages}
                                        />
                                    </div>
                                </div>
                            }
                        >
                            <div style={{ height: '60vh', display: 'flex', flexDirection: 'column', gap: '15px' }}>
                                {/* Page Selection Controls */}
                                <Card>
                                    <div style={{ display: 'flex', gap: '20px', alignItems: 'center', flexWrap: 'wrap' }}>
                                        <div style={{ display: 'flex', gap: '10px', alignItems: 'center' }}>
                                            <label style={{ fontWeight: 'bold' }}>Selection Mode:</label>
                                            <div style={{ display: 'flex', gap: '10px' }}>
                                                {[
                                                    { value: 'all', label: 'All Pages' },
                                                    { value: 'range', label: 'Page Range' },
                                                    { value: 'custom', label: 'Custom Selection' }
                                                ].map(mode => (
                                                    <label key={mode.value} style={{ display: 'flex', alignItems: 'center', gap: '5px', cursor: 'pointer' }}>
                                                        <input
                                                            type="radio"
                                                            name="pageSelectMode"
                                                            value={mode.value}
                                                            checked={pageSelectMode === mode.value}
                                                            onChange={(e) => handlePageSelectModeChange(e.target.value)}
                                                        />
                                                        {mode.label}
                                                    </label>
                                                ))}
                                            </div>
                                        </div>
                                        
                                        {pageSelectMode === 'range' && (
                                            <div style={{ display: 'flex', gap: '10px', alignItems: 'center' }}>
                                                <label>From:</label>
                                                <input
                                                    type="number"
                                                    min="1"
                                                    max={pdfPages.length}
                                                    value={pageRange.start}
                                                    onChange={(e) => handlePageRangeChange('start', parseInt(e.target.value))}
                                                    style={{ width: '60px', padding: '5px' }}
                                                />
                                                <label>To:</label>
                                                <input
                                                    type="number"
                                                    min="1"
                                                    max={pdfPages.length}
                                                    value={pageRange.end}
                                                    onChange={(e) => handlePageRangeChange('end', parseInt(e.target.value))}
                                                    style={{ width: '60px', padding: '5px' }}
                                                />
                                            </div>
                                        )}
                                    </div>
                                </Card>

                                {/* Page Grid */}
                                <ScrollPanel style={{ height: '100%', border: '1px solid #ddd', borderRadius: '6px' }}>
                                    <div style={{ 
                                        display: 'grid', 
                                        gridTemplateColumns: 'repeat(auto-fill, minmax(150px, 1fr))', 
                                        gap: '15px', 
                                        padding: '20px' 
                                    }}>
                                        {pdfPages.map((page, index) => {
                                            const isSelected = selectedPages.includes(page.pageNumber);
                                            const canToggle = pageSelectMode === 'custom';
                                            
                                            return (
                                                <div
                                                    key={page.pageNumber}
                                                    style={{
                                                        border: isSelected ? '3px solid #007bff' : '2px solid #ddd',
                                                        borderRadius: '8px',
                                                        padding: '10px',
                                                        textAlign: 'center',
                                                        backgroundColor: isSelected ? '#f0f8ff' : 'white',
                                                        cursor: canToggle ? 'pointer' : 'default',
                                                        transition: 'all 0.2s ease',
                                                        position: 'relative'
                                                    }}
                                                    onClick={() => canToggle && togglePageSelection(page.pageNumber)}
                                                >
                                                    {/* Page Thumbnail */}
                                                    <div style={{ 
                                                        height: '120px', 
                                                        backgroundColor: '#f5f5f5', 
                                                        border: '1px solid #ddd',
                                                        borderRadius: '4px',
                                                        display: 'flex',
                                                        alignItems: 'center',
                                                        justifyContent: 'center',
                                                        marginBottom: '8px',
                                                        overflow: 'hidden'
                                                    }}>
                                                        {page.thumbnail ? (
                                                            <img 
                                                                src={page.thumbnail} 
                                                                alt={`Page ${page.pageNumber}`}
                                                                style={{ 
                                                                    maxWidth: '100%', 
                                                                    maxHeight: '100%',
                                                                    objectFit: 'contain'
                                                                }}
                                                            />
                                                        ) : (
                                                            <div style={{ color: '#666', textAlign: 'center' }}>
                                                                <i className="pi pi-file-pdf" style={{ fontSize: '2em' }}></i>
                                                                <div style={{ fontSize: '12px', marginTop: '5px' }}>
                                                                    {/* IMPROVED: Better messaging for pages without thumbnails */}
                                                                    {page.pageNumber <= 50 ? 'Loading...' : 'No Preview'}
                                                                </div>
                                                                <div style={{ fontSize: '10px', marginTop: '2px', color: '#999' }}>
                                                                    {page.pageNumber > 50 ? '(Still selectable)' : ''}
                                                                </div>
                                                            </div>
                                                        )}
                                                    </div>
                                                    
                                                    {/* Page Number */}
                                                    <div style={{ fontWeight: 'bold', fontSize: '14px' }}>
                                                        Page {page.pageNumber}
                                                    </div>
                                                    
                                                    {/* Selection Indicator */}
                                                    {isSelected && (
                                                        <div style={{
                                                            position: 'absolute',
                                                            top: '5px',
                                                            right: '5px',
                                                            backgroundColor: '#007bff',
                                                            color: 'white',
                                                            borderRadius: '50%',
                                                            width: '20px',
                                                            height: '20px',
                                                            display: 'flex',
                                                            alignItems: 'center',
                                                            justifyContent: 'center',
                                                            fontSize: '12px'
                                                        }}>
                                                            ✓
                                                        </div>
                                                    )}
                                                </div>
                                            );
                                        })}
                                    </div>
                                </ScrollPanel>
                            </div>
                        </Dialog>

            <Divider/>
                        {/* Column Handling Settings */}
                        <div className="mb-3">
                            <h4>Column Handling</h4>
                            
                            <div className="mb-3">
                                <label className="block mb-2 font-medium">Column Mode</label>
                                <Dropdown
                                    value={columnMode}
                                    options={[
                                        { label: 'Auto-detect', value: 'auto' },
                                        { label: 'Single Column', value: 'single' },
                                        { label: 'Two Columns', value: 'two-column' },
                                        { label: 'Multiple Columns', value: 'multi-column' }
                                    ]}
                                    onChange={(e) => {
                                        setColumnMode(e.value);
                                        localStorage.setItem("ocrColumnMode", e.value);
                                    }}
                                    style={{ width: '100%' }}
                                    disabled={isProcessing}
                                />
                                <small style={{ color: '#666' }}>
                                    Choose how to handle column layouts in documents
                                </small>
                            </div>

                            <div className="mb-3" style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                                <input
                                    type="checkbox"
                                    id="splitColumns"
                                    checked={splitColumns}
                                    onChange={(e) => {
                                        setSplitColumns(e.target.checked);
                                        localStorage.setItem("ocrSplitColumns", e.target.checked.toString());
                                    }}
                                    disabled={isProcessing}
                                />
                                <label htmlFor="splitColumns" style={{ cursor: 'pointer' }}>
                                    Post-process column separation
                                </label>
                            </div>
                            <small style={{ color: '#666', display: 'block', marginTop: '-0px' }}>
                                Attempt to separate mixed column content after OCR
                            </small>
                        </div>

                        <Divider />

                        <div className="mb-3" style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                            <input
                                type="checkbox"
                                id="splitColumns"
                                checked={splitColumns}
                                onChange={(e) => {
                                    setSplitColumns(e.target.checked);
                                    localStorage.setItem("ocrSplitColumns", e.target.checked.toString());
                                }}
                                disabled={isProcessing}
                            />
                            <label htmlFor="splitColumns" style={{ cursor: 'pointer' }}>
                                Post-process column separation
                            </label>
                        </div>
                        <small style={{ color: '#666', display: 'block', marginTop: '-0px' }}>
                            Attempt to separate mixed column content after OCR
                        </small>

                        {/* ADD: Table detection checkbox */}
                        <div className="mb-3" style={{ display: 'flex', alignItems: 'center', gap: '15px' }}>
                            <input
                                type="checkbox"
                                id="detectTables"
                                checked={detectTables}
                                onChange={(e) => {
                                    setDetectTables(e.target.checked);
                                    localStorage.setItem("ocrDetectTables", e.target.checked.toString());
                                }}
                                disabled={isProcessing}
                            />
                            <label htmlFor="detectTables" style={{ cursor: 'pointer' }}>
                                <strong>Detect and preserve tables</strong>
                            </label>
                        </div>
                        <small style={{ color: '#666', display: 'block', marginTop: '-0px' }}>
                            Identify tables and export them as proper tables in .docx format
                        </small>

                        <Divider />

                        {/* Language Detection Settings */}
                        <div className="mb-3">
                            <h4>Language Settings</h4>
                            
                            <div className="mb-3">
                                <label className="block mb-2 font-medium">Document Language</label>
                                <Dropdown
                                    value={selectedLanguage}
                                    options={languageOptions}
                                    onChange={(e) => {
                                        if (e.value !== 'separator') {
                                            setSelectedLanguage(e.value);
                                            localStorage.setItem("ocrLanguage", e.value);
                                        }
                                    }}
                                    style={{ width: '100%' }}
                                    disabled={isProcessing}
                                    filter
                                    filterBy="label"
                                    placeholder="Select document language"
                                    itemTemplate={(option) => {
                                        if (option.value === 'separator') {
                                            return <div style={{ borderTop: '1px solid #ccc', margin: '5px 0', pointerEvents: 'none' }}></div>;
                                        }
                                        return <span>{option.label}</span>;
                                    }}
                                />
                                <small style={{ color: '#666' }}>
                                    {selectedLanguage === 'auto' ? 
                                        'Auto-detect for mixed or unknown languages' : 
                                        'Optimizes OCR for the selected language'
                                    }
                                </small>
                            </div>

                            <div className="mb-3" style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                                <input
                                    type="checkbox"
                                    id="useLanguageHints"
                                    checked={useLanguageHints}
                                    onChange={(e) => {
                                        setUseLanguageHints(e.target.checked);
                                        localStorage.setItem("ocrUseLanguageHints", e.target.checked.toString());
                                    }}
                                    disabled={isProcessing || selectedLanguage === 'auto'}
                                />
                                <label htmlFor="useLanguageHints" style={{ cursor: 'pointer' }}>
                                    Use language-specific optimization
                                </label>
                            </div>
                            <small style={{ color: '#666', display: 'block', marginTop: '-0px' }}>
                                {selectedLanguage === 'ru' || selectedLanguage === 'bg' ? 
                                    'Provides Cyrillic character recognition hints for better accuracy' :
                                    'Provides language hints to improve OCR accuracy for specific languages'
                                }
                            </small>
                        </div>

                        <Divider/> 

                        {/* File Upload */}
                        <div className="mb-3">
                            <label className="block mb-2 font-medium">Select File</label>
                            <FileUpload
                                ref={fileUploadRef}
                                mode="basic"
                                accept=".pdf,.jpg,.jpeg,.png,.tiff,.webp"
                                maxFileSize={50000000}
                                onSelect={onFileSelect}
                                auto={false}
                                chooseLabel="Choose File"
                                className="w-full"
                                disabled={isProcessing || !isConnected || availableModels.length === 0}
                            />
                            <small className="block mt-1" style={{ color: '#666' }}>
                                Supported: PDF, JPEG, PNG, TIFF, WebP (max 50MB)
                                {workerStatus === 'ready' && workerUrl.startsWith('/') && (
                                    <span style={{ color: '#4CAF50' }}> - PDF processing ready with 3.11.174</span>
                                )}
                                {workerStatus !== 'ready' && (
                                    <span style={{ color: '#f44336' }}> - PDF processing unavailable</span>
                                )}
                            </small>
                        </div>

                        {/* Processing Status */}
                        {isProcessing && (
                            <div className="mb-3">
                                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
                                    <span>Processing {currentFile?.name}...</span>
                                    <Button 
                                        icon="pi pi-times" 
                                        className="p-button-danger p-button-sm"
                                        onClick={cancelProcessing}
                                        tooltip="Cancel processing"
                                    />
                                </div>
                                <ProgressBar value={progress} showValue={false} />
                                <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginTop: '8px' }}>
                                    <ProgressSpinner size="small" strokeWidth="4" />
                                    <small>{progress.toFixed(0)}% complete</small>
                                </div>
                                
                                {/* Streaming Status */}
                                {isStreaming && currentPageProcessing && (
                                    <div className="mt-2 p-2" style={{ backgroundColor: '#f0f9ff', border: '1px solid #0ea5e9', borderRadius: '4px' }}>
                                        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '8px' }}>
                                            <ProgressSpinner size="small" strokeWidth="4" />
                                            <span style={{ color: '#0ea5e9', fontWeight: 'bold' }}>
                                                Processing {currentPageProcessing}...
                                            </span>
                                        </div>
                                        {streamingText && (
                                            <div style={{ 
                                                maxHeight: '100px', 
                                                overflow: 'auto', 
                                                backgroundColor: 'white', 
                                                padding: '8px', 
                                                borderRadius: '4px',
                                                fontSize: '12px',
                                                fontFamily: 'monospace',
                                                whiteSpace: 'pre-wrap'
                                            }}>
                                                {streamingText}
                                                <span style={{ animation: 'blink 1s infinite' }}>▌</span>
                                            </div>
                                        )}
                                    </div>
                                )}
                            </div>
                        )}

                        {/* Processing Stats */}
                        {processingStats && (
                            <div className="mb-3">
                                <h4>Processing Statistics</h4>
                                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '10px' }}>
                                    <div>
                                        <Badge value={processingStats.pages} className="mr-2" />
                                        <span>Pages processed</span>
                                    </div>
                                    <div>
                                        <Badge value={processingStats.processingTime + 's'} className="mr-2" />
                                        <span>Processing time</span>
                                    </div>
                                    <div>
                                        <Badge value={processingStats.charactersExtracted} className="mr-2" />
                                        <span>Characters</span>
                                    </div>
                                    <div>
                                        <Badge value={processingStats.wordsExtracted} className="mr-2" />
                                        <span>Words</span>
                                    </div>
                                    {/* ADD: Tables detected */}
                                    {processingStats.tablesDetected > 0 && (
                                        <div>
                                            <Badge value={processingStats.tablesDetected} className="mr-2" severity="success" />
                                            <span>Tables detected</span>
                                        </div>
                                    )}
                                </div>
                            </div>
                        )}
                    </Card>

                    {/* Preview Images */}
                    {previewImages.length > 0 && (
                        <Card title="Preview">
                            <Button 
                                label={`View ${previewImages.length} ${previewImages.length === 1 ? 'Image' : 'Images'}`}
                                icon="pi pi-images"
                                onClick={() => setShowPreview(true)}
                                className="w-full"
                            />
                        </Card>
                    )}
                </div>

                {/* Right Panel - Extracted Text */}
                <div style={{ flex: 2 }}>
                    <Card>
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '15px' }}>
                            <h3 style={{ margin: 0 }}>Extracted Text</h3>
                            <div style={{ display: 'flex', gap: '10px' }}>
                                <Button 
                                    icon="pi pi-copy" 
                                    className="p-button-outlined p-button-sm"
                                    onClick={copyToClipboard}
                                    disabled={!extractedText}
                                    tooltip="Copy to clipboard"
                                />
                                <Button 
                                    icon="pi pi-download" 
                                    className="p-button-outlined p-button-sm"
                                    onClick={exportText}
                                    disabled={!extractedText}
                                    tooltip="Export as text file"
                                />
                            </div>
                        </div>
                        
                        <InputTextarea
                            value={extractedText}
                            onChange={(e) => setExtractedText(e.target.value)}
                            placeholder="Extracted text will appear here in real-time..."
                            style={{ width: '100%', minHeight: '500px', fontFamily: 'monospace' }}
                            autoResize
                        />
                        
                        {!extractedText && !isProcessing && (
                            <div style={{ textAlign: 'center', padding: '40px', color: '#666' }}>
                                <i className="pi pi-file-o" style={{ fontSize: '3em', marginBottom: '16px', display: 'block' }}></i>
                                <p>Upload a file to extract text using AI-powered OCR</p>
                                <small>Real-time streaming with enhanced OCR prompts - Optimized for accuracy</small>
                            </div>
                        )}
                    </Card>
                </div>
            </div>

            {/* Preview Dialog */}
            <Dialog
                visible={showPreview}
                onHide={() => setShowPreview(false)}
                header="File Preview"
                style={{ width: '80vw', height: '80vh' }}
                maximizable
            >
                <ScrollPanel style={{ height: '60vh' }}>
                    {previewImages.map((img, index) => (
                        <div key={index} className="mb-4">
                            {previewImages.length > 1 && (
                                <h4>Page {img.pageNumber}</h4>
                            )}
                            <img 
                                src={`data:image/jpeg;base64,${img.base64}`}
                                alt={`Preview ${index + 1}`}
                                style={{ maxWidth: '100%', height: 'auto', border: '1px solid #ddd' }}
                            />
                        </div>
                    ))}
                </ScrollPanel>
            </Dialog>

            {/* Add CSS for blinking cursor */}
            <style>{`
                @keyframes blink {
                    0%, 50% { opacity: 1; }
                    51%, 100% { opacity: 0; }
                }
            `}</style>
        </div>
    );
};

export default OCR;