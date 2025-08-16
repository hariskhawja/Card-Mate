/**
 * Card-Mate Mobile Camera Detection JavaScript
 * Handles camera access, WebSocket communication, and real-time card detection
 */

// DOM elements
const status = document.getElementById('status');
const connectionStatus = document.getElementById('connectionStatus');
const detectionCount = document.getElementById('detectionCount');
const cardsDetected = document.getElementById('cardsDetected');
const lastUpdate = document.getElementById('lastUpdate');
const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const startCameraBtn = document.getElementById('startCamera');
const startDetectionBtn = document.getElementById('startDetection');
const stopDetectionBtn = document.getElementById('stopDetection');

// State variables
let mediaStream = null;
let detectionSocket = null;
let isDetecting = false;
let isConnected = false;
let detectionInterval = null;

/**
 * Show status message with appropriate styling
 * @param {string} message - The message to display
 * @param {string} type - Message type (info, success, error, warning)
 */
function showStatus(message, type = 'info') {
    status.textContent = message;
    status.className = `status status-${type}`;
    
    // Clear status after 5 seconds for non-error messages
    if (type !== 'error') {
        setTimeout(() => {
            status.textContent = '';
            status.className = 'status';
        }, 5000);
    }
}

/**
 * Update connection status display
 * @param {string} statusText - Status text to display
 * @param {boolean} isConnectedState - Whether connected or not
 */
function updateConnectionStatus(statusText, isConnectedState) {
    connectionStatus.textContent = statusText;
    isConnected = isConnectedState;
    
    if (isConnected) {
        connectionStatus.className = 'connection-status connected';
    } else if (statusText === 'Connecting...') {
        connectionStatus.className = 'connection-status connecting';
    } else {
        connectionStatus.className = 'connection-status disconnected';
    }
}

/**
 * Start mobile camera with rear camera preference
 */
async function startCamera() {
    try {
        showStatus('Requesting camera access...', 'info');
        
        // Request access to rear camera with high quality
        const constraints = {
            video: {
                facingMode: { exact: 'environment' }, // Rear camera
                width: { ideal: 1280 },
                height: { ideal: 720 },
                frameRate: { ideal: 30 }
            },
            audio: false
        };
        
        mediaStream = await navigator.mediaDevices.getUserMedia(constraints);
        video.srcObject = mediaStream;
        video.style.display = 'block';
        
        // Set up canvas for frame capture
        canvas.width = 640;
        canvas.height = 480;
        
        startCameraBtn.disabled = true;
        startDetectionBtn.disabled = false;
        
        showStatus('Camera started successfully!', 'success');
        
    } catch (error) {
        console.error('Error accessing camera:', error);
        
        // Try fallback to any available camera
        try {
            const fallbackConstraints = {
                video: {
                    width: { ideal: 1280 },
                    height: { ideal: 720 }
                },
                audio: false
            };
            
            mediaStream = await navigator.mediaDevices.getUserMedia(fallbackConstraints);
            video.srcObject = mediaStream;
            video.style.display = 'block';
            
            canvas.width = 640;
            canvas.height = 480;
            
            startCameraBtn.disabled = true;
            startDetectionBtn.disabled = false;
            
            showStatus('Camera started (fallback mode)', 'warning');
            
        } catch (fallbackError) {
            console.error('Fallback camera access failed:', fallbackError);
            showStatus('Failed to access camera. Please check permissions.', 'error');
        }
    }
}

/**
 * Connect to WebSocket for real-time detection
 */
function connectToDetectionStream() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws/camera`;
    
    updateConnectionStatus('Connecting...', false);
    
    try {
        detectionSocket = new WebSocket(wsUrl);
        
        detectionSocket.onopen = function(event) {
            console.log('Connected to mobile detection stream');
            updateConnectionStatus('Connected', true);
            showStatus('Connected to detection system', 'success');
        };
        
        detectionSocket.onmessage = function(event) {
            try {
                const data = JSON.parse(event.data);
                updateDetectionResults(data);
            } catch (error) {
                console.error('Error parsing detection data:', error);
            }
        };
        
        detectionSocket.onerror = function(error) {
            console.error('WebSocket error:', error);
            updateConnectionStatus('Connection Error', false);
            showStatus('Connection error with detection system', 'error');
        };
        
        detectionSocket.onclose = function(event) {
            console.log('Detection stream disconnected');
            updateConnectionStatus('Disconnected', false);
            showStatus('Disconnected from detection system', 'warning');
            
            // Stop detection if connection lost
            if (isDetecting) {
                stopDetection();
            }
        };
        
    } catch (error) {
        console.error('Error creating WebSocket connection:', error);
        updateConnectionStatus('Failed to Connect', false);
        showStatus('Failed to connect to detection system', 'error');
    }
}

/**
 * Capture frame and send for detection (LOW FREQUENCY APPROACH)
 */
function captureAndSendFrame() {
    if (!mediaStream || !isDetecting || !isConnected) {
        return;
    }
    
    try {
        // Draw current video frame to canvas
        const ctx = canvas.getContext('2d');
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        
        // Convert to base64 image with standard quality
        const imageData = canvas.toDataURL('image/jpeg', 0.8);
        const base64Data = imageData.split(',')[1];
        
        // Send frame data via WebSocket
        if (detectionSocket && detectionSocket.readyState === WebSocket.OPEN) {
            detectionSocket.send(JSON.stringify({
                type: 'frame',
                data: base64Data,
                timestamp: Date.now()
            }));
        }
        
    } catch (error) {
        console.error('Error capturing frame:', error);
    }
}

/**
 * Start detection process
 */
function startDetection() {
    if (!mediaStream) {
        showStatus('Please start camera first', 'error');
        return;
    }
    
    showStatus('Starting real-time detection...', 'info');
    
    // Connect to detection WebSocket
    connectToDetectionStream();
    
    // Start sending frames for detection with LOWER FREQUENCY
    isDetecting = true;
    detectionInterval = setInterval(captureAndSendFrame, 500); // 2 FPS - Lower frequency
    
    startDetectionBtn.disabled = true;
    stopDetectionBtn.disabled = false;
    
    showStatus('Real-time detection started!', 'success');
}

/**
 * Stop detection process
 */
function stopDetection() {
    isDetecting = false;
    
    if (detectionInterval) {
        clearInterval(detectionInterval);
        detectionInterval = null;
    }
    
    if (detectionSocket) {
        detectionSocket.close();
        detectionSocket = null;
    }
    
    updateConnectionStatus('Disconnected', false);
    startDetectionBtn.disabled = false;
    stopDetectionBtn.disabled = true;
    
    showStatus('Detection stopped', 'info');
}

/**
 * Update detection results display
 * @param {Object} data - Detection results data
 */
function updateDetectionResults(data) {
    // Update detection count
    if (data.count !== undefined) {
        detectionCount.textContent = data.count;
    }
    
    // Update cards detected display
    if (data.cards && Array.isArray(data.cards) && data.cards.length > 0) {
        const cardElements = data.cards.map(card => {
            let cardText = '';
            let cardClass = 'card-item';
            
            if (Array.isArray(card) && card.length >= 2) {
                const [num, suit] = card;
                const numText = getCardNumber(num);
                cardText = `${numText} of ${suit}`;
                
                // Add suit-specific styling
                if (suit === 'Hearts') {
                    cardClass += ' card-hearts';
                } else if (suit === 'Diamonds') {
                    cardClass += ' card-diamonds';
                } else if (suit === 'Spades') {
                    cardClass += ' card-spades';
                } else if (suit === 'Clubs') {
                    cardClass += ' card-clubs';
                }
            } else if (typeof card === 'string') {
                cardText = card;
            } else {
                cardText = 'Unknown card';
            }
            
            return `<span class="${cardClass}">${cardText}</span>`;
        }).join(' ');
        
        cardsDetected.innerHTML = cardElements;
    } else if (data.count === 0) {
        cardsDetected.innerHTML = '<span style="color: var(--text-secondary); font-style: italic;">No cards detected in current frame</span>';
    }
    
    // Update last update time
    lastUpdate.textContent = new Date().toLocaleTimeString();
    
    // Show real-time status for successful detections
    if (data.count > 0) {
        showStatus(`Mobile detection: ${data.count} card(s) found`, 'success');
    }
}

/**
 * Convert card number to readable text
 * @param {number} num - Card number
 * @returns {string} Readable card number
 */
function getCardNumber(num) {
    const numberMap = {
        1: "Ace", 2: "Two", 3: "Three", 4: "Four", 5: "Five",
        6: "Six", 7: "Seven", 8: "Eight", 9: "Nine", 10: "Ten",
        11: "Jack", 12: "Queen", 13: "King"
    };
    return numberMap[num] || num.toString();
}

// Event listeners
startCameraBtn.addEventListener('click', startCamera);
startDetectionBtn.addEventListener('click', startDetection);
stopDetectionBtn.addEventListener('click', stopDetection);

// Clean up when page unloads
window.addEventListener('beforeunload', () => {
    stopDetection();
    if (mediaStream) {
        mediaStream.getTracks().forEach(track => track.stop());
    }
});

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    showStatus('Ready to start mobile camera detection', 'info');
});
