/**
 * webcam.js – Browser-based face recognition via getUserMedia
 * Sends webcam frames to /recognize endpoint every 2 seconds.
 */

const video = document.getElementById('webcam');
const status = document.getElementById('recog-status');
const nameDisplay = document.getElementById('recog-name');
const canvas = document.createElement('canvas');

let streaming = false;
let recognizing = false;
let lastResult = null;
let intervalId = null;

// ─── Start Camera ─────────────────────────────────────────────────────────

function startCamera() {
    // Check for Secure Context (HTTPS or localhost)
    if (!window.isSecureContext && location.hostname !== 'localhost' && location.hostname !== '127.0.0.1') {
        setStatus('error', '❌ Camera requires HTTPS or localhost access. (Secure Context Error)');
        showToast('❌ Browser blocks camera on non-HTTPS IP addresses.', 'error');
        return;
    }

    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        setStatus('error', '❌ Camera API not supported or blocked by browser settings.');
        return;
    }
    setStatus('loading', '📷 Starting camera...');
    navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } })
        .then(stream => {
            video.srcObject = stream;
            video.play();
            streaming = true;
            setStatus('ready', '✅ Camera ready. Position your face in the frame.');
            // Auto-recognize every 2 seconds
            intervalId = setInterval(sendFrame, 2000);
        })
        .catch(err => {
            setStatus('error', `❌ Camera error: ${err.message}`);
        });
}

function stopCamera() {
    if (video.srcObject) {
        video.srcObject.getTracks().forEach(t => t.stop());
        streaming = false;
    }
    if (intervalId) clearInterval(intervalId);
    setStatus('idle', '');
}

// ─── Capture & Send ──────────────────────────────────────────────────────────

function sendFrame() {
    if (!streaming || recognizing) return;
    recognizing = true;

    canvas.width = video.videoWidth || 640;
    canvas.height = video.videoHeight || 480;
    canvas.getContext('2d').drawImage(video, 0, 0);
    const imageData = canvas.toDataURL('image/jpeg', 0.7);

    setStatus('loading', '🔍 Recognizing...');

    fetch('/recognize_ajax', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image: imageData })
    })
    .then(r => r.json())
    .then(data => {
        recognizing = false;
        if (data.success) {
            if (data.name && data.name !== 'Unknown') {
                if (data.already_marked) {
                    setStatus('already', `🕐 ${data.name} — Already marked today!`);
                } else {
                    setStatus('success', `✅ Welcome, ${data.name}! Attendance marked.`);
                    showToast(`✅ ${data.name} marked Present!`, 'success');
                    stopCamera();
                }
            } else {
                setStatus('unknown', '❓ Face not recognized. Try adjusting lighting.');
            }
        } else {
            setStatus('error', data.message || 'Recognition error.');
        }
    })
    .catch(err => {
        recognizing = false;
        setStatus('error', `Error: ${err.message}`);
    });
}

// ─── UI helpers ───────────────────────────────────────────────────────────────

function setStatus(type, message) {
    if (!status) return;
    const colors = {
        idle: '',
        loading: '#d29922',
        ready: '#3fb950',
        success: '#3fb950',
        already: '#7c6ff7',
        unknown: '#d29922',
        error: '#f85149'
    };
    status.style.color = colors[type] || '#e6edf3';
    status.textContent = message;
}

function showToast(msg, type = 'info') {
    const toast = document.createElement('div');
    toast.style.cssText = `
        position:fixed; bottom:24px; right:24px; z-index:9999;
        padding:14px 22px; border-radius:10px; font-weight:600;
        background:${type === 'success' ? '#3fb950' : type === 'error' ? '#f85149' : '#7c6ff7'};
        color:#fff; box-shadow:0 8px 24px rgba(0,0,0,0.3);
        animation: slideIn 0.3s ease;
    `;
    toast.textContent = msg;
    document.body.appendChild(toast);
    setTimeout(() => toast.remove(), 4000);
}

// ─── Button bindings ──────────────────────────────────────────────────────────

document.addEventListener('DOMContentLoaded', () => {
    const startBtn = document.getElementById('start-camera');
    const stopBtn = document.getElementById('stop-camera');
    if (startBtn) startBtn.addEventListener('click', startCamera);
    if (stopBtn) stopBtn.addEventListener('click', stopCamera);
});
