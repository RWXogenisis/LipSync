const video = document.createElement('video');
video.setAttribute('autoplay', '');
video.setAttribute('playsinline', '');
document.body.appendChild(video);

async function setupCamera() {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    video.srcObject = stream;
    return new Promise((resolve) => {
        video.onloadedmetadata = () => resolve(video);
    });
}

async function loadModels() {
    await faceapi.nets.tinyFaceDetector.loadFromUri('/models');
    await faceapi.nets.faceLandmark68Net.loadFromUri('/models');
}

async function extractLipRegion() {
    const canvas = document.createElement('canvas');
    document.body.appendChild(canvas);
    const ctx = canvas.getContext('2d');
    
    const displaySize = { width: video.videoWidth, height: video.videoHeight };
    faceapi.matchDimensions(canvas, displaySize);
    
    setInterval(async () => {
        const detections = await faceapi.detectSingleFace(video, new faceapi.TinyFaceDetectorOptions()).withFaceLandmarks();
        
        if (detections) {
            const landmarks = detections.landmarks;
            const lipPoints = landmarks.getMouth();
            
            const minX = Math.min(...lipPoints.map(p => p.x));
            const minY = Math.min(...lipPoints.map(p => p.y));
            const maxX = Math.max(...lipPoints.map(p => p.x));
            const maxY = Math.max(...lipPoints.map(p => p.y));
            
            const width = maxX - minX;
            const height = maxY - minY;
            
            canvas.width = width;
            canvas.height = height;
            ctx.drawImage(video, minX, minY, width, height, 0, 0, width, height);
        }
    }, 100);
}

(async () => {
    await loadModels();
    await setupCamera();
    extractLipRegion();
})();