let captureInterval = 38; // 1000ms / 26.33 = 37.98ms (for 25 FPS)
let subtitleOverlay;

function createSubtitleOverlay() {
    subtitleOverlay = document.createElement('div');
    subtitleOverlay.id = 'subtitleOverlay';
    subtitleOverlay.style.position = 'fixed';
    subtitleOverlay.style.bottom = '50px';
    subtitleOverlay.style.left = '50%';
    subtitleOverlay.style.transform = 'translateX(-50%)';
    subtitleOverlay.style.backgroundColor = 'rgba(0, 0, 0, 0.7)';
    subtitleOverlay.style.color = 'white';
    subtitleOverlay.style.padding = '10px';
    subtitleOverlay.style.borderRadius = '5px';
    subtitleOverlay.style.fontSize = '20px';
    subtitleOverlay.style.fontFamily = 'Arial, sans-serif';
    subtitleOverlay.style.zIndex = '9999';
    document.body.appendChild(subtitleOverlay);
}

function updateSubtitle(text) {
    if (!subtitleOverlay) {
        createSubtitleOverlay();
    }
    subtitleOverlay.innerText = text;
}

function captureFrame() {
    const video = document.querySelector('video'); // Select the playing video
    if (!video) return;

    const canvas = document.createElement('canvas');
    canvas.width = 320; // Reduce resolution for speed
    canvas.height = 180;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    canvas.toBlob(blob => {
        sendFrameToServer(blob);
    }, 'image/jpeg', 0.7); // 70% quality
}

function sendFrameToServer(frameBlob) {
    const formData = new FormData();
    formData.append('frame', frameBlob);

    fetch('http://localhost:5000/process_frame', {
        method: 'POST',
        body: formData,
        mode: 'cors'
    })
    .then(response => response.json())
    .then(data => {
        if (data.text) {
            updateSubtitle(data.text);
        } else {
            console.warn('No subtitle received:', data);
        }
    })
    .catch(error => {
        console.error('Error sending frame:', error);
    });
}

// Start capturing frames at 25 FPS
setInterval(captureFrame, captureInterval);


//.mpg - single video - feed this to the model
//sleep(1.5) 