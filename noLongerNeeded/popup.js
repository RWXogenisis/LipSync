// popup.js - Controls the subtitle toggle in the browser extension

document.getElementById("toggleSubtitles").addEventListener("click", async () => {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });

    chrome.scripting.executeScript({
        target: { tabId: tab.id },
        function: toggleSubtitles
    });
});

document.getElementById("toggleSubtitles").addEventListener("click", async () => {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });

    chrome.scripting.executeScript({
        target: { tabId: tab.id },
        files: ["content.js"]
    }, () => {
        if (chrome.runtime.lastError) {
            console.error("Error injecting content.js:", chrome.runtime.lastError);
        } else {
            console.log("content.js injected successfully");
            chrome.scripting.executeScript({
                target: { tabId: tab.id },
                function: toggleSubtitles
            });
        }
    });
});

function toggleSubtitles() {
    if (window.subtitleInterval) {
        clearInterval(window.subtitleInterval);
        window.subtitleInterval = null;
        console.log("Subtitles disabled");
    } else {
        window.subtitleInterval = setInterval(captureFrame, 40); // 25 FPS
        console.log("Subtitles enabled");
    }
}