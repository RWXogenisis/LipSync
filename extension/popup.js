/**
 * Triggered when the "Start Process" button is clicked.
 * Retrieves the current YouTube tab URL, sends it to the Flask backend, 
 * and polls for processing status.
 */
document.getElementById("startProcess").addEventListener("click", async () => {
    document.getElementById("status").innerText = "Fetching YouTube URL...";
    console.log("[DEBUG] Button Clicked");

    // Ask background script for the active tab's URL
    chrome.runtime.sendMessage({ action: "getActiveTabURL" }, (response) => {
        console.log("[DEBUG] Active Tab Response:", response); // Debugging

        // Validate if the URL is a YouTube video
        if (response.url && response.url.includes("youtube.com/watch")) {
            document.getElementById("status").innerText = "URL Retrieved. Starting preprocessing...";
            
            // Send the YouTube URL to the Flask server
            console.log("[DEBUG] Sending to Flask:", response.url);
            fetch("http://localhost:5001/receive_url", { 
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ youtubeURL: response.url })
            })
            .then(response => response.json())
            .then(data => {
                console.log("[DEBUG] Receive URL Response:", data);
                if (data.success) {
                    document.getElementById("status").innerText = "Processing video...";
                    checkProcessingStatus(); // Start polling the server
                } else {
                    document.getElementById("status").innerText = "Error: " + data.message;
                }
            })
            .catch(error => {
                console.error("[ERROR] Fetch Error:", error);
                document.getElementById("status").innerText = "Error sending URL. Please check the server";
            });
        } else {
            console.log("[ERROR] Invalid YouTube URL");
            document.getElementById("status").innerText = "Please open a YouTube video.";
        }
    });
});

/**
 * Polls the backend every 2 seconds to check the video processing status.
 * Updates the UI with progress and opens a new tab when complete.
 */
async function checkProcessingStatus() {
    const interval = setInterval(async () => {
        const response = await fetch("http://localhost:5001/processing_status");
        const data = await response.json();

        console.log("[DEBUG] Status:", data.status, data.processed_count, "/", data.total_segments);

        document.getElementById("status").innerText =
            `Processing: ${data.processed_count} / ${data.total_segments}`;

        // Once processing is complete, stop polling and open the player
        if (data.status === "complete") {
            clearInterval(interval);
            document.getElementById("status").innerText = "Processing complete!";
            
            // Open new tab when processing is done
            chrome.tabs.create({ url: "video_player.html" });
        }
    }, 2000); // poll every 2 seconds
}
