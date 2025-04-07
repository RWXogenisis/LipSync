// Wait for the DOM to be fully loaded
document.addEventListener("DOMContentLoaded", async function () {
    let ytplayer = document.getElementById("ytplayer");
    let lastTimestampSent = -1; // Ensure 0 gets sent
    let isTrackingStarted = false; // Prevent multiple intervals
    let segmentDisplay = document.getElementById("segmentData");

    /**
     * Fetches a YouTube URL from the Flask server and loads the video into the iframe.
     * Displays appropriate status messages based on success or failure.
     */
    async function fetchYouTubeURL() {
        try {
            document.getElementById("status").innerText = "Fetching video URL from server...";
            let response = await fetch("http://localhost:5001/get_url");
            let data = await response.json();

            console.log("[DEBUG] Server Response:", data);

            if (data.youtubeURL) {
                // Extract video ID from the URL
                let videoId = new URL(data.youtubeURL).searchParams.get("v");
                console.log("[DEBUG] Extracted Video ID:", videoId);

                if (videoId) {
                    ytplayer.src = `https://www.youtube.com/embed/${videoId}?enablejsapi=1`;
                    document.getElementById("status").innerText = "Video Loaded";
                } else {
                    document.getElementById("status").innerText = "Invalid video ID!";
                }
            } else {
                document.getElementById("status").innerText = "No YouTube URL received.";
            }
        } catch (error) {
            console.error("[ERROR] Fetch failed:", error);
            document.getElementById("status").innerText = "Error fetching video.";
        }
    }

    /**
     * Sends the current timestamp (in seconds) to the server.
     * Displays segment information or error messages based on the response.
     * 
     * @param {number} timestamp - The timestamp to send to the server.
     */
    async function sendTimestampToServer(timestamp) {
        console.log("Calling server")
        lastTimestampSent = timestamp;
        try {
            let response = await fetch("http://localhost:5001/update_timestamp", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ timestamp: timestamp })
            });
    
            if (!response.ok) {
                throw new Error("Failed to fetch data from server. Status: " + response.status);
            }
    
            let data = await response.json();
            console.log(`[DEBUG] Timestamp Sent: ${timestamp}s`);
            console.log("[DEBUG] Segment Data Received:", data.segment_data);
    
            if (data.segment_data) {
                segmentDisplay.innerText = `Segment Data: ${JSON.stringify(data.segment_data)}`;
            } else {
                segmentDisplay.innerText = `Error: No segment data received.`;
            }
        } catch (error) {
            console.error("[ERROR] Failed to send timestamp:", error);
            segmentDisplay.innerText = `Error: ${error.message}`;
        }
    }
    
    /**
     * Returns a throttled version of a function that can only be called every `limit` ms.
     * 
     * @param {Function} func - Function to throttle.
     * @param {number} limit - Time limit in milliseconds.
     * @returns {Function} Throttled function.
     */
    function throttle(func, limit) {
        let inThrottle;
        return function () {
            const args = arguments;
            const context = this;
            if (!inThrottle) {
                func.apply(context, args);
                inThrottle = true;
                setTimeout(() => (inThrottle = false), limit);
            }
        };
    }

    const throttledSendTimestamp = throttle(sendTimestampToServer, 2500);

    /**
     * Listens to messages from the YouTube iframe and extracts current time info.
     * Triggers a throttled timestamp update to the server if necessary.
     */
    window.addEventListener("message", function (event) {
        if (event.origin !== "https://www.youtube.com") return;

        let data;
        try {
            data = JSON.parse(event.data);
        } catch (e) {
            return;
        }

        if (data.info && typeof data.info.currentTime !== "undefined") {
            let currentTime = Math.floor(data.info.currentTime);
            document.getElementById("timestamp").innerText = `Current Timestamp: ${currentTime}s`;

            if (currentTime !== lastTimestampSent) {
                console.log("Condition Satisfied, calling throttle");
                throttledSendTimestamp(currentTime);
            }
        }
    });

    /**
     * Starts sending "listening" postMessage events to the iframe every second.
     * This is required for the YouTube player to start posting currentTime messages.
     */
    function startTracking() {
        if (!isTrackingStarted) {
            isTrackingStarted = true;
            setInterval(() => {
                if (ytplayer && ytplayer.contentWindow) {
                    // Triggers YouTube API to start sending currentTime info
                    ytplayer.contentWindow.postMessage('{"event":"listening","id":1}', '*');
                }
            }, 1000);
        }
    }

    // Initialize the player and tracking
    fetchYouTubeURL();
    startTracking();
});
