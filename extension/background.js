/**
 * Background listener for messages from content or popup scripts.
 * Handles request to get the URL of the currently active browser tab.
 */
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    // Check for the specific action: "getActiveTabURL"
    if (request.action === "getActiveTabURL") {
        // Query the currently active tab in the current window
        chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
            if (tabs.length > 0) {
                // Respond with the URL of the active tab
                sendResponse({ url: tabs[0].url });
            } else {
                // Respond with an error if no active tab is found
                sendResponse({ error: "No active tab found." });
            }
        });
        // Required to keep the message channel open for the async response above
        return true;
    }
});
