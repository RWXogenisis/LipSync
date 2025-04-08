const express = require('express');
const multer = require('multer');
const dgram = require('dgram');
const app = express();
const upload = multer();
const udpClient = dgram.createSocket('udp4');

const UDP_PORT = 5005;
const UDP_HOST = '127.0.0.1';

app.post('/process_frame', upload.single('frame'), (req, res) => {
    if (!req.file) return res.status(400).send('No frame uploaded');

    udpClient.send(req.file.buffer, UDP_PORT, UDP_HOST, err => {
        if (err) {
            console.error("UDP send error:", err);
            res.status(500).send("Failed to send UDP packet");
        } else {
            res.json({ status: "Frame sent" });
        }
    });
});

app.listen(5000, () => {
    console.log("HTTP to UDP Proxy running on http://localhost:5000");
});
