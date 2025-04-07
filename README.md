# LipCap - Real-Time Lip Reading Subtitle Generator

LipCap is a browser extension that generates **real-time subtitles** for video content by analyzing **lip movements** using **deep learning models**—not audio. Designed to enhance accessibility on platforms like **Udemy**, LipCap is especially useful for:

- Hearing-impaired individuals  
- Non-native language learners  
- Noisy or low-quality audio environments

---

## Table of Contents

- [About the Project](#-about-the-project)
- [Problem Statement](#-problem-statement)
- [Scope](#-scope)
- [Getting Started](#-getting-started)
- [Training](#-training)
- [Deployment (Browser Extension)](#-deployment-browser-extension)
- [License](#-license)
- [Acknowledgements](#-acknowledgements)

---

## About the Project

LipCap uses **computer vision** to generate **accurate, grammatically coherent subtitles** by analyzing lip movements in video frames. It employs **Spatiotemporal Convolutional Neural Networks (ST-CNNs)** and **Bidirectional LSTMs (Bi-LSTMs)** for feature extraction and sequence modeling.

Unlike audio-based transcription tools, this system operates effectively even in poor audio conditions, offering a more inclusive and robust solution for subtitle generation.

---

## Problem Statement

Traditional subtitle generators depend on clear audio, which isn't always available. LipCap solves this by using **only visual input**—lip movements—to generate subtitles. This approach is particularly helpful for:

- Noisy or silent environments
- Hearing-impaired users
- Platforms that lack reliable subtitles

---

## Scope

- Focused on **e-learning platforms** like Udemy
- Works with short video clips (e.g., GRID corpus)
- Real-time video-to-text transcription
- Designed as a **JavaScript-based browser extension**
- Utilizes **pre-trained deep learning models** using PyTorch framework
- Supports **real-time sentence-level subtitle generation**

---

## Getting Started

### Prerequisites

- Python 3.9 or 3.10
- CUDA (if using GPU) – [Install based on your system](https://pytorch.org/get-started/locally/)

### 📁 Setup Instructions

1. Clone the repository:
   ```bash
   git clone https://github.com/RWXogenisis/LipCap.git
   cd LipCap
   ```

2. *(Optional but recommended)* Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Update the `--extra-index-url` in `requirements.txt` according to your CUDA version from [PyTorch local install guide](https://pytorch.org/get-started/locally/).

4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🧪 Training

To train the model:

```bash
python train.py
```

*Note: Make sure your dataset is set up as required in the repo.*

---

## Deployment (Browser Extension)

### Step-by-step Instructions

1. Open a Chromium-based browser (Chrome, Edge, etc.)
2. Go to the Extensions page:
   - `chrome://extensions` or `edge://extensions`
3. Enable **Developer Mode**
4. Click **"Load Unpacked"**
5. Select the `extension/` folder from the cloned repo

### Running the Extension

1. Open a test video (e.g., a custom stitched video from GRID corpus)
2. In terminal, run:
   ```bash
   python server.py
   ```
3. Click the browser extension and then click **“Start Process”**
4. A new tab will open with the video and live subtitle predictions

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Acknowledgements

- GRID Corpus for training data
- PyTorch and OpenCV for model training and preprocessing
- Udemy and other e-learning platforms for inspiration

---
