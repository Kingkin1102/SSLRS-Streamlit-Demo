# SSLRS: Simultaneous Sign-Languages Reader System

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-WebRTC-red.svg)

This repository contains the source code for the **Live Demo** of the SSLRS (Simultaneous Sign-Languages Reader System) research project. This application is designed to translate Indonesian Sign Language (SIBI and BISINDO) in real-time through a webcam.

## 🚀 Key Features
- **Real-time Detection:** Utilizes `streamlit-webrtc` to process video directly from the browser with low latency.
- **MediaPipe Landmark Extraction:** Efficiently extracts 21 hand joint coordinates (x, y, z).
- **Dual-Head CNN-LSTM:** A Deep Learning model trained to recognize 107 classes (Letters, Numbers, and Words) for both SIBI and BISINDO simultaneously.
- **Robust Normalization:** Implements *Global Anchor Normalization* to make the model resilient to hand position shifts.

## 🛠️ How to Run Locally
If you want to run this application on your own machine:

1. Clone this repository:
   ```bash
   git clone https://github.com/Kingkin1102/SSLRS-Streamlit-Demo.git
2. Install the required libraries:
   pip install -r requirements.txt
3. Run the Streamlit application:
   streamlit run app.py
