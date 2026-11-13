import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import tensorflow as tf
import json
import av
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

st.set_page_config(page_title="SSLRS Real-Time", layout="centered")
st.title("🤟 SSLRS: Real-Time Sign Language Reader")
st.write("Show your SIBI or BISINDO hand gesture to the camera. The prediction updates in real-time.")

@st.cache_resource
def load_models():
    model = tf.keras.models.load_model('sibi_bisindo_dual_head_best.keras', compile=False)
    with open('sign_meta.json', 'r') as f:
        meta = json.load(f)
    idx_to_sign = {int(k): v for k, v in meta['idx_to_sign'].items()}
    
    base_options = mp_python.BaseOptions(model_asset_path='hand_landmarker.task')
    options = vision.HandLandmarkerOptions(
        base_options=base_options, num_hands=2, 
        min_hand_detection_confidence=0.5,
        running_mode=vision.RunningMode.IMAGE
    )
    landmarker = vision.HandLandmarker.create_from_options(options)
    return model, idx_to_sign, landmarker

model, idx_to_sign, landmarker = load_models()

def clean_label(raw_label):
    label = raw_label
    if raw_label.startswith("SIBI_"):
        label = raw_label.replace("SIBI_", "SIBI: ")
    elif raw_label.startswith("BISINDO_") and ".0" in raw_label:
        try:
            num = int(float(raw_label.replace("BISINDO_", "")))
            if 0 <= num <= 25:
                label = f"BISINDO Letter {chr(65 + num)}"
        except:
            pass
    elif raw_label.startswith("BISINDO_") and raw_label[8:].isdigit():
        label = f"BISINDO Number {raw_label[8:]}"
    elif raw_label.startswith("BISINDO_"):
        text = raw_label.replace("BISINDO_", "")
        if len(text) == 1:
            label = f"BISINDO Letter {text}"
        else:
            label = f"Word: {text}"
    return label

class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.current_text = ""

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        r = landmarker.detect(mp_img)
        
        prediction_text = "Detecting..."
        
        if r.hand_landmarks:
            raw_pts = []
            for hl in r.hand_landmarks:
                pts = np.array([[lm.x, lm.y, lm.z] for lm in hl], dtype=np.float32)
                raw_pts.append(pts)
                
            hands = []
            global_anchor = raw_pts[0][0].copy()  
            for pts in raw_pts:
                normalized_pts = pts - global_anchor
                scale = np.linalg.norm(normalized_pts[9])
                if scale > 1e-6:
                    normalized_pts = normalized_pts / scale
                hands.append(normalized_pts.flatten().astype(np.float32))

            while len(hands) < 2:
                hands.append(np.zeros(63, dtype=np.float32))
                
            feat = np.concatenate(hands[:2]).astype(np.float32)
            seq = np.tile(feat, (30, 1))[np.newaxis, ...].astype(np.float32)
            
            lang_p, sign_p = model(seq, training=False)
            sign_idx = int(tf.argmax(sign_p[0]).numpy())
            conf = float(sign_p[0][sign_idx])
            
            if conf > 0.60:
                label = clean_label(idx_to_sign[sign_idx])
                prediction_text = f"{label} ({conf*100:.1f}%)"
                
                if label not in self.current_text[-20:] and conf > 0.80:
                    self.current_text += label + " "
        
        cv2.putText(img, f"Prediction: {prediction_text}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(img, f"Text: {self.current_text}", (20, 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
        return av.VideoFrame.from_ndarray(img, format="bgr24")

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

webrtc_streamer(
    key="sign-language",
    mode=webrtc_streamer.MediaStreamMode.SENDRECV,
    video_processor_factory=VideoProcessor,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)