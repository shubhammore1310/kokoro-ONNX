import numpy as np
import torch
from onnxruntime import InferenceSession
import soundfile as sf

# Your tokens
tokens = [50, 157, 43, 135, 16, 53, 135, 46, 16, 43, 102, 16, 56, 156, 57, 135, 6, 16, 102, 62, 61, 16, 70, 56, 16, 138, 56, 156, 72, 56, 61, 85, 123, 83, 44, 83, 54, 16, 53, 65, 156, 86, 61, 62, 131, 83, 56, 4, 16, 54, 156, 43, 102, 53, 16, 156, 72, 61, 53, 102, 112, 16, 70, 56, 16, 138, 56, 44, 156, 76, 158, 123, 56, 16, 62, 131, 156, 43, 102, 54, 46, 16, 102, 48, 16, 81, 47, 102, 54, 16, 54, 156, 51, 158, 46, 16, 70, 16, 92, 156, 135, 46, 16, 54, 156, 43, 102, 48, 4, 16, 81, 47, 102, 16, 50, 156, 72, 64, 83, 56, 62, 16, 156, 51, 158, 64, 83, 56, 16, 44, 157, 102, 56, 16, 44, 156, 76, 158, 123, 56, 4]

# Take the model and voices from here.
# https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX/tree/main
# Load reference style
voice_name = torch.load("af_heart.pt", weights_only=True)
ref_s = voice_name  # Shape: (511, 1, 256)

# Convert to numpy
if isinstance(ref_s, torch.Tensor):
    ref_s = ref_s.cpu().numpy()

# ✅ Reduce (511, 1, 256) → (1, 256)
# Option 1: Mean pooling over time (most robust)
ref_s = ref_s.mean(axis=0)  # Now (1, 256)

# Ensure shape is exactly (1, 256)
assert ref_s.shape == (1, 256), f"After mean: {ref_s.shape}"

# Convert to float32
ref_s = ref_s.astype(np.float32)

# Prepare tokens
tokens = np.array([[0, *tokens, 0]], dtype=np.int64)

# Load model
model_path = 'dump/model.onnx'
sess = InferenceSession(model_path, providers=['CUDAExecutionProvider'])

# Inputs
input_feed = {
    'input_ids': tokens,
    'style': ref_s,
    'speed': np.array([1.0], dtype=np.float32)
}

# Run
audio = sess.run(None, input_feed)[0]  # (1, T)
print("Audio shape:", audio.shape)
print("Audio dtype:", audio.dtype)
# Save
audio = np.clip(audio, -1.0, 1.0)
audio_int16 = (audio * 32767).astype(np.int16)
sf.write("output.wav", audio_int16, samplerate=24000)
print("✅ Audio saved to output.wav")
