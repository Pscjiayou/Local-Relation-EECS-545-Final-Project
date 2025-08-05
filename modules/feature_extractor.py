import torch
import whisper
from transformers import CLIPProcessor, CLIPModel

print("✅ feature_extractor.py loaded successfully")

class AdaptiveFrameSampler:
    def __init__(self, model="openai/clip-vit-base-patch32"):
        self.model = CLIPModel.from_pretrained(model)
        self.processor = CLIPProcessor.from_pretrained(model)

    def sample(self, frames, top_k=32):
        inputs = self.processor(images=frames, return_tensors="pt")
        with torch.no_grad():
            embeddings = self.model.get_image_features(**inputs)
        scores = (embeddings[1:] - embeddings[:-1]).norm(p=2, dim=1)
        key_indices = scores.topk(top_k).indices.tolist()
        return [frames[i] for i in sorted(key_indices)]

class WhisperASR:
    def __init__(self):
        self.model = whisper.load_model("base")

    def transcribe(self, audio_path):
        result = self.model.transcribe(audio_path)
        return result['text'], result['segments']
