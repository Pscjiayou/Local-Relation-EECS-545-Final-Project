import torch
from PIL import Image
from transformers import CLIPProcessor
import os

from modules import (
    AdaptiveFrameSampler,
    WhisperASR,
    ASRDenoiser,
    event_segmentation,
    DPPSelector,
    smooth_summary,
    MemoryAugmentedTransformer
)

def test_adaptive_frame_sampler():
    print("🔍 Testing AdaptiveFrameSampler...")
    dummy_image = Image.new("RGB", (224, 224), color="white")
    frames = [dummy_image for _ in range(40)]
    sampler = AdaptiveFrameSampler()
    sampled = sampler.sample(frames, top_k=8)
    assert len(sampled) == 8
    print("✅ AdaptiveFrameSampler passed.")

def test_whisper_asr():
    print("🔍 Testing WhisperASR... (this may take time)")
    model = WhisperASR()
    # You can use an actual .wav path here
    text, segments = model.transcribe("test_audio.wav")
    print("📝 Transcribed Text:", text[:100])
    print("✅ WhisperASR passed.")

def test_asr_denoiser():
    print("🔍 Testing ASRDenoiser...")
    denoiser = ASRDenoiser()
    result = denoiser(["thiss is an errror text"])
    print("🧼 Denoised:", result)
    print("✅ ASRDenoiser passed.")

def test_event_segmentation():
    print("🔍 Testing event_segmentation...")
    class DummySegModel:
        def predict_boundaries(self, feats):
            return [0, feats.shape[0]//2, feats.shape[0]]

    feats = torch.randn(10, 512)
    segments = event_segmentation(feats, DummySegModel())
    assert len(segments) == 2
    print("✅ event_segmentation passed.")

def test_summary_refinement():
    print("🔍 Testing DPPSelector and smoothing...")
    events = ["event1", "event2", "event3", "event4"]
    selector = DPPSelector()
    filtered = selector.select(events)
    smoothed = smooth_summary(filtered)
    print("📌 Final Summary:", smoothed)
    print("✅ Summary refinement passed.")

def test_memory_transformer():
    print("🔍 Testing MemoryAugmentedTransformer...")
    class DummyBase(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = type('', (), {})()
            self.config.hidden_size = 768
        def forward(self, x): return x

    model = MemoryAugmentedTransformer(DummyBase())
    input_tensor = torch.randn(2, 10, 768)
    out = model(input_tensor)
    assert out.shape == input_tensor.shape
    print("✅ MemoryAugmentedTransformer passed.")

if __name__ == "__main__":
    print("🧪 Starting module tests...\n")
    test_adaptive_frame_sampler()
    # test_whisper_asr()  # uncomment if test_audio.wav is available
    test_asr_denoiser()
    test_event_segmentation()
    test_summary_refinement()
    test_memory_transformer()
    print("\n✅ All module tests completed.")
