import torch
from transformers import BartForConditionalGeneration, BartTokenizer

class ASRDenoiser(torch.nn.Module):
    def __init__(self, model='facebook/bart-large'):
        super().__init__()
        self.tokenizer = BartTokenizer.from_pretrained(model)
        self.model = BartForConditionalGeneration.from_pretrained(model)

    def forward(self, text_list):
        inputs = self.tokenizer(text_list, return_tensors="pt", padding=True, truncation=True)
        outputs = self.model.generate(**inputs)
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
