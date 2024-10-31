import os 
current_dir = os.path.dirname(os.path.abspath(__file__))

# ensure we import the forked whisper package
import sys
sys.path.insert(0, os.path.join(current_dir,'whisper'))

import whisper
import torch
from transformers import WhisperProcessor
import numpy as np
from datasets import load_dataset

transformers_model_id = "openai/whisper-small.en"
openai_model_id = "small.en"

torch_device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float32

# ====================================================================================================
# LOAD DATA AND BUILD INPUT FEATURES AS IN THE ORIGINAL TEST 
processor = WhisperProcessor.from_pretrained(transformers_model_id)

dataset = load_dataset("distil-whisper/librispeech_long", "clean", split="validation")
sample = dataset[0]["audio"]["array"]
sampling_rate = dataset[0]["audio"]["sampling_rate"]

sample = [*sample[:15 * sampling_rate], *np.zeros(16 * sampling_rate).tolist(), *sample[15 * sampling_rate:]]
sample = np.array(sample)

inputs = processor(
    sample,
    sampling_rate=16_000,
    padding="longest",
    truncation=False,
    return_attention_mask=True,
    return_tensors="pt",
)
#====================================================================================================

openai_model = whisper.load_model(openai_model_id)

openai_gen_kwargs = {
    "condition_on_previous_text": False,
    "fp16": False,
    "without_timestamps": False,
}

output = openai_model.transcribe(
    inputs.input_features.squeeze(),
    **openai_gen_kwargs,
)

for seg in output["segments"]:
    print(f"{seg['start']} -> {seg['end']}: {seg['text']}")
