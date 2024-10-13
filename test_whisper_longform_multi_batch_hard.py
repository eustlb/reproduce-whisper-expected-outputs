import os 
current_dir = os.path.dirname(os.path.abspath(__file__))

# ensure we import the forked whisper package
import sys
sys.path.insert(0, os.path.join(current_dir,'whisper'))

import whisper
import torch
from transformers import WhisperProcessor
import numpy as np
import regex as re 
from datasets import load_dataset, Audio

transformers_model_id = "openai/whisper-tiny.en"
openai_model_id = "tiny.en"

torch_device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float32

# ====================================================================================================
# LOAD DATA AND BUILD INPUT FEATURES AS IN THE ORIGINAL TEST 
processor = WhisperProcessor.from_pretrained(transformers_model_id)

ds = load_dataset("distil-whisper/meanwhile", "default")["test"]
ds = ds.cast_column("audio", Audio(sampling_rate=16000))

num_samples = 8

audio = ds[:num_samples]["audio"]
audios = [x["array"] for x in audio]
#====================================================================================================

openai_model = whisper.load_model(openai_model_id)

openai_gen_kwargs = {
    "condition_on_previous_text": False,
    "fp16": False,
    "without_timestamps": False,
    "logprob_threshold": -2.0, # necessary to avoid trigering temp fallback that will introduce randomness
}

texts = []
for audio in audios:
    inputs = processor(audio, return_tensors="pt", truncation=False, sampling_rate=16_000)
    inputs = inputs.to(device=torch_device)
    input_features = inputs.input_features
    openai_outputs = openai_model.transcribe(
        inputs.input_features.squeeze(), 
        **openai_gen_kwargs,
    )
    texts.append(openai_outputs["text"])

print("EXPECTED_TEXTS: ")
for text in texts:
    print(text)