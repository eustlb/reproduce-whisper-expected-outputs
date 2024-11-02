import os 
current_dir = os.path.dirname(os.path.abspath(__file__))

# ensure we import the forked whisper package
import sys
sys.path.insert(0, os.path.join(current_dir,'whisper'))

import whisper
import torch
from transformers import WhisperProcessor, set_seed
import numpy as np
from datasets import load_dataset, Audio

transformers_model_id = "openai/whisper-tiny"
openai_model_id = "tiny"

torch_device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float32

# ====================================================================================================
# LOAD DATA AND BUILD INPUT FEATURES AS IN THE ORIGINAL TEST 
set_seed(0)
processor = WhisperProcessor.from_pretrained(transformers_model_id)

dataset = load_dataset("distil-whisper/librispeech_long", "clean", split="validation")
sample = dataset[0]["audio"]

input_features = processor(
    sample["array"], return_tensors="pt", truncation=False, sampling_rate=sample["sampling_rate"]
).input_features
input_features = input_features.to(torch_device)
#====================================================================================================

openai_model = whisper.load_model(openai_model_id)

openai_gen_kwargs = {
    "condition_on_previous_text": False,
    "fp16": False,
    "without_timestamps": False,
}

result = openai_model.transcribe(
    input_features.squeeze(), 
    **openai_gen_kwargs,
)

for seg in result["segments"]:
    print("text: ", seg["text"])
    print("timestamps: ", (seg["start"], seg["end"]))