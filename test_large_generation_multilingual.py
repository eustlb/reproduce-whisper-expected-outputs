import os 
current_dir = os.path.dirname(os.path.abspath(__file__))

# ensure we import the forked whisper package
import sys
sys.path.insert(0, os.path.join(current_dir,'whisper'))

import whisper
import torch
import datasets
from transformers import WhisperProcessor, set_seed
import numpy as np
from datasets import load_dataset, Audio

transformers_model_id = "openai/whisper-large-v3"
openai_model_id = "large-v3"

torch_device = "cuda:0" if torch.cuda.is_available() else "cpu"
set_seed(0)
torch_dtype = torch.float32

# ====================================================================================================
# LOAD DATA AND BUILD INPUT FEATURES AS IN THE ORIGINAL TEST 
ds = load_dataset(
    "facebook/multilingual_librispeech", "german", split="test", streaming=True, trust_remote_code=True
)
ds = ds.cast_column("audio", datasets.Audio(sampling_rate=16_000))

processor = WhisperProcessor.from_pretrained(transformers_model_id)
input_speech = next(iter(ds))["audio"]["array"]
input_features = processor(input_speech, return_tensors="pt", sampling_rate=16_000).input_features
input_features = input_features.to(torch_device)
#====================================================================================================

openai_model = whisper.load_model(openai_model_id)

openai_gen_kwargs = {
    "condition_on_previous_text": False,
    "fp16": False,
    "without_timestamps": True,
    "sample_len": 20,
    "language": "german",
}

results = []
result = openai_model.transcribe(
    input_features.squeeze(),
    task="transcribe",
    **openai_gen_kwargs,
)
print("EXPECTED_TRANSCRIPT: ", result["text"])

result = openai_model.transcribe(
    input_features.squeeze(),
    task="translate",
    **openai_gen_kwargs,
)
print("EXPECTED_TRANSCRIPT: ", result["text"])