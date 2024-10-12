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

transformers_model_id = "openai/whisper-large-v3"
torch_device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float32

# ====================================================================================================
# LOAD DATA AND BUILD INPUT FEATURES AS IN THE ORIGINAL TEST 
def load_datasamples(num_samples):
    ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    speech_samples = ds.sort("id").select(range(num_samples))[:num_samples]["audio"]
    return [x["array"] for x in speech_samples]

# Load and process input data
processor = WhisperProcessor.from_pretrained(transformers_model_id)
input_speech = np.concatenate(load_datasamples(4))
input_features = processor(
    input_speech, return_tensors="pt", sampling_rate=16_000, return_token_timestamps=True
).input_features
input_features = input_features.to(torch_device, dtype=torch_dtype)
#====================================================================================================

openai_model_id = "large-v3"
openai_model = whisper.load_model(openai_model_id)

openai_gen_kwargs = {
    "condition_on_previous_text": False,
    "fp16": False,
    "without_timestamps": False,
}

openai_outputs = openai_model.transcribe(
    input_features.squeeze(), 
    **openai_gen_kwargs,
)

tokens = [torch.tensor(seg['tokens']) for seg in openai_outputs['segments']]
tokens = torch.cat(tokens, dim=0)

print(f"EXPECTED_OUTPUT: {tokens}")

for seg in openai_outputs['segments']:
    print(f"start: {seg['start']}, end: {seg['end']}, text: {seg['text']}")