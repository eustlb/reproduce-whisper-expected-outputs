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

ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean")
one_audio = np.concatenate([x["array"] for x in ds["validation"]["audio"]], dtype=np.float32)

input_features = processor(
one_audio, return_tensors="pt", truncation=False, padding="longest", sampling_rate=16_000
)["input_features"]
input_features = input_features.to(device=torch_device)
#====================================================================================================

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

EXPECTED_TEXT = openai_outputs['text']  
print(f"EXPECTED_TEXT: {EXPECTED_TEXT}")

tokens = [torch.tensor(seg['tokens']) for seg in openai_outputs['segments']]
tokens = torch.cat(tokens, dim=0)

decoded_with_timestamps = processor.decode(tokens, skip_special_tokens=False, decode_with_timestamps=True)
no_timestamp_matches = re.split(r"<\|[\d\.]+\|>", decoded_with_timestamps)
print("".join(no_timestamp_matches) == EXPECTED_TEXT)

timestamp_matches = re.findall(r"<\|[\d\.]+\|>", decoded_with_timestamps[0])
timestamp_floats = [float(t[2:-2]) for t in timestamp_matches]
is_increasing = all(timestamp_floats[i] <= timestamp_floats[i + 1] for i in range(len(timestamp_floats) - 1))
print(is_increasing)
