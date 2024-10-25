import os 
current_dir = os.path.dirname(os.path.abspath(__file__))

# ensure we import the forked whisper package
import sys
sys.path.insert(0, os.path.join(current_dir,'whisper'))

import whisper
import torch
from transformers import WhisperProcessor
import numpy as np
from datasets import load_dataset, Audio

transformers_model_id = "openai/whisper-tiny"
openai_model_id = "tiny"

torch_device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float32

# ====================================================================================================
# LOAD DATA AND BUILD INPUT FEATURES AS IN THE ORIGINAL TEST 
processor = WhisperProcessor.from_pretrained(transformers_model_id)
ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation[:-1]")
ds = ds.cast_column("audio", Audio(sampling_rate=16_000))
input_speech = ds[0]["audio"]["array"]

input_features = processor(
    input_speech, return_tensors="pt", truncation=False, padding="longest", sampling_rate=16_000
)["input_features"]
input_features = input_features.to(device=torch_device)
#====================================================================================================

openai_model = whisper.load_model(openai_model_id)

prompt = "Mr. Kilter, Brionno."  # let's force Quilter -> Kilter, Brion -> Brionno

openai_gen_kwargs = {
    "condition_on_previous_text": True,
    "fp16": False,
    "without_timestamps": False,
    "initial_prompt": prompt,
}

result = openai_model.transcribe(
    input_features.squeeze(), 
    **openai_gen_kwargs,
)

print("EXPECTED_TEXT: ", result["text"])
