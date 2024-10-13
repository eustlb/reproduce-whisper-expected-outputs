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

ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation[:-1]")
one_audio = np.concatenate([x["array"] for x in ds["audio"]], dtype=np.float32)

input_features = processor(
    one_audio, return_tensors="pt", truncation=False, padding="longest", sampling_rate=16_000
)["input_features"]
input_features = input_features.to(device=torch_device)
#====================================================================================================

openai_model = whisper.load_model(openai_model_id)

openai_gen_kwargs = {
    "condition_on_previous_text": True,
    "fp16": False,
    "without_timestamps": False,
}

prompt = "Mr. Kilter, Brionno."  # let's force Quilter -> Kilter, Brion -> Brionno

# this is equivalent only to the "first-segment" prompt_condition_type in transformers
openai_outputs = openai_model.transcribe(
    input_features.squeeze(), 
    initial_prompt=prompt,
    **openai_gen_kwargs,
)

first_text = ds[0]["text"].lower()
last_text = ds[-1]["text"].lower()

# condition on first segment correctly changes to kilter in first segment, but does not transcribe "brianno" correctly
assert "kilter" in openai_outputs["text"][: len(first_text)].lower()
assert "brionno" not in openai_outputs["text"][-len(last_text) :].lower()
