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

transformers_model_id = "openai/whisper-large-v3"
openai_model_id = "large-v3"

torch_device = "cuda:0" if torch.cuda.is_available() else "cpu"
set_seed(0)
torch_dtype = torch.float32

# ====================================================================================================
# LOAD DATA AND BUILD INPUT FEATURES AS IN THE ORIGINAL TEST 
def _load_datasamples(num_samples):
        ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        # automatic decoding with librispeech
        speech_samples = ds.sort("id").select(range(num_samples))[:num_samples]["audio"]

        return [x["array"] for x in speech_samples]

processor = WhisperProcessor.from_pretrained(transformers_model_id)
input_speech = _load_datasamples(4)
input_features = processor(input_speech, return_tensors="pt", sampling_rate=16_000).input_features
input_features = input_features.to(torch_device)
#====================================================================================================

openai_model = whisper.load_model(openai_model_id)

openai_gen_kwargs = {
    "condition_on_previous_text": False,
    "fp16": False,
    "without_timestamps": True,
    "sample_len": 20,
    "task": "translate"
}

results = []
for input_f in input_features:
    result = openai_model.transcribe(
        input_f, 
        **openai_gen_kwargs,
    )
    results.append(result)

for r in results:
    print("EXPECTED_TRANSCRIPT: ", r["text"])
    print("EXPECTED_LOGITS: ", r["segments"][0]["tokens"])