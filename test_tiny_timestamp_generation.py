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

transformers_model_id = "openai/whisper-tiny"
openai_model_id = "tiny"

torch_device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float32

# ====================================================================================================
# LOAD DATA AND BUILD INPUT FEATURES AS IN THE ORIGINAL TEST 
def load_datasamples(num_samples):
    ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    speech_samples = ds.sort("id").select(range(num_samples))[:num_samples]["audio"]
    return [x["array"] for x in speech_samples]

processor = WhisperProcessor.from_pretrained(transformers_model_id)
input_speech = np.concatenate(load_datasamples(4))
input_features = processor(
    input_speech, return_tensors="pt", sampling_rate=16_000, return_token_timestamps=True
).input_features
input_features = input_features.to(torch_device, dtype=torch_dtype)
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

tokens = [torch.tensor(seg['tokens']) for seg in openai_outputs['segments']]
tokens = torch.cat(tokens, dim=0)

# we need to add decoder input token ids and eos token id
decoder_input_tokens = torch.tensor([50258, 50259, 50359,])
eos_token = torch.tensor([50257])
tokens = torch.cat([decoder_input_tokens, tokens, eos_token], dim=0)

print(f"EXPECTED_OUTPUT: {tokens}")

# as in transformers@37ea04013b34b39c01b51aeaacd8d56f2c62a7eb
EXPECTED_OUTPUT = torch.tensor([50258, 50259, 50359, 50364, 2221, 13, 2326, 388, 391, 307, 264, 50244, 295, 264, 2808, 5359, 11, 293, 321, 366, 5404, 281, 2928, 702, 14943, 13, 50692, 50692, 6966, 307, 2221, 13, 2326, 388, 391, 311, 9060, 1570, 1880, 813, 702, 1871, 13, 50926, 50926, 634, 5112, 505, 300, 412, 341, 42729, 3196, 295, 264, 1064, 11, 365, 5272, 293, 12904, 9256, 450, 10539, 51208, 51208, 949, 505, 11, 14138, 10117, 490, 3936, 293, 1080, 3542, 5160, 881, 26336, 281, 264, 1575, 13, 51552, 51552, 634, 575, 12525, 22618, 1968, 6144, 35617, 7354, 1292, 6, 589, 307, 534, 10281, 934, 439, 11, 293, 51836, 51836, 50257])  # fmt: skip

try:
    is_correct = torch.allclose(tokens, EXPECTED_OUTPUT)
except:
    is_correct = False

print(f"is correct ? {is_correct}")

for seg in openai_outputs['segments']:
    print(f"start: {seg['start']}, end: {seg['end']}, text: {seg['text']}")

