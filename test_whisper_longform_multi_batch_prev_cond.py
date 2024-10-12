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

ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean")
one_audio = np.concatenate([x["array"] for x in ds["validation"]["audio"]], dtype=np.float32)
audios = []
audios.append(one_audio[110000:])
audios.append(one_audio[:800000])
audios.append(one_audio[80000:])
audios.append(one_audio[:])
#====================================================================================================

openai_model = whisper.load_model(openai_model_id)

openai_gen_kwargs = {
    "condition_on_previous_text": True,
    "fp16": False,
    "without_timestamps": False,
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

print(f"EXPECTED_TEXT_1: {texts[0]}")
print(f"EXPECTED_TEXT_2: {texts[1]}")
print(f"EXPECTED_TEXT_3: {texts[2]}")
print(f"EXPECTED_TEXT_4: {texts[3]}")
