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
ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
ds = ds.cast_column("audio", Audio(sampling_rate=16000))
num_samples = 8

audio = ds[:num_samples]["audio"]
audios = [x["array"] for x in audio]

inputs = processor(
    audios,
    return_tensors="pt",
    sampling_rate=16_000,
)
inputs = inputs.to(device=torch_device)
#====================================================================================================

openai_model = whisper.load_model(openai_model_id)

openai_gen_kwargs = {
    "condition_on_previous_text": False,
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

print(f"EXPECTED_TEXT:")
for text in texts:
    print(text)

# as in transformers@37ea04013b34b39c01b51aeaacd8d56f2c62a7eb
EXPECTED_TEXT = [
    ' Mr. Kfilter is the apostle of the Middle Classes and we are glad to welcome his gospel.',
    " Nor is Mr. Qilter's manner less interesting than his matter.",
    ' He tells us that at this festive season of the year, with Christmas and roce beef, looming before us, similarly drawn from eating and its results occur most readily to the mind.',
    ' He has grabbed those with her surfered trigger late and his work is really a great after all, and can discover it in it but little of Rocky Ithaka.',
    " L'Neile's pictures are a sort of upguards and add-um paintings, and Maessin's exquisite Itals are a national as a jingo poem. Mr. Birkett Foster's landscapes smiled at one much in the same way that Mr. Carcher used to flash his teeth. And Mr. John Collier gives his sitter a cheerful slapper in the back, before he says,",
    ' It is obviously unnecessary for us, to point out how luminous these criticisms are, how delicate and expression.',
    ' On the general principles of art and Mr. Kriltor rights with equal lucidity.',
    ' Painting, he tells us is of a different quality to mathematics and finish in art is adding more effect.',
]

for text, expected_text in zip(texts, EXPECTED_TEXT):
    print(f"is correct ? {text == expected_text}")