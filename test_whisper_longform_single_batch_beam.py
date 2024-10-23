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
    "condition_on_previous_text": True,
    "fp16": False,
    "without_timestamps": False,
    "beam_size": 2,
}

torch.manual_seed(0)
result = openai_model.transcribe(
    input_features.squeeze(), 
    **openai_gen_kwargs,
)

print(f"EXPECTED_TEXT: {result['text']}")

# as in transformers@37ea04013b34b39c01b51aeaacd8d56f2c62a7eb
EXPECTED_TEXT = [" Mr. Quilter is the apostle of the middle classes, and we are glad to welcome his gospel. Nor is Mr. Quilter's manner less interesting than his matter. He tells us that at this festive season of the year, with Christmas and roast beef looming before us, similes drawn from eating and its results occur most readily to the mind. He has grave doubts whether Sir Frederick Layton's work is really Greek after all, and can discover in it but little of rocky Ithaca. Linnell's pictures are a sort of up-gards and atom paintings, and Mason's exquisite idles are as national as a jingo poem. Mr. Burkett Foster's landscapes smile at one much in the same way that Mr. Carker used to flash his teeth. When Mr. John Collier gives his sitter a cheerful slap in the back, before he says, like a shampooer and a Turkish bath, next man, it is obviously unnecessary for us to point out how luminous these criticisms are, how delicate an expression. On the general principles of art, Mr. Quilter writes with equal lucidity. He tells us is of a different quality to mathematics, and finish in art is adding more effect. As for etchings, there are two kinds, British and foreign. He laments most bitterly the divorce that has been made between decorative art and what we usually call pictures. Mix a customary appeal to the last judgment and reminds us that in the great days of art with Michelangelo was the furnishing upholsterer. Near the fire, any ornaments Fred brought home from India on the mental board. In fact, he is quite severe on Mr. Ruskin for not recognizing that a picture should denote the frailty of man, and remarks was pleasing courtesy in felicitous grace that many faces are feeling. Only, unfortunately, his own work never does get good. Mr. Quilter has missed his chance, for he has failed even to make himself the topper of painting. By Harry Quilter, M.A., because he was sleeping instead of conquering, the lovely rose princess has become a fiddle without a bow, while poor Shaggy sits there, accooing dove. He has gone and gone for good, answered polychrome, who had managed to squeeze into the room beside the dragon, and had witnessed the occurrences with much interest. I have remained a prisoner only because I wished to be one. And with this, he stepped forward and burst the stout chains as easily as if they had been threads. The little girl had been asleep, but she heard the wraps and opened the door. The king has flooded this grace, and your friends are asking for you. I begged Ruggado long ago to send him away, but he would not do so. I also offered to help your brother to escape, but he would not go. He eats and sleeps very steadily, replied the new king. I hope he doesn't work too hard, since Shaggy. He doesn't work at all. In fact, there's nothing he can do in these dominions, as well as our gnomes, whose numbers are so great that it worries us to keep them all busy. Not exactly, we've turned Calico. Where is my brother now, inquired Shaggy. In the metal forest. Where is that? The metal forest is in the great domed cavern, the largest in all our dominions, replied Calico. Calico hesitated. However, if we look sharp, we may be able to discover one of these secret ways. Oh no, I'm quite sure he didn't. That's funny, remarked Betsy thoughtfully. I don't believe and knew any magic, or she'd have worked it before. I do not know, confessed Shaggy. True, a great Calico. Calico went to the big gong and pounded on it, just as Ruggado used to do, but no one answered the summons. Having returned to the Royal Cavern, Calico first pounded the gong and then sat in the throne, wearing Ruggado's discarded ruby crown, and holding in his hand to scepter which Ruggado had so often thrown at his head. A man said to the universe, Sir, I exist. Sweat covered Breon's body, trickling into the tight-laying cloth that was the only german who wore. The cut on his chest was still dripping blood. The ache of his overstrained eyes, even the soaring arena around him with thousands of spectators, retrovealities not worth thinking about. His instant panic was followed by a small, sharp, blow high on his chest. One minute, a voice said, and a time buzzer sounded, a minute is not a very large measure of time, and his body needed every fraction of it. The buzzers were, triggered his muscles into complete relaxation. Oli's heart and lungs worked on at a strong, measured rate. He was in reverie, sliding along the borders of consciousness. The contestants in the twenties needed undisturbed rest. Therefore, nights in the dormitories were as quiet as death. Particularly so, on this last night, when only two of the little cubicles were occupied, the thousands of others standing with dark empty doors. The other voice snapped with a harsh urgency clearly used to command. I'm here because the matter is of utmost importance, and brand is the one I must see. Now stand aside. The twenties, he must have drawn his gun because the intruder said quickly, but that away you're being a fool. Out there was silence then, and still wondering, Breon was once more asleep. Ten seconds, he asked the handler who was needing his aching muscles. A red-haired mountain of a man with an apparently inexhaustible store of energy. There could be little art in this last and final round of fencing. Just thrust and parry and victory to the stronger. Every man who entered the twenties had his own training tricks. There appeared to be an immediate association with the death trauma, as if the two were inextricably linked into one. The strength that enables someone in a trance to hold his body stiff and unsupported except at two points, the head and heels. This is physically impossible when conscious. Breon's head died before during the twenties and death during the last round was, in some ways, easier than defeat. Breeding deeply, Breon's softly spoke the auto-hypnotic phrases that triggered the process. When the buzzer sounded, he pulled his foil from his second startled grasp and ran forward. Our role looked amazed at the sudden fury of the attack, then smiled. He thought it was the last burst of energy. He knew how close they both were to exhaustion. Breon saw something close to panic on his opponent's face when the man finally recognized his error. A wave of despair rolled out from our rogue. Breon sensed it and knew the fifth point was his. In the powerful twist that's rest of the side, in and under the guard."]

print(f"is correct ? {result['text'] == EXPECTED_TEXT[0]}")
