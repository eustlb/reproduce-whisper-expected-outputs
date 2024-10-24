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
ds = load_dataset("distil-whisper/meanwhile", "default")["test"]
ds = ds.cast_column("audio", Audio(sampling_rate=16000))

num_samples = 8

audio = ds[:num_samples]["audio"]
audios = [x["array"] for x in audio] 
#====================================================================================================

openai_model = whisper.load_model(openai_model_id)

openai_gen_kwargs = {
    "condition_on_previous_text": True,
    "fp16": False,
    "without_timestamps": False,
    "beam_size": 5,
    "logprob_threshold": -2.0,
}

texts = []
for audio in audios:
    input_features = processor(
        audio, return_tensors="pt", truncation=False, padding="longest", sampling_rate=16_000
    )["input_features"]
    input_features = input_features.to(device=torch_device)
    openai_outputs = openai_model.transcribe(
        input_features.squeeze(), 
        **openai_gen_kwargs,
    )
    texts.append(openai_outputs["text"])

print("EXPECTED_TEXT: ")
for text in texts:
    print(text)

# as in transformers@37ea04013b34b39c01b51aeaacd8d56f2c62a7eb
EXPECTED_TEXT = [
            " Folks, if you watch the show, you know I spent a lot of time right over there. Patiently and astutely scrutinizing the boxwood and mahogany chest set of the day's biggest stories, developing the central headline pawns, definitely maneuvering an oh-so-topical night to F6, faming of classic Sicilian, named or variation on the news, all the while seeing eight moves deep and patiently marshalling the latest press releases into a Fisher shows in lip-nitsky attack that culminates in the elegant lethal slow-played, all-pass on checkmate that is my nightly monologue, but sometimes sometimes folks I sometimes I start to the wake-up side down in the monkey bars of a condemned playground on a super fun site, get all hepped up on goofballs, rummage that would discard a tag bag of defective toys, yank out a fistball of disembodied doll limbs, toss them on a stain kid's place mad from a defunct denies, set up a table inside a rusty cargo container down by the warf and challenge toothless drifters to the godless bughouse blitz of tournament that is my segment, meanwhile.",
            " Folks, I spent a lot of time right over there night after night, actually. Carefully selecting for you the day's newsiest, most aerodynamic headlines, stress testing on those topical anti-lock breaks and power steering, painstakingly stitching, leather seating, so soft, it would make JD power and her associates blush. To create the luxury sedan that is my nightly monologue, but sometimes I just sometimes focus. I lurched to consciousness in the back of an abandoned school bus and slapped myself awake with a crusty floor mat. Before using a mouse-bitten timing belt to strap some old plywood to a couple of discarded oil drums, then by the light of a heathen-moon render a gas tank out of an empty big gulp, filled with white claw and de-natured alcohol, then light a match and let her rip in the dis-mented one man, soapbox derby of news that is my segment.",
            " Ladies and gentlemen, you know, I spent a lot of time right over there, raising the finest hosting news cattle firmly, yet tenderly milking the latest headlines from their jokes, swollen teats, churning the daily stories into the decadent Provincil style triple cream-breed. It is my nightly monologue, but sometimes sometimes I stagger home hungry after being released by the police and root around in the neighbor's trash can for an old milk carton scrape out the blooming dairy residue into the remains of a wet cheese rod I won from a rat in a pre-drawn street fight. Put it in a discarded paint can to leave it to ferment next to a trash fire than a hunker down in hallucinate while eating the Listeria latent demon custard of news that is my segment.",
            " Folks, you watched this show, you know I spend most of my time right over there, carefully sorting through the days, big stories, and selecting only the most subtle, and unblemished ostrich and crocodile news leather, which I then entrust to artisan graduates of the Ickel Greg Waferandi, who carefully died them in a pallet of bright, zesty shades, and adorn them in the finest most topical inlay work, using hand tools and double magnifying glasses, then assemble them according to now classic and elegant geometry using our signature saddle stitching, and line it with bees, wax, coated linen, and finally attach a mallet hammered strap, purled hardware, and close-shet to create for you the one of a kind hope kutur, Ernme, is burkin bag that is my monologue, but sometimes, sometimes folks, sometimes. Sometimes I wake up in the last car of an abandoned rollercoaster at Coney Island where I'm hiding from the triads, I have some engine lubricants out of a safe way bag and staggered down the shore to tear the sail off a beach skoener, then I ripped the coaxial cable out of an RV and elderly couple from Utah, Hank, and Mabel, lovely folks, and use it to stitch the sail into a loose pouch-like rock sack, and I stow in the back of a garbage truck to the junkyard, where I pick through to the debris for only the broken toys that make me the saddest, until I have loaded for you, the hobo fugitives bug out bindle of news that",
            " You know, folks, I spent a lot of time crafting for you a bespoke playlist of the day's big stories right over there. meticulously selecting the most topical chakra affirming scented candles, using Feng Shui, to perfectly align the joke energy in the exclusive boutique yoga retreat that is my monologue, but sometimes just sometimes, I go to the dumpster behind the waffle house at three in the morning, take off my shirt, cover myself and use fry oil, wrap my hands and some old duct tape I stole from a broken car window, pound a six pack of blueberry hard-seller and a second pill, as I stole from a parked ambulance, then arm wrestle a raccoon in the back alley vision quest of news that is my segment.",
            " You know, folks, I spend most of my time right over there. Mining the days, biggest, most important stories, collecting the finest, most topical iron or hand hammering it into joke panels, then I craft sheets of bronze and blazing with patterns that tell an epic tale of conquest and glory. Then, using the Germanic tradition press, black process, I place thin sheets of foil against the scenes and by hammering or otherwise applying pressure from the back, I project these scenes into a pair of cheat cards and a face plate, and finally using fluted strips of white, alloyed molding, I divide the designs into framed panels and hold it all together using bronze rivets to create the beautiful and intimidating, Anglo-Saxon battle helm that is my nightly monologue. But sometimes, sometimes, folks. Sometimes, just sometimes, I come to my senses fully naked on the deck of a pirate-be-seed, melee, container ship that picked me up floating on the detached door of a porta-potty in the Indian Ocean. Then, after a sunstroke induced realization of the crew of this ship plans to sell me an exchange for a bag of oranges to fight off scurvy, I lead a mutiny using only a PVC pipe and a pool chain that accepting my new role as captain and declaring myself King of the Windark Seas. I grab a dirty mop bucket covered in barnacles and adorn it with the teeth of the vanquished to create these shopping wet pirate crown of news that is my segment. Me wild!",
            " Folks, if you watch this show, you know I spend most of my time right over there carefully blending for you the day's newsiest, most topical flower eggs, milk and butter. And straining into a fine batter to make delicate and informative comedy pancakes, then I glaze them in the juice and zest of the most relevant midnight valencio oranges. And doubts at all, and I find delimane de voyage cognac, before from bang and basting them tables, I deserve you the James Beard Award worthy creeps to ZET. That is my nightly monologue, but sometimes sometimes folks, I wake up in the baggage hole of Greyhound bus, it's being hoisted by the scrapyard claw toward the burn pit. Escape to a nearby abandoned price chopper where I scrounge for old bread scraps, busted up in bags of starfruit candies and expired eggs. Chuck it all on a dirty hubcap and slap it over a tire fire before using the legs of a strained pair of sweatpants and as ovenmets to extract and serve the demented transients pound cake of news that is my segment.",
            (
                " Folks, if you watch the show and I hope you do, I spend a lot of time right over there. Tirelessly studying the lineage of the day's most important thoroughbred stories and whole-stiner headlines, working with the best trainers money can buy to rear their comedy offspring with a hand that is stern yet gentle into the triple crown winning equine specimen that is my nightly monologue. But sometimes sometimes folks I break into an unincorporated veterinary genetics lab. And grab whatever test tubes I can find and then under a grow light I got from a discarded chia pet. I mixed the pill for DNA of a horse and whatever was in a tube labeled Keith Cohen-Extra. Slurring the concoction with caffeine pills and a microwave bread bowl, I scream sing a prayer to Janice initiator of human life and God of Transformation as a half horse, half man freak ceases to life before me and the hideous collection of loose animal parts and corrupted men tissue that is my segment. Meanwhile!",
                " Folks, if you watch the show and I hope you do, I spend a lot of time right over there. Tirelessly studying the lineage of the day's most important thoroughbred stories and whole-stiner headlines, working with the best trainers money can buy to rear their comedy offspring with a hand that is stern yet gentle into the triple crown winning equine specimen that is my nightly monologue. But sometimes sometimes folks I break into an unincorporated veterinary genetics lab. And grab whatever test tubes I can find and then under a grow light I got from a discarded chia pet. I mixed the pill for DNA of a horse and whatever was in a tube labeled Keith Cohen-Extra. Slurring the concoction with caffeine pills and a microwave bread bowl, I screamed sing a prayer to Janice initiator of human life and God of Transformation as a half horse, half man freak ceases to life before me and the hideous collection of loose animal parts and corrupted men tissue that is my segment. Meanwhile!",
    )
]

for text, expected_text in zip(texts, EXPECTED_TEXT):
    if isinstance(expected_text, str):
        print(f"is correct ? {text == expected_text}")
    elif isinstance(expected_text, tuple):
        print(f"is correct ? {text== expected_text[0] or text == expected_text[1]}")
