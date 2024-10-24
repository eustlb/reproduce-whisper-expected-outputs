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

ds = load_dataset("distil-whisper/meanwhile", "default")["test"]
ds = ds.cast_column("audio", Audio(sampling_rate=16000))

num_samples = 8

audio = ds[:num_samples]["audio"]
audios = [x["array"] for x in audio]
#====================================================================================================

openai_model = whisper.load_model(openai_model_id)

openai_gen_kwargs = {
    "condition_on_previous_text": False,
    "fp16": False,
    "without_timestamps": False,
    "logprob_threshold": -2.0, # necessary to avoid trigering temp fallback that will introduce randomness
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

print("EXPECTED_TEXT: ")
for text in texts:
    print(text)

# as in transformers@37ea04013b34b39c01b51aeaacd8d56f2c62a7eb
EXPECTED_TEXT = [
    " Folks, if you watch the show, you know, I spent a lot of time right over there. Patiently and astutely scrutinizing the boxwood and mahogany chest set of the day's biggest stories developing the central headline pawns, definitely maneuvering an oso topical night to F6, fainting a classic Sicilian, nade door variation on the news, all the while seeing eight moves deep and patiently marshalling the latest press releases into a fisher's shows in Lip Nitsky attack that culminates in the elegant lethal slow-played, all-passant checkmate that is my nightly monologue. But sometimes, sometimes, folks, I. CHEERING AND APPLAUSE Sometimes I startle away, cubside down in the monkey bars of a condemned playground on a super fun site. Get all hept up on goofballs. Rummage that were discarded tag bag of defective toys. Yank out a fist bowl of disembodied doll limbs, toss them on a stained kid's place mat from a defunct dennies. set up a table inside a rusty cargo container down by the Wharf and challenged toothless drifters to the godless bughouse blitz of tournament that is my segment. Meanwhile.",
    " Folks, I spend a lot of time right over there, night after night after night, actually. Carefully selecting for you the day's noosiest, most aerodynamic headlines, stress testing, and those topical anti-lock breaks and power steering, painstakingly stitching, leather seating so soft, it would make JD power and her associates blush to create the luxury sedan that is my nightly monologue. But sometimes, you sometimes, folks. I lurched a consciousness in the back of an abandoned school and slap myself awake with a crusty floor mat. Before using a mouse-bitten timing belt to strap some old plywood to a couple of discarded oil drums, then by the light of a heathen moon, render a gas tank out of an empty big gulp, fill with white claw and denatured alcohol, then light a match and let her rip and the demented one man soapbox derby of news that is my segment. Me, Guadalupe! No!",
    " Ladies and gentlemen, you know, I spent a lot of time right over there Raising the finest Holstein news cattle firmly yet tenderly milking the latest headlines from their jokes swollen teats Churning the daily stories into the decadent proven-style style triple cream breed that is my nightly monologue But sometimes sometimes folks I stagger home hungry after being released by the police and Root around in the neighbor's trash can for an old milk carton scrape out the blooming dairy residue into the remains of a wet cheese rod I won from a rat in a pre-donned street fight. Put it in a discarded paint can to leave it to ferment next to a trash fire then hunker down and hallucinate while eating the listeria laden demon custard of news that is my segment. You mean one of them.",
    " Folks, if you watch this show, you know I spend most of my time right over there carefully sorting through the day's biggest stories and selecting only the most subtle and unblemished ostrich and crocodile news leather, which I then entrust to artisan graduates of the Ichol Gregoire Ferrandi, who carefully dye them in a palette of bright zesty shades and adorn them in the finest and most topical inlay work using hand tools and double magnifying glasses, then assemble them according to now classic and elegant geometry using our signature saddles stitching. In line it with bees, wax, coated linen, finely attached a mallet, hammered strap, pearled hardware, and close-shit to create for you the one-of-a-kind hoke couture, Erme's Birkin bag that is my monologue. But sometimes, sometimes folks, sometimes. Sometimes I wake up in the last car of an abandoned roller coaster at Coney Island where I'm I'm hiding from the triads. I have some engine lubricants out of a safe way bag and stagger down the shore to tear the sail off a beach schooner. Then I rip the coaxial cable out of an RV and elderly couple from Utah, Hank, and Mabel lovely folks. And use it to stitch the sail into a loose pouch like a rock sack. And I stow away in the back of a garbage truck to the junkyard where I pick through to the debris for only the broken toys that make me the saddest until I have loaded for you. The Hobo Fugitives bug out, bindle of news that is my segment. Me one!",
    " You know, folks, I spent a lot of time crafting for you a bespoke playlist of the day's biggest stories right over there. Meticulously selecting the most topical chakra affirming scented candles, and using Feng Shui to perfectly align the joke energy in the exclusive boutique yoga retreat that is my monologue. But sometimes just sometimes I go to the dumpster behind the waffle house at three in the morning, take off my shirt, cover myself, and used fry oil, wrap my hands with some double-duct tape by stole from the broken car window. Pound a six-pack of blueberry hard-seltzer and a sack of pills I stole from a parked ambulance. Then arm wrestle a raccoon in the back alley vision quest of news that is my segment. Meanwhile!",
    " You know, folks, I spend most of my time right over there. Mining the day's biggest, most important stories, collecting the finest, most topical iron or hand hammering it into joke panels. Then I craft sheets of bronze and blazing with patterns that tell an epic tale of conquest and glory. Then, using the Germanic tradition press-black process, I place thin sheets of foil against the scenes and by hammering or otherwise applying pressure from the back, I project these scenes into a pair of cheat cards in a faceplate and, finally, using fluted strips of white alloyed molding, I divide the designs into framed panels and hold it all together using bronze rivets to create the beautiful and intimidating, Anglo-Saxon battle helm that is my nightly monologue. Sometimes, sometimes folks. Sometimes, just sometimes, I come into my sense as fully naked on the deck of a pirate besieged melee container ship that picked me up floating on the detached door of a portapotty in the Indian Ocean. Then after a sunstroke-induced realization of the crew of this ship plans to sell me an exchange for a bag of oranges to fight off scurvy, I lead a mutiny using only a PVC pipe at a pool chain that accepting my new role as Captain and declaring myself king of the windarc seas. I grab a dirty mop bucket covered in barnacles and adorn it with the teeth of the vanquished to create the sopping wet pirate crown of news that is my segment. Meanwhile!",
    " Folks, if you watch this show, you know I spend most of my time right over there carefully blending for you the day's Newsiest most topical flower eggs milk and butter and Stranding into a fine batter to make delicate and informative comedy pancakes Then I glaze them in the juice and zest of the most relevant midnight Valencia oranges and douse it all and a fine Dela main de voyage cognac Before prom baying and basting them tables. I deserve for you the James Beard award worthy crepe suzzette That is my nightly monologue, but sometimes just sometimes folks. I wake up in the baggage hold of Greyhound bus. It's being hoisted by the scrap yard claw toward the burn pit. Escape to a nearby abandoned price chopper where I scrounge for old bread scraps and busted open bags of starfruit candies and expired eggs. Chuck it all on a dirty hubcap and slap it over a tire fire before using the legs of a strain, pair of sweatpants and as oven mitts to extract and serve the demented transience poundcake of news that is my segment. Me, Guadalupe!",
    " Folks, if you watched the show and I hope you do, I spent a lot of time right over there. Tiredlessly studying the lineage of the days most important thoroughbred stories and whole-stiner headlines, working with the best trainers, money can buy to rear their comedy offspring with a hand that is stern yet gentle into the triple crown winning equine specimen. That is my nightly monologue, but sometimes, sometimes, folks, I break into an unincorporated veterinary genetics lab and grab whatever test tubes I can find and then under a grow light I got from a discarded chia pet. I mixed the pilfered DNA of a horse and whatever was in a tube labeled Keith Colan extra. Slurrying the concoction with caffeine pills and a microwave red bull, I screamed, sang a prayer to Janice, initiator of human life and God of transformation as a half horse, half man, freak. Seizes to life before me and the hideous collection of loose animal parts and corrupted man tissue that is my segment. Meanwhile!"
]

print(f"is correct ? {texts[0] == EXPECTED_TEXT[0]}")
print(f"is correct ? {texts[1] == EXPECTED_TEXT[1]}")
print(f"is correct ? {texts[2] == EXPECTED_TEXT[2]}")
print(f"is correct ? {texts[3] == EXPECTED_TEXT[3]}")
print(f"is correct ? {texts[4] == EXPECTED_TEXT[4]}")
print(f"is correct ? {texts[5] == EXPECTED_TEXT[5]}")
print(f"is correct ? {texts[6] == EXPECTED_TEXT[6]}")
print(f"is correct ? {texts[7] == EXPECTED_TEXT[7]}")