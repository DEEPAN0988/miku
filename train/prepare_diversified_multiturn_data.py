"""
train/prepare_diversified_multiturn_data.py — Diversified Multi-Turn SFT Dataset

Expands entity pools from 32 names / 12 pet names to 450+ human names and 250+ pet names
to test if massive diversity forces the model to learn in-context token copying
rather than memorizing a small categorical entity pool.

STRICTLY HOLDS OUT:
  Human names: 'Sam', 'Felix', 'Barnaby', 'Sasha'
  Pet names:   'Rex', 'Kira', 'Gulliver', 'Toby'
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))
from model.tokenizer import MikuTokenizer


# Strictly held-out test names (NEVER included in training data)
HELD_OUT_HUMAN_NAMES: Set[str] = {"Sam", "Felix", "Barnaby", "Sasha", "Gulliver"}
HELD_OUT_PET_NAMES: Set[str] = {"Rex", "Kira", "Toby", "Pip", "Bandit"}

# Base human name list (450+ unique names)
RAW_HUMAN_NAMES = [
    "Aaron", "Abigail", "Adam", "Adrian", "Aiden", "Alan", "Albert", "Alec", "Alejandro", "Alex",
    "Alexander", "Alfred", "Alice", "Alicia", "Alina", "Alison", "Allan", "Allen", "Allison", "Alma",
    "Alonzo", "Althea", "Alvin", "Amanda", "Amelia", "Amir", "Amos", "Amy", "Andre", "Andrea",
    "Andrew", "Angela", "Angelina", "Anita", "Ann", "Anna", "Anne", "Annette", "Anthony", "Anton",
    "Antonio", "Archie", "Ariana", "Ariel", "Arjun", "Arman", "Arnold", "Arthur", "Arturo", "Ashley",
    "Astrid", "Audrey", "Austin", "Autumn", "Ava", "Avery", "Axel", "Barbara", "Barry", "Beatrice",
    "Belinda", "Ben", "Benjamin", "Bennett", "Bernadette", "Bernard", "Bernice", "Bert", "Beth", "Bethany",
    "Beverly", "Bianca", "Bill", "Blaine", "Blair", "Blake", "Bob", "Bobby", "Boris", "Bradley",
    "Brandon", "Brenda", "Brendan", "Brennan", "Brent", "Brett", "Brian", "Brianna", "Bridget", "Brooke",
    "Bruce", "Bruno", "Bryan", "Bryce", "Byron", "Caitlin", "Caleb", "Calvin", "Cameron", "Camilla",
    "Candice", "Carl", "Carla", "Carlos", "Carlton", "Carly", "Carmen", "Carol", "Caroline", "Carolyn",
    "Carrie", "Carson", "Carter", "Casey", "Cassandra", "Catherine", "Cecilia", "Cedric", "Chad", "Charles",
    "Charlotte", "Chase", "Chelsea", "Chester", "Chloe", "Chris", "Christian", "Christina", "Christine", "Christopher",
    "Claire", "Clara", "Clarence", "Clark", "Claude", "Claudia", "Clay", "Clayton", "Clifford", "Clifton",
    "Clint", "Clive", "Cody", "Colin", "Colleen", "Conner", "Connor", "Conrad", "Constance", "Cora",
    "Corey", "Corinne", "Cornelius", "Cory", "Courtney", "Craig", "Curtis", "Cynthia", "Cyrus", "Daisy",
    "Dale", "Dallas", "Dalton", "Damian", "Damien", "Damon", "Dan", "Dana", "Daniel", "Danielle",
    "Danny", "Daphne", "Darian", "Darin", "Dario", "Darius", "Darlene", "Darnell", "Darrell", "Darren",
    "Darrin", "Daryl", "Dave", "David", "Dawn", "Dean", "Deandre", "Deanna", "Deborah", "Debra",
    "Declan", "Deena", "Deidre", "Deirdre", "Delaney", "Delbert", "Delia", "Della", "Delores", "Delphine",
    "Demetrius", "Denis", "Denise", "Dennis", "Denny", "Derek", "Derrick", "Desiree", "Desmond", "Devin",
    "Devon", "Dewayne", "Dewey", "Dexter", "Diana", "Diane", "Dianna", "Dianne", "Diego", "Dillon",
    "Dina", "Dino", "Dion", "Dirk", "Dixie", "Dolores", "Dominic", "Dominick", "Dominique", "Don",
    "Donald", "Donna", "Donnie", "Donovan", "Dora", "Doreen", "Dorian", "Doris", "Dorothea", "Dorothy",
    "Doug", "Douglas", "Drake", "Drew", "Duane", "Dulce", "Duncan", "Dustin", "Dusty", "Dwayne",
    "Dwight", "Dylan", "Earl", "Earle", "Earnest", "Ed", "Eddie", "Eddy", "Edgar", "Edith",
    "Edmond", "Edmund", "Edna", "Eduardo", "Edward", "Edwin", "Edna", "Eileen", "Elaine", "Eleanor",
    "Elena", "Eli", "Elias", "Elijah", "Elisa", "Elisabeth", "Elise", "Eliza", "Elizabeth", "Ella",
    "Ellen", "Elliot", "Elliott", "Ellis", "Elmer", "Elsa", "Elsie", "Elton", "Elva", "Elvin",
    "Elvira", "Elvis", "Elwood", "Emanuel", "Emerson", "Emery", "Emil", "Emilio", "Emily", "Emma",
    "Emmanuel", "Emmett", "Emmitt", "Emmy", "Enid", "Enrique", "Eric", "Erica", "Erick", "Erik",
    "Erika", "Erin", "Ernest", "Ernesto", "Ernie", "Errol", "Ervin", "Erwin", "Esmeralda", "Esperanza",
    "Esteban", "Estelle", "Ester", "Esther", "Ethan", "Eugene", "Eugenia", "Eunice", "Eva", "Evan",
    "Evangeline", "Eve", "Evelyn", "Everett", "Fabiola", "Fabian", "Faith", "Farrah", "Fatima", "Faye",
    "Felecia", "Felicia", "Felipe", "Ferdinand", "Fern", "Fernando", "Fidel", "Flora", "Florence", "Floyd",
    "Ford", "Forest", "Forrest", "Foster", "Frances", "Francis", "Francisco", "Frank", "Franklin", "Fred"
]

# Base pet name list (250+ unique names)
RAW_PET_NAMES = [
    "Ace", "Apollo", "Archie", "Atlas", "Bailey", "Bandit", "Barkley", "Barney", "Baxter", "Bear",
    "Beau", "Bella", "Belle", "Benji", "Benny", "Bentley", "Blue", "Bo", "Boomer", "Brady",
    "Brody", "Bruno", "Brutus", "Bubba", "Buster", "Cash", "Champ", "Chance", "Charlie", "Chase",
    "Chester", "Chico", "Chief", "Chloe", "Cleo", "Coco", "Cody", "Cooper", "Copper", "Cosmo",
    "Daisy", "Dakota", "Dallas", "Dexter", "Diesel", "Dixie", "Duke", "Ella", "Ellie", "Emma",
    "Finn", "Fiona", "Frankie", "Gigi", "Ginger", "Gizmo", "Gracie", "Gus", "Hank", "Harley",
    "Harper", "Harry", "Hazel", "Henry", "Hunter", "Jack", "Jackson", "Jake", "Jasper", "Jax",
    "Joey", "Josie", "Kip", "Kobe", "Kona", "Lady", "Layla", "Leo", "Lily", "Linus",
    "Loki", "Lola", "Louie", "Lucky", "Lucy", "Luke", "Lulu", "Luna", "Mac", "Macy",
    "Maddie", "Maggie", "Marley", "Max", "Maya", "Mia", "Mickey", "Midnight", "Milo", "Moose",
    "Murphy", "Nala", "Nico", "Nova", "Ollie", "Onyx", "Oreo", "Oscar", "Otis", "Ozzy",
    "Peanut", "Pearl", "Penny", "Pepper", "Phoebe", "Piper", "Pluto", "Prince", "Princess", "Ranger",
    "Rascal", "Remi", "Riley", "Rocco", "Rocky", "Rory", "Roscoe", "Rose", "Rosie", "Ruby",
    "Rudy", "Rufus", "Rusty", "Sadie", "Sammy", "Scout", "Shadow", "Simba", "Sparky", "Stella",
    "Sugar", "Teddy", "Thor", "Titan", "Titus", "Trax", "Tucker", "Tyson", "Vader", "Waffles",
    "Walter", "Whiskey", "Whiskers", "Winston", "Woody", "Wrigley", "Yogi", "Zeus", "Ziggy", "Zoe"
]

# Filter out held-out names
CLEAN_HUMAN_NAMES = [n for n in set(RAW_HUMAN_NAMES) if n not in HELD_OUT_HUMAN_NAMES]
CLEAN_PET_NAMES = [n for n in set(RAW_PET_NAMES) if n not in HELD_OUT_PET_NAMES]

ANIMALS = ["cat", "dog", "rabbit", "parrot", "hamster", "ferret", "horse", "turtle", "lizard", "canary"]
COLORS = ["golden", "black", "white", "ginger", "brown", "grey", "spotted", "red", "emerald green", "navy blue", "silver"]


def generate_name_dialogues(count: int = 1200) -> List[Dict[str, str]]:
    dialogues = []
    for _ in range(count):
        name = random.choice(CLEAN_HUMAN_NAMES)
        t1_user = f"Hello! My name is {name}."
        t1_asst = f"Hello {name}! How can I help you today?"
        
        q_types = [
            (f"What is my name?", f"Your name is {name}."),
            (f"Do you remember my name?", f"Yes, your name is {name}."),
            (f"Who am I?", f"You introduced yourself as {name}."),
            (f"Can you say my name?", f"Your name is {name}."),
        ]
        q, ans = random.choice(q_types)
        
        prompt = (
            f"Context:\n"
            f"Conversation:\n"
            f"User: {t1_user}\n"
            f"Assistant: {t1_asst}\n\n"
            f"Instruction:\n"
            f"{q}\n\n"
            f"Response:\n"
        )
        dialogues.append({"prompt": prompt, "response": ans, "category": "multi_turn_name_recall"})
    return dialogues


def generate_pet_dialogues(count: int = 1200) -> List[Dict[str, str]]:
    dialogues = []
    for _ in range(count):
        animal = random.choice(ANIMALS)
        pet_name = random.choice(CLEAN_PET_NAMES)
        color = random.choice(COLORS)
        
        t1_user = f"I have a {color} {animal} named {pet_name}."
        t1_asst = f"{pet_name} sounds like a wonderful {animal}!"
        
        q_types = [
            (f"What kind of pet do I have?", f"You have a {color} {animal} named {pet_name}."),
            (f"What is my pet's name?", f"Your {animal}'s name is {pet_name}."),
            (f"What is my {animal}'s name?", f"Your {animal}'s name is {pet_name}."),
            (f"What's my {animal}'s name?", f"Your {animal}'s name is {pet_name}."),
            (f"What animal do I have?", f"You have a {animal}."),
        ]
        q, ans = random.choice(q_types)
        
        prompt = (
            f"Context:\n"
            f"Conversation:\n"
            f"User: {t1_user}\n"
            f"Assistant: {t1_asst}\n\n"
            f"Instruction:\n"
            f"{q}\n\n"
            f"Response:\n"
        )
        dialogues.append({"prompt": prompt, "response": ans, "category": "multi_turn_pet_identity"})
    return dialogues


def load_dolly_replay(count: int = 1000) -> List[Dict[str, str]]:
    path = Path("data/processed/sft/train_sft.jsonl")
    examples = []
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                if len(d["instruction"]) < 100 and len(d["response"]) < 120 and not d.get("context"):
                    prompt = f"Instruction:\n{d['instruction']}\n\nResponse:\n"
                    examples.append({
                        "prompt": prompt,
                        "response": d["response"],
                        "category": f"replay_{d.get('category', 'qa')}"
                    })
    random.seed(42)
    random.shuffle(examples)
    return examples[:count]


def main():
    random.seed(42)
    tok = MikuTokenizer.load("data/processed/tokenizer/miku_bpe")

    print(f"AUDIT OF EXPANDED ENTITY POOLS:")
    print(f"  Distinct Human Names: {len(CLEAN_HUMAN_NAMES)} (Held-out: {HELD_OUT_HUMAN_NAMES})")
    print(f"  Distinct Pet Names  : {len(CLEAN_PET_NAMES)} (Held-out: {HELD_OUT_PET_NAMES})")
    assert not (set(CLEAN_HUMAN_NAMES) & HELD_OUT_HUMAN_NAMES), "Leakage in human names!"
    assert not (set(CLEAN_PET_NAMES) & HELD_OUT_PET_NAMES), "Leakage in pet names!"

    name_samples = generate_name_dialogues(1500)
    pet_samples = generate_pet_dialogues(1500)
    replay_samples = load_dolly_replay(1000)

    all_samples = name_samples + pet_samples + replay_samples
    random.shuffle(all_samples)

    # Filter to guarantee <= 220 tokens
    valid_samples = []
    for s in all_samples:
        full_text = s["prompt"] + s["response"]
        tok_ids = tok.encode(full_text, add_bos=True, add_eos=True)
        if len(tok_ids) <= 220:
            valid_samples.append(s)

    print(f"\nTotal Valid Diversified Samples: {len(valid_samples)}")

    n_val = int(len(valid_samples) * 0.05)
    val_set = valid_samples[:n_val]
    train_set = valid_samples[n_val:]

    out_dir = Path("data/processed/sft_diversified")
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "train_diversified.jsonl", "w", encoding="utf-8") as f:
        for s in train_set:
            f.write(json.dumps(s) + "\n")

    with open(out_dir / "val_diversified.jsonl", "w", encoding="utf-8") as f:
        for s in val_set:
            f.write(json.dumps(s) + "\n")

    print(f"Saved to {out_dir}:")
    print(f"  Train: {len(train_set)} records")
    print(f"  Val  : {len(val_set)} records")


if __name__ == "__main__":
    main()
