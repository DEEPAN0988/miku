"""
train/prepare_multiturn_data.py — Prepares Multi-Turn SFT Dataset

Builds a targeted multi-turn reference resolution dataset formatted with the
context-prefix template validated in v0.2:
  Context:
  Conversation:
  User: <turn 1 user>
  Assistant: <turn 1 asst>

  Instruction:
  <turn 2 user query>

  Response:
  <turn 2 expected response>

Also mixes in a 25% replay buffer of clean single-turn Dolly-15k examples to
prevent catastrophic forgetting of single-turn instruction following.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))
from model.tokenizer import MikuTokenizer


NAMES = [
    "Alice", "Bob", "Charlie", "Diana", "Evan", "Fiona", "George", "Hannah",
    "Ian", "Julia", "Kevin", "Laura", "Michael", "Nina", "Oliver", "Paula",
    "Quinn", "Rachel", "Sam", "Tina", "Umar", "Victoria", "Will", "Xena",
    "Yusuf", "Zoe", "Alex", "Jordan", "Taylor", "Morgan", "Casey", "Riley"
]

ANIMALS = ["cat", "dog", "rabbit", "parrot", "hamster", "ferret", "horse", "turtle"]
PET_NAMES = ["Whiskers", "Buddy", "Shadow", "Bella", "Luna", "Max", "Charlie", "Rocky", "Milo", "Daisy", "Coco", "Oliver"]
COLORS = ["golden", "black", "white", "ginger", "brown", "grey", "spotted", "red", "emerald green", "navy blue"]

CITIES_COUNTRIES = [
    ("Tokyo", "Japan"), ("Paris", "France"), ("London", "the United Kingdom"),
    ("Berlin", "Germany"), ("Rome", "Italy"), ("Ottawa", "Canada"),
    ("Canberra", "Australia"), ("Madrid", "Spain"), ("Beijing", "China"),
    ("Cairo", "Egypt"), ("Brasilia", "Brazil"), ("New Delhi", "India"),
    ("Seoul", "South Korea"), ("Stockholm", "Sweden"), ("Oslo", "Norway"),
    ("Athens", "Greece"), ("Dublin", "Ireland"), ("Vienna", "Austria")
]

LANDMARKS = [
    ("The Eiffel Tower", "Paris", "1889"),
    ("The Colosseum", "Rome", "80 AD"),
    ("The Statue of Liberty", "New York", "1886"),
    ("The Taj Mahal", "Agra", "1648"),
    ("The Great Pyramid", "Giza", "around 2560 BC"),
    ("The Parthenon", "Athens", "438 BC"),
    ("Big Ben", "London", "1859"),
    ("The Sydney Opera House", "Sydney", "1973"),
]

OBJECTS_COLORS = [
    ("car", "red"), ("bicycle", "blue"), ("backpack", "yellow"),
    ("jacket", "black"), ("notebook", "green"), ("hat", "purple"),
    ("guitar", "sunburst orange"), ("umbrella", "rainbow"),
]


def generate_name_dialogues(count: int = 600) -> List[Dict[str, str]]:
    dialogues = []
    for _ in range(count):
        name = random.choice(NAMES)
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
        dialogues.append({
            "prompt": prompt,
            "response": ans,
            "category": "multi_turn_name_recall"
        })
    return dialogues


def generate_pet_dialogues(count: int = 600) -> List[Dict[str, str]]:
    dialogues = []
    for _ in range(count):
        animal = random.choice(ANIMALS)
        pet_name = random.choice(PET_NAMES)
        color = random.choice(COLORS)
        
        t1_user = f"I have a {color} {animal} named {pet_name}."
        t1_asst = f"{pet_name} sounds like a wonderful {animal}!"
        
        q_types = [
            (f"What kind of pet do I have?", f"You have a {color} {animal} named {pet_name}."),
            (f"What is my pet's name?", f"Your {animal}'s name is {pet_name}."),
            (f"What animal do I have?", f"You have a {animal}."),
            (f"What color is my {animal}?", f"Your {animal} is {color}."),
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
        dialogues.append({
            "prompt": prompt,
            "response": ans,
            "category": "multi_turn_pet_identity"
        })
    return dialogues


def generate_pronoun_dialogues(count: int = 600) -> List[Dict[str, str]]:
    dialogues = []
    for _ in range(count):
        landmark, city, year = random.choice(LANDMARKS)
        
        t1_user = f"{landmark} is located in {city}."
        t1_asst = f"Yes, {landmark} is one of the most famous landmarks in {city}."
        
        q_types = [
            (f"When was it built?", f"It was built in {year}."),
            (f"Where is it?", f"It is located in {city}."),
            (f"What city is it in?", f"It is in {city}."),
            (f"Is it well known?", f"Yes, {landmark} is a world-famous landmark in {city}."),
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
        dialogues.append({
            "prompt": prompt,
            "response": ans,
            "category": "multi_turn_pronoun_resolution"
        })
    return dialogues


def generate_attribute_dialogues(count: int = 600) -> List[Dict[str, str]]:
    dialogues = []
    for _ in range(count):
        obj, col = random.choice(OBJECTS_COLORS)
        t1_user = f"I just bought a {col} {obj}."
        t1_asst = f"Congratulations on your new {col} {obj}!"
        
        q_types = [
            (f"What color is my {obj}?", f"Your {obj} is {col}."),
            (f"What did I just buy?", f"You bought a {col} {obj}."),
            (f"Do you remember the color of the {obj}?", f"Yes, the {obj} is {col}."),
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
        dialogues.append({
            "prompt": prompt,
            "response": ans,
            "category": "multi_turn_attribute_recall"
        })
    return dialogues


def generate_spatial_dialogues(count: int = 600) -> List[Dict[str, str]]:
    dialogues = []
    for _ in range(count):
        city, country = random.choice(CITIES_COUNTRIES)
        t1_user = f"I am planning a trip to {city} next summer."
        t1_asst = f"That sounds exciting! {city} is a wonderful destination in {country}."
        
        q_types = [
            (f"Where am I traveling to?", f"You are planning a trip to {city} in {country}."),
            (f"What country is that in?", f"{city} is located in {country}."),
            (f"When am I going there?", f"You mentioned you are planning to go next summer."),
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
        dialogues.append({
            "prompt": prompt,
            "response": ans,
            "category": "multi_turn_spatial_tracking"
        })
    return dialogues


def load_dolly_replay(count: int = 1000) -> List[Dict[str, str]]:
    """Load clean single-turn Dolly examples to prevent catastrophic forgetting."""
    path = Path("data/processed/sft/train_sft.jsonl")
    examples = []
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                # Keep concise single-turn examples
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

    print("Generating synthetic multi-turn dialogues...")
    multi_turn_samples = []
    multi_turn_samples.extend(generate_name_dialogues(600))
    multi_turn_samples.extend(generate_pet_dialogues(600))
    multi_turn_samples.extend(generate_pronoun_dialogues(600))
    multi_turn_samples.extend(generate_attribute_dialogues(600))
    multi_turn_samples.extend(generate_spatial_dialogues(600))
    print(f"Generated {len(multi_turn_samples)} targeted multi-turn samples.")

    print("Loading Dolly-15k single-turn replay buffer...")
    replay_samples = load_dolly_replay(1000)
    print(f"Loaded {len(replay_samples)} replay samples.")

    all_samples = multi_turn_samples + replay_samples
    random.shuffle(all_samples)

    # Filter by token length to guarantee strictly within 256 tokens
    valid_samples = []
    lengths = []
    for s in all_samples:
        full_text = s["prompt"] + s["response"]
        tok_ids = tok.encode(full_text, add_bos=True, add_eos=True)
        if len(tok_ids) <= 220:  # well within 256 limit
            valid_samples.append(s)
            lengths.append(len(tok_ids))

    print(f"\nTotal Valid Dataset Size: {len(valid_samples)} samples")
    print(f"Min Tokens: {min(lengths)}, Max Tokens: {max(lengths)}, Mean Tokens: {sum(lengths)/len(lengths):.1f}")

    # Split 95% train / 5% val
    n_val = int(len(valid_samples) * 0.05)
    val_set = valid_samples[:n_val]
    train_set = valid_samples[n_val:]

    out_dir = Path("data/processed/sft_multiturn")
    out_dir.mkdir(parents=True, exist_ok=True)

    train_file = out_dir / "train_multiturn.jsonl"
    val_file = out_dir / "val_multiturn.jsonl"

    with open(train_file, "w", encoding="utf-8") as f:
        for s in train_set:
            f.write(json.dumps(s) + "\n")

    with open(val_file, "w", encoding="utf-8") as f:
        for s in val_set:
            f.write(json.dumps(s) + "\n")

    print(f"\nSaved:")
    print(f"  Train: {train_file} ({len(train_set)} records)")
    print(f"  Val  : {val_file} ({len(val_set)} records)")


if __name__ == "__main__":
    main()
