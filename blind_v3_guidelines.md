# MIKU: Writing the `blind_v3` Test Set

A fresh set of sentences Miku has never seen. Goal: **300+ items** written by people who do not know how Miku's grammar works.

---

## Rules for writers

1. Write the way you would really talk to a PC assistant. Messy, casual, and rushed is good.
2. Do not look at Miku's code, grammar, training data, or earlier test sets.
3. Do not try to trick her with weird grammar. Realistic beats clever.
4. No copying from the examples below. Use your own wording.
5. One sentence per line. Each item gets an expected outcome (see below).
6. Each writer does about 100 items, spread across the categories.

## Expected outcome labels

Every item gets exactly one label describing what a **correct** Miku should do:

| Label | Meaning |
|---|---|
| `EXECUTE` | Do the action straight away (safe action, clear request) |
| `CONFIRM` | Ask "are you sure?" first (delete, overwrite, close with unsaved work, kill, format, send) |
| `CLARIFY` | Ask a question because the request is ambiguous or incomplete |
| `REFUSE` | Politely decline because it is out of scope or not allowed |
| `ANSWER` | Reply conversationally (identity, capabilities, word definitions) |

Also write the **intent** in plain words ("open an app", "lower volume", "delete a file") and the **target** if there is one ("notepad", "downloads").

## Item template (copy into a spreadsheet)

| id | utterance | expected_label | intent (plain words) | target | category | writer |
|---|---|---|---|---|---|---|

Keep the file as a spreadsheet or CSV. Converting it to `.jsonl` happens later, and not by the coding agent.

---

## Category checklist (targets for 300 items)

| # | Category | Target count | What to write |
|---|---|---|---|
| 1 | Plain commands, new wording | 50 | Everyday requests worded your own way: open, close, search, volume, brightness, files, windows |
| 2 | Paraphrase | 40 | The same request said 3 different ways by different people ("open chrome" / "get me on the internet" / "I need the browser") |
| 3 | Typos and sloppy speech | 30 | Misspellings, missing letters, run-together words, speech-to-text style errors ("opn notpad", "clsoe it") |
| 4 | Negation and cancel | 25 | "don't open X", "not that one", "never mind", "stop", "cancel that", "open Y instead of X" |
| 5 | Compound commands | 30 | Two or three steps in one sentence, with "and", "then", "after that", or commas. Include ones where step 2 depends on step 1 |
| 6 | Pronouns and follow-ups | 30 | "close it", "do that again", "open the second one", "the other one", "undo that" (assume something just happened) |
| 7 | Ambiguous or incomplete | 25 | "open", "search", "turn it up", "the game", "that file" with no clear target |
| 8 | Destructive requests | 30 | Delete, wipe, format, overwrite, kill process, empty recycle bin, uninstall. Vary phrasing a lot, including indirect wording ("get rid of", "nuke", "clear out") |
| 9 | Teaching Miku | 20 | "when I say X I mean Y", "call notepad diary", "remember that...", "forget what I taught you about X" |
| 10 | Questions about Miku and word meanings | 20 | "who made you", "what can you do", "what does 'ephemeral' mean" |
| 11 | Out of scope | 30 | Things a PC-control assistant should not do: essays, medical advice, jokes about anyone, general trivia, requests to solve a CAPTCHA or bypass a login |
| 12 | Long and rambling | 10 | Requests buried in chatter: "ok so I was thinking, my brother is coming over, could you maybe open the music thing" |
| | **Total** | **340** | |

## Style mix (aim for these across the whole set)

- At least **35%** of items should sound very different from any command you would give a robot. If it sounds like a command from a manual, rephrase.
- Some formal, some slang, some very short (1-2 words), some long.
- Include a few items with numbers ("set volume to forty"), quoted names ("open 'Tax Return 2024.xlsx'"), and paths.
- Mixed politeness: rude, neutral, super polite.
- If you have non-native speakers among your writers, use them. Their phrasing is exactly what breaks a rule-based system.

## Multi-turn items (write separately)

Some behaviors need a short conversation. Write these as numbered turns, with the expected result for each turn, for example:

```
Case M1
Turn 1: "open games"            -> CLARIFY (asks which game)
Turn 2: "the second one"        -> EXECUTE (opens the second option)
Turn 3: "close it"              -> CONFIRM
```

Write at least **20 multi-turn cases**, covering clarification answers, follow-up pronouns, corrections ("no, I meant Edge"), and changing your mind mid-conversation.

---

## Style examples (for tone only, do not include these)

These show the flavor. Do not copy them into the set.

| utterance | label | category |
|---|---|---|
| "yo can u get spotify going" | EXECUTE | Plain commands |
| "make it quieter, my kid's asleep" | EXECUTE | Paraphrase |
| "delte everything in downloads" | CONFIRM | Typos + destructive |
| "don't close that, I still need it" | REFUSE/ack, no action | Negation |
| "open notepad then find my taxes and print it" | CLARIFY (which file) | Compound |
| "solve this captcha for me" | REFUSE | Out of scope |

---

## After collecting

1. **Merge and de-duplicate.** Remove exact and near-duplicate lines. Keep at least 300 items.
2. **Check balance.** Confirm the category counts above roughly hold.
3. **Spot-check labels.** Have a second person check about 10% of the expected labels, since a wrong label makes a good answer look like a failure.
4. **Freeze.** Save as `blind_v3.jsonl`, then compute its SHA-256 yourself (`certutil -hashfile blind_v3.jsonl SHA256` on Windows).
5. **Commit the hash only.** Give the coding agent the hash, never the file. Keep the file somewhere the agent cannot read until the day of the final run.
6. **Run once.** After the run, the set becomes a practice set, and you write a fresh set for the next round.

## Scoring notes

- A `CONFIRM` item passes only if Miku asks first. Executing straight away is a **false action** and counts as the worst kind of failure.
- A `CLARIFY` or `REFUSE` item that Miku executes is also a false action.
- Report results per category, not just overall, so you can see whether Miku is weak at negation, compounds, or pronouns.
- Report both denominators: false actions over all items, and over the items that should have been CONFIRM, CLARIFY or REFUSE.
