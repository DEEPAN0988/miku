import json
from collections import Counter

data = json.load(open('eval/dev_blind_v1_audit.json'))
print(f"Total failures: {len(data['failed_items'])}")
print("\n--- Failure Breakdown by Expected Intent ---")
exp_counts = Counter(f['expected_intent'] for f in data['failed_items'])
for exp, count in exp_counts.most_common():
    print(f"  {exp:<25}: {count}")

print("\n--- Failure Breakdown by Utterance Pattern / Cause ---")
causes = Counter()
for f in data['failed_items']:
    u = f['utterance'].lower()
    exp = f['expected_intent']
    pred = f['predicted_intent']
    if exp == 'TEACH_WORD_ALIAS':
        causes['teaching_phrasing'] += 1
    elif exp == 'OUT_OF_SCOPE':
        causes['out_of_scope_open_or_query'] += 1
    elif exp == 'CLARIFICATION_NEEDED':
        causes['clarification_needed_not_asked'] += 1
    elif exp == 'NEGATION_REFUSAL':
        causes['negation_refusal_miss'] += 1
    elif exp == 'COMPOUND_COMMAND':
        causes['compound_not_split'] += 1
    elif exp == 'PRONOUN_COMMAND':
        causes['pronoun_resolution_miss'] += 1
    elif 'open' in u or 'launch' in u or 'start' in u:
        causes['paraphrase_open_app'] += 1
    elif 'volume' in u or 'mute' in u or 'sound' in u:
        causes['paraphrase_volume'] += 1
    elif 'calc' in u or 'math' in u or 'plus' in u or 'minus' in u:
        causes['paraphrase_calc'] += 1
    elif 'time' in u or 'date' in u:
        causes['paraphrase_time'] += 1
    elif 'file' in u or 'delete' in u or 'remove' in u:
        causes['destructive_or_file'] += 1
    else:
        causes['other_command_paraphrase'] += 1

for cause, count in causes.most_common():
    print(f"  {cause:<30}: {count}")

print("\n--- First 10 Failed Items ---")
for i, f in enumerate(data['failed_items'][:10], 1):
    print(f"[{i:02d}] \"{f['utterance']}\" | Expected: {f['expected_intent']} | Actual: {f['predicted_intent']} (Conf: {f['confidence']})")

