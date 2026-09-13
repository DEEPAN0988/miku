import json

with open("logs/phase8_tool_sft/evaluation_results.json", "r") as f:
    d = json.load(f)

ood = d["ood_phrasings"]
print(f"Total OOD cases: {len(ood)}")
fails = []
for i, c in enumerate(ood, 1):
    status = "PASS" if c["neural_action_match"] else "FAIL"
    print(f"{i:2d}. [{status}] \"{c['instruction']}\"")
    print(f"    Expected: {c['expected_action']} | Neural: {c['parsed_action']}")
    if not c["neural_action_match"]:
        fails.append(c)

print(f"\nNeural Passed: {len(ood) - len(fails)}/{len(ood)} ({(len(ood)-len(fails))/len(ood)*100:.1f}%)")
print(f"Neural Failed: {len(fails)}/{len(ood)}")
