import json

with open("logs/phase8_tool_sft/evaluation_results.json", "r") as f:
    d = json.load(f)

for suite in ["in_distribution", "ood_phrasings", "novel_arguments"]:
    cases = d[suite]
    n = len(cases)
    neural_ok = sum(1 for c in cases if c["neural_action_match"])
    anchored_ok = sum(1 for c in cases if c["anchored_action_match"])
    arg_ok = sum(1 for c in cases if c["arg_match"])
    det_arg_ok = sum(1 for c in cases if c["deterministic_arg_match"])
    print(f"=== {suite.upper()} ({n} cases) ===")
    print(f"  Neural Action: {neural_ok}/{n} ({neural_ok/n*100:.1f}%)")
    print(f"  Anchored Action: {anchored_ok}/{n} ({anchored_ok/n*100:.1f}%)")
    print(f"  Model Arg Match: {arg_ok}/{n} ({arg_ok/n*100:.1f}%)")
    print(f"  Deterministic Fallback Arg: {det_arg_ok}/{n} ({det_arg_ok/n*100:.1f}%)")
    for c in cases:
        print(f"  Query: \"{c['instruction']}\"")
        print(f"    Expected: {c['expected_action']} | Neural: {c['parsed_action']} (OK: {c['neural_action_match']}) | Anchored: {c['anchored_action']} (OK: {c['anchored_action_match']})")
