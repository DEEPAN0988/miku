import json
with open('capabilities.json') as f:
    caps = json.load(f)
for k, v in caps.items():
    print(f"{k}: {len(v['phrasings_covered'])}")
