import json
path = 'eval/data/dev_blind_v1.jsonl'
with open(path, encoding='utf-8') as f:
    records = [json.loads(l) for l in f if l.strip()]

# These 3 items: close it / kill it / terminate that application
# resolve to 'notepad' which risks data loss, so is_destructive must be True
targets = {'blind_131', 'blind_133', 'blind_134'}
changed = 0
for r in records:
    if r.get('id') in targets:
        old = r.get('is_destructive')
        r['is_destructive'] = True
        changed += 1
        print('Updated ' + r['id'] + ': is_destructive ' + str(old) + ' -> True  text=' + r['text'])

with open(path, 'w', encoding='utf-8') as f:
    for r in records:
        f.write(json.dumps(r) + '\n')
print('Done. Changed ' + str(changed) + ' records.')
