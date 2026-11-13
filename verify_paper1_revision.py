"""Verify output invariants and preservation of original inputs/results."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from test_revision_repairs import load_script

ROOT=Path('..');checks=[]
def check(ok,label):
 if not ok:raise AssertionError(label)
 checks.append(label)
def digest(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
for name in ['manifest.json','input_manifest.json']:
 man=json.load(open(ROOT/'revision_work/2026-10-19/baseline'/name))
 protected=[r for r in man['files'] if name=='input_manifest.json' or r['path'].startswith('code/results/') or r['path'] in ['Paper_1_Draft_MR_v1.md','Concept_Audit_Roadmap_v1.md','Historical_Ground_Truth_Claims_v1.md','Historical_Ground_Truth_Claims_v1_Addendum_A.md']]
 for r in protected:check(digest(ROOT/r['path'])==r['sha256'],'preserved '+r['path'])
for case,base in [('ego_depletion','ego_depletion_full_v2'),('ptsd_trauma','ptsd_trauma_v2')]:
 out=Path('results')/(base+'_revision_20261019')
 drift=json.load(open(out/'correspondence_drift.json'))
 for term,t in drift['results'].items():
  for r in t['pairs']:
   check(r['n']==r['floor_n'] and 2*r['n']<=min(t['n'][r['from']],t['n'][r['to']]),f'{case} {term} {r["from"]} {r["to"]} equal floor n')
  check(all(int(p[:4])<=2020 for p in t['periods']),case+' complete periods '+term)
 for null in ['venue','journal','degree']:
  d=json.load(open(out/f'citation_communities_{null}null.json'))
  check(not d['term_distribution'] and not d['communities'],case+' no invented partition '+null)
  for r in d['insulation']:
   check(np.isclose(r['same_term_edges']/r['edges_from_term_docs'],r['same_term_share_obs']),case+' observed share '+r['term']+null)
   check(np.isclose(r['same_term_share_obs']/r['same_term_share_null'],r['insulation_ratio']),case+' share ratio '+r['term']+null)
 m=load_script('27_citation_communities.py',f'configs/{case}_revision_20261019.yaml');nodes,src,dst=m.load()
 check(len(nodes)==len({n['oa'] for n in nodes}),case+' unique graph work IDs')
 check(len(src)==len(set(zip(src.tolist(),dst.tolist()))),case+' unique graph edges')
 check(all(1995<=n['year']<=2024 for n in nodes),case+' graph dates')
 inst=[json.loads(l) for l in open(out/'instrument_mentions.jsonl')]
 check(all(len(r['instruments'])==len({(i['family'],i['version']) for i in r['instruments']}) for r in inst),case+' no duplicate family-version per document')
 check(all(r['year']<=2024 for r in inst),case+' instrument dates')
 packet=[json.loads(l) for l in open(out/'source_review_packet.jsonl')]
 check(len(packet)==60 and len({r['document_id'] for r in packet})==60,case+' 60 unique review documents')
 check(all(all(v is None for v in r['review'].values()) for r in packet),case+' human judgments not fabricated')
a=json.load(open('results/ego_depletion_full_v2_revision_20261019/revision_audit.json'))
for r in a['joint_document_subset']['rows']:
 check(r['anchor_rates']['rrr_2021']['eligible_from']==2022,'2021 anchor eligibility '+r['venue'])
 check(r['anchor_rates']['rrr_2021']['n']<r['common_n'],'anchor-specific denominator '+r['venue'])
c=json.load(open('results/ptsd_trauma_v2_revision_20261019/correspondence_cross.json'))
check(c['settings']['matched_n']==26 and c['settings']['set_sizes']['ANC']==53,'historical unique-document sample')
report={'passed':len(checks),'checks':checks,'status':'passed','scope':'numerical invariants, artifact preservation, and review-packet integrity; not scientific validation'}
(ROOT/'revision_work/2026-10-19/verification.json').write_text(json.dumps(report,indent=2))
print(f'Passed {len(checks)} checks; original data, outputs, and specified v1 documents preserved.')
