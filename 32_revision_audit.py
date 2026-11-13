"""2026-10-19 descriptive audit and review packet; run from code/.
No semantic labels are generated. Abstracts are evidence, not independent validation.
"""
import collections
import difflib
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pipeline_config as pc
from revision_core import document_id, matched_size


def module(path, config, extra=()):
    old = sys.argv
    try:
        sys.argv = [path, '--config', config, *extra]
        spec = importlib.util.spec_from_file_location('audit_'+path[:2], path)
        m = importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
        return m
    finally:sys.argv = old


def citation_key(d):
    doi=(d.get('doi') or '').strip().lower()
    if doi:return 'doi:'+doi
    basis=pc.normalize_text(d.get('title') or '')+'|'+str(d.get('year') or '')
    return 't:'+hashlib.sha1(basis.encode()).hexdigest()[:16]


def anchor_rates(rows, resolved, start, end, lag):
    out={}
    for label,a in resolved.items():
        eligible=[r for r in rows if max(start,a['year']+lag)<=r['year']<=end]
        n=len(eligible);k=sum(a['oa'] in r['refs'] for r in eligible)
        out[label]={'eligible_from':max(start,a['year']+lag),'eligible_to':end,'lag':lag,
                    'n':n,'citers':k,'share':k/n if n else None}
    return out


def main():
    config=sys.argv[sys.argv.index('--config')+1]
    m=module('28_instrument_extraction.py',config);cfg=m.CFG
    out=cfg.results
    docs=[json.loads(l) for l in open(cfg['docs_path'])]
    works=[json.loads(l) for l in open(Path(cfg['docs_path']).parent/'citations/openalex_works.jsonl')]
    work_counts=collections.Counter(w['openalex_id'] for w in works)
    cutoff=cfg.get('revision_analysis_year_max',2024)
    resolved_keys={w['doc_key'] for w in works}; valid=[d for d in docs if d['year']<=cutoff]
    bystratum={}
    for st in sorted({d.get('stratum','unknown') for d in docs}):
        ds=[d for d in valid if d.get('stratum','unknown')==st]
        bystratum[st]={'n':len(ds),'resolved':sum(citation_key(d) in resolved_keys for d in ds)}
    coverage={'stored_records':len(docs),'stored_year_range':[min(d['year'] for d in docs),max(d['year'] for d in docs)],
              'after_2024':sum(d['year']>cutoff for d in docs),'analysis_records':len(valid),
              'after_2025_in_2025_bin':sum(d['year']>2025 and d['period']=='2025-2025' for d in docs),
              'resolved_analysis_records':sum(citation_key(d) in resolved_keys for d in valid),
              'resolved_without_references':sum(not w.get('referenced_works') for w in works if w['corpus_year']<=cutoff),
              'duplicate_openalex_ids':sum(n>1 for n in work_counts.values()),
              'records_in_repeated_openalex_resolutions':sum(n for n in work_counts.values() if n>1),
              'records_in_repeated_resolutions_through_2024':sum(work_counts[w['openalex_id']]>1 and w['corpus_year']<=cutoff for w in works),
              'unidentifiable_analysis_records':sum(document_id(d) is None for d in valid),'by_stratum':bystratum}
    mentions={d['document_id']:d['instruments'] for d in map(json.loads,open(out/'instrument_mentions.jsonl')) if d['document_id']}
    available=set(); packet_groups=collections.defaultdict(list)
    focal='ego depletion' if cfg['slug']=='ego_depletion' else 'trauma'
    for d,txt in m.iter_texts():
        ident=document_id(d)
        if ident:available.add(ident)
        # Balanced positive/negative source-review candidates, not prevalence sample.
        wanted=bool(set(d['terms']) & ({'ego depletion','self control'} if focal=='ego depletion' else {'trauma','ptsd','posttraumatic stress disorder','vicarious trauma','secondary traumatic stress'}))
        if wanted and ident:
            predicted=ident in mentions
            packet_groups[(predicted,d['period'])].append({'document_id':ident,'title':d['title'],'year':d['year'],'journal':d['journal'],'doi':d.get('doi'),'passage':txt})
    coverage['abstracts_available_unique_id']=len(available)
    rng=np.random.default_rng(20261019);sample=[]
    for positive in (False,True):
        groups=[rng.permutation(v).tolist() for (flag,p),v in sorted(packet_groups.items()) if flag==positive]
        for _ in range(30):
            live=[g for g in groups if g]
            if not live:break
            g=live[len([r for r in sample if r['document_id'] in mentions]) % len(live)] if positive else live[len(sample)%len(live)]
            sample.append(g.pop())
    rng.shuffle(sample)
    with open(out/'source_review_packet.jsonl','w') as f,open(out/'source_review_predictions.jsonl','w') as pred:
        for i,r in enumerate(sample):
            r['review_id']=cfg['slug']+'-'+str(i+1).zfill(3)
            r['review']={k:None for k in ['instrument_present','instrument_identity','version','study_role','focal_construct_relation','definition_or_commitment','full_text_needed','notes']}
            f.write(json.dumps(r)+'\n')
            pred.write(json.dumps({'review_id':r['review_id'],'predicted_instruments':mentions.get(r['document_id'],[])})+'\n')
    # Absolute contrasts in the SAME representation, on common complete periods.
    cells=json.load(open(Path(cfg['revision_baseline_results'])/'creep_breadth.json'))['cells']
    contrasts={}
    for term in (['ego depletion','self control','self regulation'] if focal=='ego depletion' else ['trauma','secondary traumatic stress','vicarious trauma','trauma informed']):
        contrasts[term]={}
        for rep in ['breadth_mpnet','breadth_masked']:
            periods=sorted({k.split('|')[1] for k in cells if k.startswith(term+'|') and int(k.split('|')[1][:4])<=2020})
            periods=[p for p in periods if all(cells.get(t+'|'+p,{}).get(rep) for t in [term,*cfg.controls])]
            if len(periods)<2:continue
            series=[]
            for p in periods:
                y=cells[term+'|'+p][rep]['mean'];ctl=float(np.mean([cells[t+'|'+p][rep]['mean'] for t in cfg.controls]))
                series.append({'period':p,'target':y,'mean_control':ctl,'difference':y-ctl})
            x=np.array([int(p[:4])+2 for p in periods]);y=np.array([r['difference'] for r in series])
            contrasts[term][rep]={'periods':series,'target_change':series[-1]['target']-series[0]['target'],
                'control_change':series[-1]['mean_control']-series[0]['mean_control'],
                'difference_in_changes':y[-1]-y[0],'difference_slope_per_year':float(np.polyfit(x,y,1)[0]),
                'status':'descriptive re-expression of saved raw breadth estimates; no CI; no equivalence claim; representations differ in encoder, masking, and aggregation'}
    report={'coverage':coverage,'breadth_contrasts':contrasts,'source_review_packet_n':len(sample)}
    if focal=='ego depletion':
        # Resolve anchors to corpus entries with the same bounded title matching as 27.
        wk={w['doc_key']:w for w in works if work_counts[w['openalex_id']]==1}; resolved={}
        for spec in cfg['citation']['anchors']+cfg['citation'].get('anchors_extra',[]):
            label,q,yr=spec
            if q.startswith('doi:'):
                w=wk.get(q);d=next((d for d in docs if citation_key(d)==q),None);score=1.
            else:
                candidates=[d for d in docs if not yr or abs(d['year']-yr)<=1]
                def score_title(d):
                    nt=pc.normalize_text(d['title']).lower();want=pc.normalize_text(q).lower()
                    return max(difflib.SequenceMatcher(None,want,nt).ratio(), .95 if want and want in nt else 0)
                d=max(candidates,key=score_title);score=score_title(d);w=wk.get(citation_key(d)) if score>=.85 else None
            if w and d:resolved[label]={'oa':w['openalex_id'],'year':int(yr or d['year']),'title':d['title'],'match_score':score,'doi':d.get('doi')}
        cat=pc.venue_categorizer(cfg)
        docbykey={citation_key(d):d for d in docs}
        rows=[]
        for w in works:
            if work_counts[w['openalex_id']] != 1:continue
            if 'ego depletion' not in w.get('terms',[]) or not 2017<=w['corpus_year']<=cutoff:continue
            d=docbykey.get(w['doc_key'])
            if d:
                rows.append({'id':document_id(d),'year':w['corpus_year'],'period':d['period'],'venue':cat(w.get('openalex_venue') or w.get('corpus_journal') or ''),'refs':set(w.get('referenced_works') or [])})
        anchors={'resolved':resolved,'eligible_rates_by_lag':{},'year_standardized':{}}
        cats=sorted({r['venue'] for r in rows})
        for lag in [0,1,2]:
            anchors['eligible_rates_by_lag'][str(lag)]={c:anchor_rates([r for r in rows if r['venue']==c],resolved,2017,cutoff,lag) for c in cats}
        # Equal year weighting on years available in BOTH work/organisational and core groups.
        work=next(c for c in cats if 'work' in c);core=next(c for c in cats if 'core' in c)
        for label,a in resolved.items():
            years=[y for y in range(max(2017,a['year']+1),cutoff+1) if all(any(r['year']==y and r['venue']==c for r in rows) for c in [work,core])]
            group={}
            for c in [work,core]:
                annual={str(y):anchor_rates([r for r in rows if r['venue']==c and r['year']==y],{label:a},y,y,1)[label] for y in years}
                group[c]={'annual':annual,'standardized_share':float(np.mean([v['share'] for v in annual.values()])) if annual else None}
            anchors['year_standardized'][label]={'years':years,'weighting':'equal weight per common year, not population weighting','groups':group}
        report['anchors']=anchors
        # Join at the document on comparable windows; no inherited partition needed.
        emb=module('30_correspondence_drift.py',config,['--terms','ego depletion']);shards=emb.load_shards()
        vectors={}
        for p,X in shards['ego_depletion'].items():
            ctl=np.vstack([shards[pc.slug(c)][p] for c in cfg.controls if p in shards.get(pc.slug(c),{})]).mean(0)
            for ident,v in zip(emb.SHARD_KEYS['ego_depletion'][p],X-ctl):vectors[str(ident)]=v
        common=[r for r in rows if r['id'] in available and r['id'] in vectors]
        joint=[]
        for c in [work,core]:
            rr=[r for r in common if r['venue']==c];fam=collections.Counter(i['family'] for r in rr for i in mentions.get(r['id'],[]))
            joint.append({'venue':c,'citation_eligible_n':sum(r['venue']==c for r in rows),'common_n':len(rr),
                          'n_naming_instrument':sum(r['id'] in mentions for r in rr),
                          'top_detected_families':fam.most_common(6),'anchor_rates':anchor_rates(rr,resolved,2017,cutoff,1)})
        report['joint_document_subset']={'window':[2017,cutoff],'unit':'unique identified focal-term document with an abstract, stored masked vector, and resolved citation record','rows':joint,
            'interpretation':'descriptive association on an explicitly shared population; name detection is not evidence of instrument role'}
        # Venue comparison conditioned on exactly the same publication years per draw.
        year_sizes={y:matched_size([sum(r['year']==y and r['venue']==c for r in common) for c in [work,core]],250) for y in range(2017,cutoff+1)}
        years=[y for y,n in year_sizes.items() if n>=1];draws=collections.defaultdict(list)
        for _ in range(100):
            a=[];b=[];fa=[];fb=[]
            for y in years:
                n=year_sizes[y]
                samples=[];floors=[]
                for c in [work,core]:
                    ids=[r['id'] for r in common if r['year']==y and r['venue']==c]
                    ix=rng.permutation(len(ids));samples.append(np.stack([vectors[ids[i]] for i in ix[:n]]))
                    floors.append(np.stack([vectors[ids[i]] for i in ix[n:2*n]]))
                a.extend(samples[0]);b.extend(samples[1]);fa.extend(floors[0]);fb.extend(floors[1])
            for label,A,B in [('between',a,b),('work_floor',a,fa),('core_floor',b,fb)]:
                for metric,value in emb.metrics(np.array(A),np.array(B)).items():draws[label+'_'+metric].append(value)
        report['joint_document_subset']['year_matched_representation_comparison']={'years':years,'n_per_year':{str(y):year_sizes[y] for y in years},'sample_size':sum(year_sizes[y] for y in years),'draws':100,'results':{k:{'mean':float(np.mean(v)),'sd':float(np.std(v))} for k,v in draws.items()},'uncertainty':'subsampling SD, not CI; known-use validation still pending'}
    out.joinpath('revision_audit.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({'coverage':coverage,'packet_n':len(sample)}));print(out/'revision_audit.json')


if __name__=='__main__':main()
