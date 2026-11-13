"""Integrity checks for research provenance and separation; not scientific validation."""
import copy
import json
import unittest
from pathlib import Path
import reading_pilot as rp

OUT = rp.ROOT/'revision_work/2026-10-21/reading_pilot'

class UnitChecks(unittest.TestCase):
    def test_aliases_group_versions_across_years_and_doi_formats(self):
        a={'title':'A <i>sample</i> research title','doi':'https://doi.org/10.1/ABC','year':2020}
        b={'title':'A sample research title','doi':'doi:10.1/DEF','year':2021}
        groups=rp.IdentityGroups(); groups.add(rp.aliases(a)); groups.add(rp.aliases(b))
        self.assertEqual(groups.find('doi:10.1/abc'),groups.find('doi:10.1/def'))

    def test_string_spans_and_bad_spans(self):
        self.assertEqual(rp.span_bounds({'start':'1','end':'4'},'abcdef'),(1,4))
        for span in ({'start':-1,'end':4},{'start':4,'end':3},{'start':0,'end':99},{'start':'oops'}):
            self.assertIsNone(rp.span_bounds(span,'abcdef'))

    def test_quote_integrity_does_not_accept_fabrication(self):
        source={'blocks':[{'passage_id':'P0001','text':'We measured task persistence.'}]}
        good={'field':'operations','statement':'Candidate','status':'interpretation_requires_review',
              'evidence':[{'passage_id':'P0001','quote':'task persistence'}]}
        self.assertTrue(rp.validate_proposals([good],source))
        for edit in ({'status':'validated'},{'evidence':[]},{'evidence':[{'passage_id':'P0001','quote':'limited resources proved'}]}, {'field':'invented'}):
            bad=copy.deepcopy(good); bad.update(edit)
            with self.assertRaises(ValueError): rp.validate_proposals([bad],source)

    def test_no_hit_is_not_semantic_absence(self):
        result=rp.retrieve({'blocks':[{'passage_id':'P1','section':'Introduction','text':'x'*100}]},'ego')
        self.assertEqual(result['meaning']['passage_ids'],[])
        self.assertIn('no semantic judgment',result['meaning']['status'])

    def test_reference_section_not_retrieved(self):
        source={'blocks':[{'passage_id':'P1','section':'References','text':'self-control task scale measure '*10}]}
        self.assertEqual(rp.retrieve(source,'ego')['operations']['passage_ids'],[])

@unittest.skipUnless((OUT/'build_manifest.json').exists(),'Prepared local development artifacts required')
class ArtifactChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.selection=json.loads((OUT/'selection_manifest.json').read_text())['documents']
        cls.reserve=list(rp.rows(OUT/'reserved_evaluation_pool.jsonl'))

    def test_assignments_and_development_separation(self):
        self.assertEqual(len(self.selection),12)
        self.assertEqual(sum(s['case']=='ego' for s in self.selection),8)
        self.assertEqual(sum(s['arm']=='independent' for s in self.selection),4)
        reserved={a for r in self.reserve for a in r['aliases']}
        for s in self.selection:
            self.assertNotIn('doi:'+rp.doi_norm(s['doi']),reserved)
            self.assertNotIn('reserved_evaluation',s['split'])
        excluded=list(rp.rows(OUT/'known_development_exclusions.jsonl'))
        self.assertEqual(len(excluded),120)
        for r in excluded: self.assertTrue(reserved.isdisjoint(r['aliases']))

    def test_reservation_and_configuration_are_unchanged(self):
        manifest=json.loads((OUT/'reservation_manifest.json').read_text())
        self.assertEqual(rp.file_hash(OUT/'reserved_evaluation_pool.jsonl'),manifest['reserved_pool_sha256'])
        selection=json.loads((OUT/'selection_manifest.json').read_text())
        self.assertEqual(rp.file_hash(rp.ROOT/'code/configs/reading_pilot_20261021.json'),selection['config_sha256'])
        for path,hash_ in manifest['inputs_sha256'].items():
            self.assertEqual(rp.file_hash(rp.ROOT/path),hash_)

    def test_independent_packets_have_no_assistance(self):
        for spec in self.selection:
            if spec['arm']!='independent': continue
            folder=OUT/'packets'/spec['pilot_id']
            for name in ('assisted.html','candidate_evidence.json','model_request.json'):
                self.assertFalse((folder/name).exists())
            self.assertFalse((OUT/'coordinator/proposals'/f"{spec['pilot_id']}.json").exists())
            self.assertNotIn('Unreviewed model proposal:',(folder/'source.html').read_text())

    def test_all_sources_reconstruct_from_original_records(self):
        locks=json.loads((OUT/'source_lock.json').read_text())
        for spec in self.selection:
            saved=json.loads((OUT/'packets'/spec['pilot_id']/'source.json').read_text())
            p=saved['source_provenance']; path=rp.ROOT/p['file']
            if 'offset' in p:
                with path.open('rb') as f: f.seek(p['offset']); raw=f.read(p['bytes'])
                self.assertEqual(rp.digest(raw),p['raw_record_sha256'])
            else:
                raw=path.read_bytes(); self.assertEqual(rp.digest(raw),p['sha256'])
            record=raw.decode() if p['kind']=='pmc_html' else json.loads(raw)
            rebuilt=rp.normalize_source(record,p['kind'])
            self.assertEqual(saved['blocks'],rebuilt['blocks'])
            self.assertEqual(saved['raw_text'],rebuilt['raw_text'])
            self.assertEqual(rp.digest(saved['raw_text']),locks[spec['pilot_id']]['source_sha256'])
            if 'original_file' in p:
                self.assertEqual(rp.file_hash(rp.ROOT/p['original_file']),p['original_sha256'])

    def test_all_drafts_are_source_linked_and_unreviewed(self):
        count=0
        for spec in self.selection:
            if spec['arm']!='assisted': continue
            folder=OUT/'packets'/spec['pilot_id']
            source=json.loads((folder/'source.json').read_text())
            record=json.loads((OUT/'coordinator/proposals'/f"{spec['pilot_id']}.json").read_text())
            self.assertEqual(record['source_sha256'],rp.digest(source['raw_text']))
            self.assertTrue(rp.validate_proposals(record['proposals'],source))
            self.assertEqual(record['human_review_status'],'not_started')
            self.assertEqual({p['field'] for p in record['proposals']},set(rp.QUESTIONS))
            self.assertIn('not from an unattended API run',record['execution'])
            count+=len(record['proposals'])
        self.assertEqual(count,40)

    def test_documents_and_reading_forms_exist(self):
        for name in ('README.md','READER_GUIDE.md','index.html'): self.assertTrue((OUT/name).exists())
        for spec in self.selection:
            for name in ('source.html','review_form.md','review_form.json','original_source.txt'):
                self.assertTrue((OUT/'packets'/spec['pilot_id']/name).exists())

if __name__=='__main__': unittest.main()
