"""Targeted regression checks for sample size, identity, and counting errors."""
import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np

from revision_core import (consolidate_vectors, distinct_edge_shares, document_id,
                           matched_size, normalize_instrument_version)


def load_script(name, config):
    old = sys.argv
    try:
        sys.argv = [name, '--config', config]
        spec = importlib.util.spec_from_file_location('test_' + name[:2], Path(name))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.argv = old


class Repairs(unittest.TestCase):
    def test_all_samples_fit_disjoint_halves(self):
        self.assertEqual(matched_size([263, 208], 250), 104)
        self.assertEqual(matched_size([55, 1011, 1852, 147], 250), 27)
        self.assertEqual(matched_size([1000, 501], 250), 250)

    def test_full_titles_do_not_collide(self):
        a = {'title': 'a' * 60 + ' affirmative'}
        b = {'title': 'a' * 60 + ' negative'}
        self.assertNotEqual(document_id(a), document_id(b))

    def test_missing_title_cannot_create_false_identity(self):
        self.assertIsNone(document_id({'title': '.', 'year': 1990}))
        self.assertNotEqual(document_id({'title':'Editorial','year':1990}), document_id({'title':'Editorial','year':1991}))
        self.assertEqual(document_id({'doi':'https://doi.org/10.1/ABC'}), document_id({'doi':'10.1/abc'}))

    def test_anchor_specific_eligibility(self):
        m = load_script('32_revision_audit.py', 'configs/ego_depletion_revision_20261019.yaml')
        rows = [{'year':2017,'refs':set()}, {'year':2021,'refs':{'b'}}, {'year':2022,'refs':{'b'}}, {'year':2024,'refs':set()}]
        r=m.anchor_rates(rows, {'old':{'year':2016,'oa':'a'}, 'new':{'year':2021,'oa':'b'}},2017,2024,1)
        self.assertEqual(r['old']['n'],4)
        self.assertEqual(r['new']['n'],2)
        self.assertEqual(r['new']['citers'],1)
        self.assertEqual(r['new']['share'],.5)

    def test_one_document_one_mean(self):
        X, keys = consolidate_vectors(np.array([[1., 0], [3, 0], [0, 1]]), ['a', 'a', 'b'])
        np.testing.assert_allclose(X, [[2, 0], [0, 1]])
        self.assertEqual(keys.tolist(), ['a', 'b'])

    def test_multilabel_targets_not_double_denominators(self):
        mask = np.array([1, 3, 2]);src = np.array([0, 0]);dst = np.array([1, 2])
        count, denom, share = distinct_edge_shares(mask, src, dst, [1, 2])
        self.assertEqual(denom[0], 2)
        np.testing.assert_allclose(share[0], [.5, 1.])
        self.assertTrue(np.isnan(share[1]).all())

    def test_version_aliases(self):
        for v in ['5', 'for dsm 5', 'for dsm-5', 'dsm 5']:
            self.assertEqual(normalize_instrument_version('ptsd checklist', v), '5')

    def test_floor_rejects_wrong_size(self):
        m = load_script('30_correspondence_drift.py', 'configs/ego_depletion_revision_20261019.yaml')
        with self.assertRaises(ValueError):
            m.floor(np.ones((208, 3)), np.random.default_rng(1), 208)

    def test_controlled_reweighting_and_new_vectors(self):
        m = load_script('30_correspondence_drift.py', 'configs/ego_depletion_revision_20261019.yaml')
        # Exact repeated prototypes isolate a known mixture change from a new vector.
        a = np.array([[1., 0., 0.]] * 80 + [[0., 1., 0.]] * 20)
        b = np.array([[1., 0., 0.]] * 20 + [[0., 1., 0.]] * 80)
        same = m.metrics(a, a);mixed = m.metrics(a, b)
        self.assertAlmostEqual(same['AMD'], 0.)
        self.assertAlmostEqual(mixed['AMD'], 0.)
        self.assertGreater(mixed['PRT'], .4)
        self.assertAlmostEqual(m.metrics(a, np.array([[0., 0., 1.]] * 100))['AMD'], 1.)

    def test_instrument_identity_and_version_counts(self):
        ego = load_script('28_instrument_extraction.py', 'configs/ego_depletion_revision_20261019.yaml')
        hits = ego.detect('Participants completed a Self-Control Scale.')
        self.assertIn('self control scale unspecified', [x['family'] for x in hits])
        trauma = load_script('28_instrument_extraction.py', 'configs/ptsd_trauma_revision_20261019.yaml')
        hits = trauma.detect('We administered the PTSD Checklist for DSM-5 (PCL-5).')
        self.assertEqual(len([x for x in hits if x['family']=='ptsd checklist' and x['version']=='5']), 1)


if __name__ == '__main__':
    unittest.main()
