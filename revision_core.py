"""Pure helpers for the 2026-10-19 corrections. No data or output side effects."""
import hashlib
import re
from collections import defaultdict

import numpy as np


def document_id(doc):
    doi = (doc.get('doi') or '').strip().lower()
    doi = re.sub(r'^https?://(?:dx\.)?doi\.org/', '', doi)
    if doi:
        return 'doi:' + doi
    title = re.sub(r'\s+', ' ', (doc.get('title') or '').strip().lower())
    if not re.search(r'[a-z0-9]', title):
        return None  # missing/punctuation-only title cannot establish document identity
    basis = title + '|' + str(doc.get('year') or '')
    return 'title-year:' + hashlib.sha256(basis.encode('utf-8')).hexdigest()


def matched_size(lengths, cap):
    """The between-set samples and every disjoint within-set half share n."""
    return min(int(cap), *(int(x) // 2 for x in lengths))


def consolidate_vectors(vectors, keys):
    """One equally weighted mean vector per document, with deterministic order."""
    if len(vectors) != len(keys):
        raise ValueError('Vector/key length mismatch')
    groups = defaultdict(list)
    for key, vector in zip(keys, vectors):
        groups[str(key)].append(vector)
    ordered = sorted(groups)
    if not ordered:
        return np.empty((0, vectors.shape[1])), np.array([], dtype=object)
    return np.stack([np.mean(groups[k], axis=0) for k in ordered]), np.array(ordered, dtype=object)


def distinct_edge_shares(mask, src, dst, bits):
    """Each outgoing edge counted once in a source-label denominator.

    Targets may have multiple labels, so row shares need not sum to one.
    """
    ma, mb = mask[src], mask[dst]
    counts = np.zeros((len(bits), len(bits)), dtype=float)
    denominators = np.zeros(len(bits), dtype=int)
    for i, bit in enumerate(bits):
        selected = (ma & bit) != 0
        denominators[i] = selected.sum()
        for j, target in enumerate(bits):
            counts[i, j] = np.count_nonzero(selected & ((mb & target) != 0))
    shares = np.divide(counts, denominators[:, None],
                       out=np.full_like(counts, np.nan), where=denominators[:, None] > 0)
    return counts, denominators, shares


def normalize_instrument_version(family, version):
    version = re.sub(r'\s+', ' ', version.lower().replace('-', ' ')).strip()
    if family == 'ptsd checklist' and version in {'5', 'dsm 5', 'for dsm 5'}:
        return '5'
    return version or 'base'
