"""Fail closed before deleting more: verify every file and trained dataset entry."""
import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / 'rice_leaf_app' / 'artifacts'
ledger = pd.read_csv(ARTIFACTS / 'more_import_manifest.csv')
source_dir = ROOT / 'more'
actual_sources = {p.relative_to(ROOT).as_posix() for p in source_dir.rglob('*') if p.is_file()}
assert actual_sources == set(ledger.source), 'Source files changed since import; do not delete more'
for row in ledger.itertuples():
    source, destination = ROOT / row.source, ROOT / row.destination
    assert source.resolve().is_relative_to(source_dir.resolve())
    assert destination.resolve().is_relative_to((ROOT / 'Rice_Leaf_Diease').resolve())
    assert hashlib.sha256(source.read_bytes()).hexdigest() == row.file_sha256
    assert hashlib.sha256(destination.read_bytes()).hexdigest() == row.file_sha256
manifest = pd.read_csv(ARTIFACTS / 'dataset_manifest.csv')
included = manifest[manifest.status.eq('included')]
assert set(ledger.destination) <= set(manifest.path), 'Model audit does not contain imported files'
assert set(ledger.pixel_sha256) <= set(included.pixel_sha256), 'Imported content missing from eligible dataset'
assert not included.pixel_sha256.duplicated().any()
expected = ledger[['pixel_sha256', 'split']].drop_duplicates()
actual = included[['pixel_sha256', 'split']]
joined = expected.merge(actual, on='pixel_sha256', suffixes=('_import', '_trained'), validate='one_to_one')
assert joined.split_import.eq(joined.split_trained).all(), 'Imported split changed unexpectedly'
metrics = json.loads((ARTIFACTS / 'metrics.json').read_text(encoding='utf-8'))
assert metrics['raw_count'] == len(manifest)
assert metrics['train_count'] == int((included.split == 'train').sum())
assert metrics['test_count'] == int((included.split == 'test').sum())
previous = json.loads((ARTIFACTS / 'metrics_before_more.json').read_text(encoding='utf-8'))
novel = ledger[ledger.reason.eq('new_pixel_group_80_20')].drop_duplicates('pixel_sha256')
for split in ('train', 'test'):
    assert metrics[f'{split}_count'] - previous[f'{split}_count'] == int(novel.split.eq(split).sum())
print(f'PASS: all {len(ledger)} source files have byte-identical copies and content in the retrained dataset; more can be deleted.')
