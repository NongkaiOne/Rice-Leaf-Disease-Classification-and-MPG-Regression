"""Copy labeled additions into existing splits; keep a verified migration ledger."""
import hashlib
import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from rice_leaf_app.config import DATA_DIR, ARTIFACTS, SEED
from rice_leaf_app.dataset import inspect_image

SOURCE = ROOT / 'more'
MAPPING = {'Healthy': 'healthy', 'Leaf Blast': 'leaf_blast', 'Sheath Blight': 'sheath_blight'}


def main():
    old = pd.read_csv(ARTIFACTS / 'dataset_manifest.csv').fillna('')
    old_index = defaultdict(list)
    for row in old.to_dict('records'):
        if row['pixel_sha256']:
            old_index[row['pixel_sha256']].append(row)
    items = []
    for folder in sorted(SOURCE.iterdir()):
        if not folder.is_dir() or folder.name not in MAPPING:
            raise ValueError(f'Unexpected source entry: {folder}')
        for path in sorted(folder.rglob('*')):
            if path.is_file():
                if path.suffix.lower() not in {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}:
                    raise ValueError(f'Unsupported file; source must not be deleted: {path}')
                items.append((path, 'unassigned', MAPPING[folder.name]))
    print(f'Inspecting {len(items)} incoming images...', flush=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        records = []
        for i, row in enumerate(pool.map(inspect_image, items), 1):
            records.append(row)
            if i % 500 == 0:
                print(f'Inspected {i}/{len(items)}', flush=True)
    groups = defaultdict(list)
    for row in records:
        if row['status'] != 'candidate':
            raise ValueError(f'Unreadable image; keeping more: {row}')
        groups[row['pixel_sha256']].append(row)
    assigned = {}
    novel = defaultdict(list)
    for digest, rows in groups.items():
        labels = {row['label'] for row in rows + old_index[digest]}
        if len(labels) != 1:
            raise ValueError(f'Conflicting labels; keeping more: {labels} {rows[0]["path"]}')
        matches = old_index[digest]
        if matches:
            # Respect the original test holdout, including originals duplicated in train.
            anchor = min(matches, key=lambda row: row['split'] != 'test')
            actual = inspect_image((ROOT / anchor['path'], anchor['split'], anchor['label']))
            if actual.get('pixel_sha256') != digest:
                raise ValueError('Existing manifest is stale. Re-audit before importing.')
            assigned[digest] = (anchor['split'], 'matches_existing_pixels')
        else:
            novel[rows[0]['label']].append(digest)
    rng = np.random.default_rng(SEED)
    for label, digests in sorted(novel.items()):
        digests = sorted(digests)
        rng.shuffle(digests)
        n_test = max(1, round(len(digests) * .2)) if len(digests) > 1 else 0
        for i, digest in enumerate(digests):
            assigned[digest] = ('test' if i < n_test else 'train', 'new_pixel_group_80_20')
    ledger = []
    for row in records:
        source = ROOT / row['path']
        split, reason = assigned[row['pixel_sha256']]
        folder_name = {'sheath_blight': 'Sheath Blight'}.get(row['label'], row['label']) if split == 'test' else row['label']
        folder = DATA_DIR / split / folder_name
        assert folder.is_dir(), folder
        file_digest = hashlib.sha256(source.read_bytes()).hexdigest()
        destination = folder / f'more_{file_digest}{source.suffix.lower()}'
        if not destination.exists():
            shutil.copy2(source, destination)
        if hashlib.sha256(destination.read_bytes()).hexdigest() != file_digest:
            raise ValueError(f'Copy verification failed: {destination}')
        ledger.append(dict(source=row['path'], destination=destination.relative_to(ROOT).as_posix(),
                           label=row['label'], split=split, reason=reason,
                           file_sha256=file_digest, pixel_sha256=row['pixel_sha256']))
    pd.DataFrame(ledger).to_csv(ARTIFACTS / 'more_import_manifest.csv', index=False)
    summary = dict(source_count=len(ledger), copied_unique_files=len({r['destination'] for r in ledger}),
                   new_pixel_groups=sum(len(v) for v in novel.values()), seed=SEED,
                   counts=pd.DataFrame(ledger).groupby(['label','split']).size().rename('count').reset_index().to_dict('records'),
                   source_deleted=False)
    (ARTIFACTS / 'more_import_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
