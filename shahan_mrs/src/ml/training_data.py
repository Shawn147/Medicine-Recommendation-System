"""Prepare condition splits and training weights without crossing holdouts."""
from collections import Counter, defaultdict

import numpy as np
from sklearn.model_selection import train_test_split


def split_dataset(rows):
    """Use fixed holdouts when supplied; otherwise make stratified 60/20/20 splits."""
    indices = np.arange(len(rows))
    target = np.array([row['condition'] for row in rows])
    if all('split' in row for row in rows):
        train, val, test = [
            np.array([i for i, row in enumerate(rows) if row['split'] == split])
            for split in ('train', 'validation', 'test')
        ]
        families = [{rows[i]['family_id'] for i in part} for part in [train, val, test]]
        if families[0] & families[1] or families[0] & families[2] or families[1] & families[2]:
            raise ValueError('Training families overlap held-out examples')
    else:
        trainval, test = train_test_split(indices, test_size=0.2, stratify=target, random_state=42)
        train, val = train_test_split(trainval, test_size=0.25, stratify=target[trainval], random_state=42)
    assert not (set(train) & set(test) or set(train) & set(val) or set(val) & set(test))
    return train, val, test


def calculate_training_weights(rows, train):
    """Give each condition and each training family equal total influence."""
    target = np.array([row['condition'] for row in rows])
    family_counts = Counter(rows[i].get('family_id', rows[i]['id']) for i in train)
    label_families = defaultdict(set)
    for i in train:
        label_families[target[i]].add(rows[i].get('family_id', rows[i]['id']))
    weights = np.ones(len(rows), dtype=np.float32)
    for i in train:
        family = rows[i].get('family_id', rows[i]['id'])
        families_in_condition = len(label_families[target[i]])
        weights[i] = 1 / (family_counts[family] * families_in_condition)
    weights[train] *= len(train) / weights[train].sum()
    return weights
