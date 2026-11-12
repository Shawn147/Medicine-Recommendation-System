"""Train with fixed legacy holdouts and family weights; illustrative variants stay in training."""
import argparse
import json
import os
import random
import time

from src.paths import ROOT
os.environ.setdefault('USE_TF', '0')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from safetensors.torch import load_file
from transformers import AutoModel, AutoTokenizer
from src.data.loading import ARTIFACT_DIR, fingerprint, load_data
from src.ml.compact_weights import dump_compact_estimator
from src.ml.condition_baseline import train_baseline
from src.ml.condition_model import BASE_MODEL, BASE_REVISION, MAX_LENGTH
from src.ml.condition_transformer import ConditionTransformer
from src.ml.evaluation import evaluate
from src.ml.features import symptom_text
from src.ml.model_selection import select_model
from src.ml.training_data import calculate_training_weights, split_dataset
from src.ml.transformer_training import fine_tune


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=12)
    parser.add_argument('--learning-rate', type=float, default=2e-5)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error('--epochs must be positive')
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.set_num_threads(4)
    # 1. Load validated examples and create leakage-free splits.
    data = load_data()
    rows = data['training']
    labels = sorted({r['condition'] for r in rows})
    indices = np.arange(len(rows))
    target = np.array([r['condition'] for r in rows])
    train, val, test = split_dataset(rows)
    texts = [symptom_text(r['symptoms']) for r in rows]
    out = ARTIFACT_DIR
    out.mkdir(parents=True, exist_ok=True)
    model_dir = out/'transformer'
    model_dir.mkdir(exist_ok=True)
    # Invalidate the completion marker while a new training run is in progress.
    if (out/'metrics.json').exists():
        (out/'metrics.json').replace(out/'previous_metrics.json')
    splits = {name: [rows[i]['id'] for i in part] for name, part in [('train',train),('validation',val),('test',test)]}
    (out/'splits.json').write_text(json.dumps(splits, indent=2)+'\n')
    print('SPLITS', {k:len(v) for k,v in splits.items()}, flush=True)
    # 2. Balance training families and fit the TF-IDF baseline.
    weights = calculate_training_weights(rows, train)
    start = time.perf_counter()
    baseline_val, baseline_c, baseline = train_baseline(
        [texts[i] for i in train], [texts[i] for i in val],
        target[train], target[val], weights[train],
    )
    baseline_seconds = time.perf_counter()-start
    dump_compact_estimator(baseline, out/'baseline.joblib')
    # 3. Encode symptom text with pretrained MiniLM.
    pretrained = ROOT/'artifacts/pretrained/minilm'
    if not (pretrained/'model.safetensors').exists():
        raise FileNotFoundError('Run python scripts/download_model.py first to download the pretrained encoder.')
    tokenizer = AutoTokenizer.from_pretrained(pretrained, local_files_only=True)
    encoder = AutoModel.from_pretrained(pretrained, local_files_only=True, use_safetensors=True)
    model = ConditionTransformer(encoder, len(labels))
    encoded = tokenizer(texts, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors='pt')
    y = torch.tensor([labels.index(t) for t in target], dtype=torch.long)
    def batch(part):
        return {k:v[part] for k,v in encoded.items()}
    def probabilities(part):
        model.eval()
        with torch.inference_mode():
            return torch.cat([model(batch(part[i:i+16])).softmax(-1) for i in range(0,len(part),16)]).numpy()
    start = time.perf_counter()
    model.eval()
    with torch.inference_mode():
        embeddings = torch.cat([model.embed(batch(indices[i:i+16])) for i in range(0,len(rows),16)]).numpy()
    # Selection uses the fixed original validation examples; generated siblings never become holdouts.
    # 4. Train a classifier on frozen embeddings, then initialize the fine-tuning head.
    frozen_val, head_c, head = select_model(lambda c: LogisticRegression(C=c, max_iter=2000, random_state=42),
                                     embeddings[train], embeddings[val], target[train], target[val], weights[train])
    dump_compact_estimator(head, out/'frozen_head.joblib')
    assert list(head.classes_) == labels
    with torch.no_grad():
        model.head.weight.copy_(torch.tensor(head.coef_, dtype=torch.float32))
        model.head.bias.copy_(torch.tensor(head.intercept_, dtype=torch.float32))
    frozen_val = float(accuracy_score(target[val], head.predict(embeddings[val])))
    # 5. Fine-tune on training examples and select a checkpoint on validation data.
    best_score, best_epoch, history = fine_tune(
        model, batch, probabilities, y, weights, train, val, target, labels,
        args.epochs, args.learning_rate, model_dir / 'model.safetensors',
    )
    training_seconds = time.perf_counter()-start
    model.load_state_dict(load_file(str(model_dir/'model.safetensors')))
    model.eval()
    encoder.config.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    # 6. Save the selected model and evaluate it once on held-out data.
    # The held-out test set is evaluated only after training/selection is finished.
    metrics = {'schema_version':1,'data_fingerprint':fingerprint(),'labels':labels,
        'source_revision':data['manifest']['revision'], 'base_model':BASE_MODEL,'base_revision':BASE_REVISION,
        'seed':42, 'split_sizes':{k:len(v) for k,v in splits.items()},
        'raw_rows':data['manifest']['raw_rows'],'unique_rows':len(rows),
        'training':{'epochs_requested':args.epochs,'best_epoch':best_epoch,'learning_rate':args.learning_rate,
                    'head_learning_rate':1e-3,'batch_size':16,'max_length':MAX_LENGTH,
                    'encoder_fine_tuned':True,'seconds':training_seconds,'history':history},
        'data_provenance':data['manifest'].get('provenance_counts', {}),
        'split_policy':data['manifest'].get('split_policy', 'Stratified 60/20/20'),
        'unevaluated_labels':sorted(set(labels) - set(target[test])),
        'limitations':'Held-out template examples are not clinical validation. Illustrative subsets are not hospital cases. New labels lack independent evaluation.',
        'optimization':{'baseline_best_C':baseline_c,
                        'baseline_validation_accuracy':baseline_val,
                        'selection':'Fixed legacy validation; equal total weight per label and per training family',
                        'frozen_head_best_C':head_c,
                        'frozen_validation_accuracy':frozen_val,
                        'transformer_validation_accuracy':best_score[0]},
        'models':{'ml':evaluate(target[test],baseline.predict_proba([texts[i] for i in test]),labels),
                  'frozen':evaluate(target[test],head.predict_proba(embeddings[test]),labels),
                  'transformer':evaluate(target[test],probabilities(test),labels)}}
    metrics['models']['ml']['training_seconds'] = baseline_seconds
    for key in ['ml','transformer']:
        start = time.perf_counter()
        if key == 'ml': baseline.predict_proba([texts[test[0]]])
        else: probabilities(test[:1])
        metrics['models'][key]['inference_ms'] = (time.perf_counter()-start)*1000
    (out/'metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
    print('TEST RESULTS', {k:v['accuracy'] for k,v in metrics['models'].items()},flush=True)

if __name__ == '__main__':
    main()
