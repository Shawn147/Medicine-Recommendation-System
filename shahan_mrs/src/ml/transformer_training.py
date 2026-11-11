"""Fine-tune MiniLM and save the best validation checkpoint."""
import json

import numpy as np
import torch
from safetensors.torch import save_file
from sklearn.metrics import accuracy_score, log_loss


def fine_tune(model, batch, probabilities, label_ids, weights, train, validation,
              targets, labels, epochs, learning_rate, checkpoint_path):
    """Train on weighted examples and stop after three unimproved epochs."""
    optimizer = torch.optim.AdamW([
        {'params': model.encoder.parameters(), 'lr': learning_rate},
        {'params': model.head.parameters(), 'lr': 1e-3},
    ], weight_decay=0.01)
    best_score, patience, history = None, 0, []
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        shuffled = np.random.permutation(train)
        for offset in range(0, len(shuffled), 16):
            indices = shuffled[offset:offset + 16]
            optimizer.zero_grad()
            logits = model(batch(indices))
            per_row = torch.nn.functional.cross_entropy(logits, label_ids[indices], reduction='none')
            loss = (per_row * torch.from_numpy(weights[indices])).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.detach()) * len(indices)

        validation_probabilities = probabilities(validation)
        predictions = np.array(labels)[validation_probabilities.argmax(1)]
        accuracy = float(accuracy_score(targets[validation], predictions))
        validation_loss = float(log_loss(targets[validation], validation_probabilities, labels=labels))
        epoch_metrics = {
            'epoch': epoch,
            'train_loss': total_loss / len(train),
            'validation_accuracy': accuracy,
            'validation_loss': validation_loss,
        }
        history.append(epoch_metrics)
        print(json.dumps(epoch_metrics), flush=True)

        score = (accuracy, -validation_loss)
        if best_score is None or score > best_score:
            best_score, best_epoch, patience = score, epoch, 0
            checkpoint = {
                name: value.detach().cpu().contiguous()
                for name, value in model.state_dict().items()
            }
            save_file(checkpoint, str(checkpoint_path))
        else:
            patience += 1
        if patience >= 3:
            break

    return best_score, best_epoch, history
