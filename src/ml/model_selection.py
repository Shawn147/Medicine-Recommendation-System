"""Compare classifier regularization settings on validation examples."""
from sklearn.metrics import accuracy_score


def select_model(factory, training_x, validation_x, train_y, validation_y, sample_weights, pipeline=False):
    """Select regularization on validation data; the test set is never inspected."""
    candidates = []
    for regularization in [1., 10., 100.]:
        candidate = factory(regularization)
        weight_parameter = 'clf__sample_weight' if pipeline else 'sample_weight'
        candidate.fit(training_x, train_y, **{weight_parameter: sample_weights})
        score = float(accuracy_score(validation_y, candidate.predict(validation_x)))
        candidates.append((score, regularization, candidate))
    return max(candidates, key=lambda item: item[0])
