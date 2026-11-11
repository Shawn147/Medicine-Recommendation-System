"""MiniLM encoder, masked mean pooling, and a linear condition head."""
from torch import nn
from torch.nn import functional as F


class ConditionTransformer(nn.Module):
    def __init__(self, encoder, class_count):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(encoder.config.hidden_size, class_count)

    def embed(self, inputs):
        hidden = self.encoder(**inputs).last_hidden_state
        mask = inputs['attention_mask'].unsqueeze(-1)
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
        return F.normalize(pooled, p=2, dim=1)

    def forward(self, inputs):
        return self.head(self.embed(inputs))
