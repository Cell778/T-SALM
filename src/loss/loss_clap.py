import torch
from torch.nn import functional as F


class CLAPLoss:

    def __init__(self, mlp_loss=False, cache_labels=True):
        super(CLAPLoss, self).__init__()
        self.mlp_loss = mlp_loss
        self.cache_labels = cache_labels
        self.prev_num_logits = 0
        self.labels = {}
    
    def __call__(self, audio_features, text_features, logit_scale):
        device = audio_features.device
        
        if self.mlp_loss: raise NotImplementedError
        logits_per_audio = logit_scale * audio_features @ text_features.T
        logits_per_text = logit_scale * text_features @ audio_features.T

        # calculated ground-truth and cache if enabled
        num_logits = logits_per_audio.shape[0]
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else: labels = self.labels[device]

        total_loss = (
            F.cross_entropy(logits_per_audio, labels) + 
            F.cross_entropy(logits_per_text, labels)
            ) / 2

        return {'total_loss': total_loss}
