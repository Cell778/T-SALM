import torch
from torch.nn import functional as F


class sCLAPLoss:

    def __init__(self, mlp_loss=False, cache_labels=True, loss_weights=[1.0]):
        self.weights = loss_weights
        self.mlp_loss = mlp_loss
        self.cache_labels = cache_labels
        self.prev_num_logits = 0
        self.labels = {}
    
    def __call__(self, audio_features, text_features, logit_scale, doa, epoch_it=0):
        if epoch_it < 3: weights = [self.weights[0], 0.0]
        else: weights = self.weights
        # pred_doa, gt_doa, cls_doa = doa
        pred_doa, gt_doa = doa
        # Support multi-event DOA shapes: pred_doa (B, n_events, 3),
        # gt_doa can be (B, n_events, 3) or legacy (B, 3).
        if pred_doa.dim() == 3 and gt_doa.dim() == 2:
            gt_doa = gt_doa.unsqueeze(1).expand(-1, pred_doa.size(1), -1)
        elif pred_doa.dim() == 2 and gt_doa.dim() == 3:
            pred_doa = pred_doa.unsqueeze(1).expand_as(gt_doa)
        # compute cosine similarity over last dim, average over events and batch
        cos_sim = F.cosine_similarity(pred_doa, gt_doa, dim=-1)
        loss_doa = (1 - cos_sim).mean()
        
        device = audio_features[0].device
        audio_feature_comb, audio_feature_sed, audio_feature_doa = audio_features
        # text_feature_comb, text_feature_sed, text_feature_doa = text_features
        text_feature_comb, text_feature_sed = text_features
        
        if self.mlp_loss: raise NotImplementedError

        logits_per_audio_comb = logit_scale * audio_feature_comb @ text_feature_comb.T # (N, N)
        logits_per_text_comb = logit_scale * text_feature_comb @ audio_feature_comb.T # (N, N)
        logits_per_audio_sed = logit_scale * audio_feature_sed @ text_feature_sed.T # (N, N)
        logits_per_text_sed = logit_scale * text_feature_sed @ audio_feature_sed.T # (N, N)
        # logits_per_audio_doa = logit_scale * audio_feature_doa @ text_feature_doa.T # (N, 8)

        # calculated ground-truth and cache if enabled
        num_logits = logits_per_audio_comb.shape[0]
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else: labels = self.labels[device]

        loss_logit_spatial_semantic = (
            F.cross_entropy(logits_per_audio_comb, labels) + 
            F.cross_entropy(logits_per_text_comb, labels)
            ) / 2
        
        loss_logit_semantic = (
            F.cross_entropy(logits_per_audio_sed, labels) + 
            F.cross_entropy(logits_per_text_sed, labels)
            ) / 2
        
        # loss_logit_doa = F.cross_entropy(logits_per_audio_doa, cls_doa)

        return {
            'loss_logit_semantic': loss_logit_semantic,
            'loss_logit_spatial_semantic': loss_logit_spatial_semantic,
            # 'loss_logit_doa': loss_logit_doa,
            'loss_doa': loss_doa,
            'total_loss': (1 - weights[1]) * loss_logit_spatial_semantic 
                + weights[1] * loss_logit_semantic + weights[0] * loss_doa

        }
