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
        """Compute training losses.

        Notes
        -----
        - Standard mode: audio/text batch sizes match (N == N), uses CLIP-style
          symmetric cross-entropy on in-batch negatives.
        - Triplet mode (stClotho): audio has shape (B*3, D) while text is (B, D)
          (we keep only positive text as anchors). In this mode we:
            * compute spatial/semantic losses on positives only (B vs B)
            * compute temporal loss as hard-negative CE with candidates [pos, neg_t]
              per anchor text_sed (B vs 2B)
        """

        weights_all = list(self.weights)
        # Backward compatible parsing:
        # - [w_doa] -> [w_doa, 0.0]
        # - [w_doa, w_sem]
        # - [w_doa, w_sem, w_temp]
        if len(weights_all) == 1:
            weights_all = [weights_all[0], 0.0]

        w_doa = weights_all[0]
        w_sem = weights_all[1]
        # temporal loss weight: by default tie it to semantic weight for backward compat
        w_temp = weights_all[2] if len(weights_all) >= 3 else w_sem

        # warmup: disable semantic loss for the first 3 epochs, but keep temporal active
        if epoch_it < 3:
            w_sem_eff = 0.0
        else:
            w_sem_eff = w_sem
        # pred_doa, gt_doa, cls_doa = doa
        pred_doa, gt_doa = doa
        # Support multi-event DOA shapes: pred_doa (B, n_events, 3),
        # gt_doa can be (B, n_events, 3) or legacy (B, 3).
        if pred_doa.dim() == 3 and gt_doa.dim() == 2:
            gt_doa = gt_doa.unsqueeze(1).expand(-1, pred_doa.size(1), -1)
        elif pred_doa.dim() == 2 and gt_doa.dim() == 3:
            pred_doa = pred_doa.unsqueeze(1).expand_as(gt_doa)

        # compute cosine similarity over last dim
        if pred_doa.dim() == 3 and gt_doa.dim() == 3 and pred_doa.size(1) == 2 and gt_doa.size(1) == 2:
            # permutation-invariant for 2-event case (swap events if beneficial)
            p0, p1 = pred_doa[:, 0, :], pred_doa[:, 1, :]
            g0, g1 = gt_doa[:, 0, :], gt_doa[:, 1, :]
            c00 = F.cosine_similarity(p0, g0, dim=-1)
            c11 = F.cosine_similarity(p1, g1, dim=-1)
            c01 = F.cosine_similarity(p0, g1, dim=-1)
            c10 = F.cosine_similarity(p1, g0, dim=-1)
            cos_no_swap = (c00 + c11) / 2
            cos_swap = (c01 + c10) / 2
            best_cos = torch.maximum(cos_no_swap, cos_swap)
            loss_doa = (1 - best_cos).mean()
        else:
            # average over events and batch
            cos_sim = F.cosine_similarity(pred_doa, gt_doa, dim=-1)
            loss_doa = (1 - cos_sim).mean()
        
        device = audio_features[0].device
        audio_feature_comb, audio_feature_sed, audio_feature_doa = audio_features
        # text_feature_comb, text_feature_sed, text_feature_doa = text_features
        text_feature_comb, text_feature_sed = text_features
        
        if self.mlp_loss: raise NotImplementedError

        triplet_mode = (
            audio_feature_comb.shape[0] != text_feature_comb.shape[0]
            or audio_feature_sed.shape[0] != text_feature_sed.shape[0]
        )

        if not triplet_mode:
            logits_per_audio_comb = logit_scale * audio_feature_comb @ text_feature_comb.T  # (N, N)
            logits_per_text_comb = logits_per_audio_comb.T
            logits_per_audio_sed = logit_scale * audio_feature_sed @ text_feature_sed.T  # (N, N)
            logits_per_text_sed = logits_per_audio_sed.T

            # calculated ground-truth and cache if enabled
            num_logits = logits_per_audio_comb.shape[0]
            if self.prev_num_logits != num_logits or device not in self.labels:
                labels = torch.arange(num_logits, device=device, dtype=torch.long)
                if self.cache_labels:
                    self.labels[device] = labels
                    self.prev_num_logits = num_logits
            else:
                labels = self.labels[device]

            loss_logit_spatial_semantic = (
                F.cross_entropy(logits_per_audio_comb, labels)
                + F.cross_entropy(logits_per_text_comb, labels)
            ) / 2

            loss_logit_semantic = (
                F.cross_entropy(logits_per_audio_sed, labels)
                + F.cross_entropy(logits_per_text_sed, labels)
            ) / 2

            # temporal loss (v2) is only meaningful in triplet mode
            loss_logit_temporal = torch.zeros((), device=device)
        else:
            # Triplet mode: audio is (B*G, D)
            # - text_sed is (B, D): positive anchors for temporal hard-negative mining
            # - text_comb is expected to be (B*G, D): ordinary samples for spatial contrastive training
            b = text_feature_sed.shape[0]
            n = audio_feature_comb.shape[0]
            assert n % b == 0, f"triplet_mode expects N % B == 0, got N={n}, B={b}"
            g = n // b
            assert g >= 2, f"triplet_mode expects group size >=2, got {g}"

            # indices of positives in flattened audio: 0, g, 2g, ...
            pos_idx = torch.arange(b, device=device, dtype=torch.long) * g

            labels_pos = torch.arange(b, device=device, dtype=torch.long)

            # spatial loss: use pos + spatial-negative as ordinary samples (2B vs 2B)
            # assume per-group order: [pos, neg_t, neg_s, ...]
            assert g >= 3, f"spatial hard selection expects group size >=3, got {g}"
            if text_feature_comb.shape[0] != n:
                raise ValueError(
                    f"triplet_mode expects text_feature_comb to match audio candidates (N), got "
                    f"text_comb={text_feature_comb.shape[0]}, audio={n}. "
                    "Ensure training_step flattens text_comb to (B*G, ...)."
                )
            spatial_offsets = torch.tensor([0, 2], device=device, dtype=torch.long)  # pos, neg_s
            spatial_idx = (pos_idx[:, None] + spatial_offsets[None, :]).reshape(-1)  # (2B,)
            audio_comb_spatial = audio_feature_comb[spatial_idx]
            text_comb_spatial = text_feature_comb[spatial_idx]
            logits_per_audio_comb = logit_scale * (audio_comb_spatial @ text_comb_spatial.T)  # (2B, 2B)
            logits_per_text_comb = logits_per_audio_comb.T

            num_logits = logits_per_audio_comb.shape[0]
            labels_spatial = torch.arange(num_logits, device=device, dtype=torch.long)
            loss_logit_spatial_semantic = (
                F.cross_entropy(logits_per_audio_comb, labels_spatial)
                + F.cross_entropy(logits_per_text_comb, labels_spatial)
            ) / 2

            # semantic loss: keep positives only (B vs B)
            audio_sed_pos = audio_feature_sed[pos_idx]
            logits_a2t_sed = logit_scale * (audio_sed_pos @ text_feature_sed.T)  # (B, B)
            logits_t2a_sed = logits_a2t_sed.T
            loss_logit_semantic = (
                F.cross_entropy(logits_a2t_sed, labels_pos)
                + F.cross_entropy(logits_t2a_sed, labels_pos)
            ) / 2

            # temporal hard-negative loss (v2): anchor = positive text_sed, candidates = [pos, neg_t]
            # assume per-group order: [pos, neg_t, neg_s, ...]
            temporal_offsets = torch.tensor([0, 1], device=device, dtype=torch.long)  # pos, neg_t
            cand_idx = (pos_idx[:, None] + temporal_offsets[None, :]).reshape(-1)  # (2B,)
            audio_sed_temporal_cands = audio_feature_sed[cand_idx]  # (2B, D)
            logits_t2a_temp = logit_scale * (text_feature_sed @ audio_sed_temporal_cands.T)  # (B, 2B)
            labels_temp = torch.arange(b, device=device, dtype=torch.long) * 2
            loss_logit_temporal = F.cross_entropy(logits_t2a_temp, labels_temp)
        # loss_logit_doa = F.cross_entropy(logits_per_audio_doa, cls_doa)

        return {
            'loss_logit_semantic': loss_logit_semantic,
            'loss_logit_spatial_semantic': loss_logit_spatial_semantic,
            "loss_logit_temporal": loss_logit_temporal,
            # 'loss_logit_doa': loss_logit_doa,
            'loss_doa': loss_doa,
            'total_loss': (1 - w_sem_eff) * loss_logit_spatial_semantic 
                + w_sem_eff * loss_logit_semantic + w_doa * loss_doa
                + w_temp * loss_logit_temporal

        }
