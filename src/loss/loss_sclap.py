import torch
from torch.nn import functional as F


class sCLAPLoss:

    def __init__(self, mlp_loss=False, cache_labels=True, loss_weights=[1.0]):
        self.weights = loss_weights
        self.mlp_loss = mlp_loss
        self.cache_labels = cache_labels
        self.prev_num_logits = 0
        self.labels = {}

    def __call__(self, audio_features, text_features, logit_scale, doa, epoch_it=0, is_triplet=False):
        """Compute training losses.

        Notes
        -----
        - Standard mode: audio/text batch sizes match (N == N), uses CLIP-style
          symmetric cross-entropy on in-batch negatives.
        - Triplet mode (stClotho): audio and text has shape (B*3, D)
          (we keep only positive text as anchors). In this mode we:
            * compute spatial/semantic losses on all 3B samples(3B vs 3B)
            * compute temporal loss as 2-way, only compute similartiy between pos_audio,pos_text(anchor),neg_t_audio
              per anchor text_sed (3B -> 2-way clasification)
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

        #spatial loss weight
        w_spatial = weights_all[3] if len(weights_all) >=4 else w_sem

        #3-way loss weight
        w_ts = weights_all[4] if len(weights_all) >=5 else w_sem

        #semantic loss weight
        w_sem_eff = w_sem

        # temporal weight: start smaller in early epochs, then ramp up to the configured final
        # value (e.g., 0.5). This helps avoid early training instability from hard negatives.
        temporal_ramp_epochs = 4
        if temporal_ramp_epochs <= 1:
            w_temp_eff = w_temp
        else:
            progress = float(epoch_it + 1) / float(temporal_ramp_epochs)
            progress = max(0.0, min(1.0, progress))
            w_temp_eff = w_temp * progress

        spatial_ramp_epochs = 4
        if spatial_ramp_epochs <= 1:
            w_spatial_eff = w_spatial
        else:
            progress = float(epoch_it + 1) / float(spatial_ramp_epochs)
            progress = max(0.0, min(1.0, progress))
            w_spatial_eff = w_spatial * progress

        ts_ramp_epochs = 4
        if ts_ramp_epochs <= 1:
            w_ts_eff = w_ts
        else:
            progress = float(epoch_it + 1) / float(ts_ramp_epochs)
            progress = max(0.0, min(1.0, progress))
            w_ts_eff = w_ts * progress
            
        # pred_doa, gt_doa, cls_doa = doa
        pred_doa, gt_doa = doa
        # Support multi-event DOA shapes: pred_doa (B, n_events, 3),
        # gt_doa can be (B, n_events, 3) or legacy (B, 3).
        if pred_doa.dim() == 3 and gt_doa.dim() == 2:
            gt_doa = gt_doa.unsqueeze(1).expand(-1, pred_doa.size(1), -1)
        elif pred_doa.dim() == 2 and gt_doa.dim() == 3:
            pred_doa = pred_doa.unsqueeze(1).expand_as(gt_doa)

        pred_doa_norm = F.normalize(pred_doa, dim=-1, eps=1e-8)
        gt_doa_norm = F.normalize(gt_doa, dim=-1, eps=1e-8)

        # compute cosine similarity over last dim
        if pred_doa_norm.dim() == 3 and gt_doa_norm.dim() == 3 and pred_doa_norm.size(1) == 2 and gt_doa_norm.size(1) == 2:
            # permutation-invariant for 2-event case (swap events if beneficial)
            p0, p1 = pred_doa_norm[:, 0, :], pred_doa_norm[:, 1, :]
            g0, g1 = gt_doa_norm[:, 0, :], gt_doa_norm[:, 1, :]
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
            cos_sim = F.cosine_similarity(pred_doa_norm, gt_doa_norm, dim=-1)
            loss_doa = (1 - cos_sim).mean()
        
        device = audio_features[0].device
        audio_feature_comb, audio_feature_sed, audio_feature_doa, audio_feature_temporal, audio_feature_triplet = audio_features
        # text_feature_comb, text_feature_sed, text_feature_doa = text_features
        text_feature_comb, text_feature_sed = text_features
        
        if self.mlp_loss: raise NotImplementedError

        triplet_mode = is_triplet 

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
            loss_logit_spatial = torch.zeros((), device=device)
        else:

            b = text_feature_sed.shape[0]
            n = audio_feature_comb.shape[0]
            g = 3
            assert n % g == 0, f"triplet_mode expects N % B == 0, got N={n}, B={b}"
            b = n // g

            # indices of positives in flattened audio: 0, g, 2g, ...
            pos_idx = torch.arange(b, device=device, dtype=torch.long) * g

            labels_pos = torch.arange(b, device=device, dtype=torch.long)

            # spatial loss: treat pos/neg_t/neg_s as ordinary samples (3B vs 3B)
            # assume per-group order: [pos, neg_t, neg_s, ...]
            assert g >= 3, f"spatial hard selection expects group size >=3, got {g}"
            if text_feature_comb.shape[0] != n:
                raise ValueError(
                    f"triplet_mode expects text_feature_comb to match audio candidates (N), got "
                    f"text_comb={text_feature_comb.shape[0]}, audio={n}. "
                    "Ensure training_step flattens text_comb to (B*G, ...)."
                )
            spatial_offsets = torch.tensor([0, 1, 2], device=device, dtype=torch.long)  # pos, neg_t, neg_s
            spatial_idx = (pos_idx[:, None] + spatial_offsets[None, :]).reshape(-1)  # (3B,)
            audio_comb_spatial = audio_feature_comb[spatial_idx]
            text_comb_spatial = text_feature_comb[spatial_idx]
            logits_per_audio_comb = logit_scale * (audio_comb_spatial @ text_comb_spatial.T)  # (3B, 3B)
            logits_per_text_comb = logits_per_audio_comb.T

            num_logits = logits_per_audio_comb.shape[0]
            labels_spatial = torch.arange(num_logits, device=device, dtype=torch.long)
            loss_logit_spatial_semantic = (
                F.cross_entropy(logits_per_audio_comb, labels_spatial)
                + F.cross_entropy(logits_per_text_comb, labels_spatial)
            ) / 2

            # semantic loss: use all 3B samples
            if text_feature_sed.shape[0] != audio_feature_sed.shape[0]:
                text_feature_sed_expanded = text_feature_sed.repeat_interleave(g, dim=0)
            else:
                text_feature_sed_expanded = text_feature_sed
                
            logits_a2t_sed = logit_scale * (audio_feature_sed @ text_feature_sed_expanded.T)  # (3B, 3B)
            logits_t2a_sed = logits_a2t_sed.T

            num_logits_sed = logits_a2t_sed.shape[0]
            labels_sematic = torch.arange(num_logits_sed, device=device, dtype=torch.long)
            loss_logit_semantic = (
                F.cross_entropy(logits_a2t_sed, labels_sematic)
                + F.cross_entropy(logits_t2a_sed, labels_sematic)
            ) / 2

            # temporal hard-negative loss (v2):

            # anchor = positive text_sed (from temporal_spatial_caption / spatialized_caption)
            # candidates = [pos, neg_t] audio_feature_sed
            # assume per-group order: [pos, neg_t, neg_s, ...]
            # cand_idx = (pos_idx[:, None] + temporal_offsets[None, :]).reshape(-1)  # (2B,)

            #text to audio
            neg_t_idx = pos_idx + 1  # negative temporal indices
            audio_temporal_pos = audio_feature_sed[pos_idx]  # extract auido feature comb for pos and neg_t only
            text_temporal_anchor = text_feature_sed[pos_idx]
            audio_temporal_neg_t = audio_feature_sed[neg_t_idx]
            sim_pos = torch.sum(text_temporal_anchor * audio_temporal_pos, dim=-1)
            sim_neg_t = torch.sum(text_temporal_anchor * audio_temporal_neg_t, dim=-1)
            logits_binary = logit_scale * torch.stack([sim_pos, sim_neg_t], dim=1)
            labels_binary = torch.zeros(b, device=device, dtype=torch.long)  
            loss_logit_temporal_t2a = F.cross_entropy(logits_binary, labels_binary)

            #audio to text
            audio_temporal_anchor = audio_temporal_pos
            text_temporal_pos = text_feature_sed[pos_idx]
            text_temporal_neg_t = text_feature_sed[neg_t_idx]
            sim_pos_a2t = torch.sum(audio_temporal_anchor * text_temporal_pos, dim=-1)
            sim_neg_t_a2t = torch.sum(audio_temporal_anchor * text_temporal_neg_t, dim=-1)
            logits_binary_a2t = logit_scale * torch.stack([sim_pos_a2t, sim_neg_t_a2t], dim=1)
            labels_binary_a2t = torch.zeros(b, device=device, dtype=torch.long) 
            loss_logit_temporal_a2t = F.cross_entropy(logits_binary_a2t, labels_binary_a2t)

            loss_logit_temporal = (loss_logit_temporal_t2a + loss_logit_temporal_a2t) / 2

            #Spatial BCE loss with hard negatives
            #text to audio
            neg_s_idx = pos_idx + 2  # negative semantic indices
            audio_spatial_pos = audio_feature_comb[pos_idx]
            text_spatial_anchor = text_feature_comb[pos_idx]
            audio_spatial_neg_s = audio_feature_comb[neg_s_idx]
            sim_pos_s_t2a = torch.sum(text_spatial_anchor * audio_spatial_pos, dim=-1)
            sim_neg_s_t2a = torch.sum(text_spatial_anchor * audio_spatial_neg_s, dim=-1)
            logits_binary_s_t2a = logit_scale * torch.stack([sim_pos_s_t2a, sim_neg_s_t2a], dim=1)
            labels_binary_s_t2a = torch.zeros(b, device=device, dtype=torch.long)
            loss_logit_spatial_s_t2a = F.cross_entropy(logits_binary_s_t2a, labels_binary_s_t2a)

            #audio to text
            audio_spatial_anchor = audio_spatial_pos
            text_spatial_pos = text_feature_comb[pos_idx]
            text_spatial_neg_s = text_feature_comb[neg_s_idx]
            sim_pos_s_a2t = torch.sum(audio_spatial_anchor * text_spatial_pos, dim=-1)
            sim_neg_s_a2t = torch.sum(audio_spatial_anchor * text_spatial_neg_s, dim=-1)
            logits_binary_s_a2t = logit_scale * torch.stack([sim_pos_s_a2t, sim_neg_s_a2t], dim=1)
            labels_binary_s_a2t = torch.zeros(b, device=device, dtype=torch.long)
            loss_logit_spatial_s_a2t = F.cross_entropy(logits_binary_s_a2t, labels_binary_s_a2t)

            loss_logit_spatial = (loss_logit_spatial_s_t2a + loss_logit_spatial_s_a2t) / 2

            #3-way loss (use triplet features)
            #text to audio
            audio_pos = audio_feature_triplet[pos_idx]
            text_anchor = text_feature_comb[pos_idx]
            audio_neg_t = audio_feature_triplet[neg_t_idx]
            audio_neg_s = audio_feature_triplet[neg_s_idx]
            sim_pos = torch.sum(text_anchor * audio_pos, dim=-1)
            sim_neg_t = torch.sum(text_anchor * audio_neg_t, dim=-1)
            sim_neg_s = torch.sum(text_anchor * audio_neg_s, dim=-1)
            logits_3way_t2a = logit_scale * torch.stack([sim_pos, sim_neg_t, sim_neg_s], dim=1)
            logits_3way_t2a = torch.clamp(logits_3way_t2a, -50.0, 50.0)
            labels_3way_t2a = torch.zeros(b, device=device, dtype=torch.long)
            loss_logit_3way_t2a = F.cross_entropy(logits_3way_t2a, labels_3way_t2a)

            #audio to text
            audio_anchor = audio_pos
            text_pos = text_feature_comb[pos_idx]
            text_neg_t = text_feature_comb[neg_t_idx]
            text_neg_s = text_feature_comb[neg_s_idx]
            sim_pos_a2t = torch.sum(audio_anchor * text_pos, dim=-1)
            sim_neg_t_a2t = torch.sum(audio_anchor * text_neg_t, dim=-1)
            sim_neg_s_a2t = torch.sum(audio_anchor * text_neg_s, dim=-1)
            logits_3way_a2t = logit_scale * torch.stack([sim_pos_a2t, sim_neg_t_a2t, sim_neg_s_a2t], dim=1)
            logits_3way_a2t = torch.clamp(logits_3way_a2t, -50.0, 50.0)
            labels_3way_a2t = torch.zeros(b, device=device, dtype=torch.long)
            loss_logit_3way_a2t = F.cross_entropy(logits_3way_a2t, labels_3way_a2t)

            loss_logit_ts = (loss_logit_3way_t2a + loss_logit_3way_a2t) / 2

        # loss_logit_doa = F.cross_entropy(logits_per_audio_doa, cls_doa)

        return {
            'loss_logit_semantic': loss_logit_semantic,
            'loss_logit_spatial_semantic': loss_logit_spatial_semantic,
            "loss_logit_temporal": loss_logit_temporal,
            "loss_logit_spatial": loss_logit_spatial,
            # 'loss_logit_doa': loss_logit_doa,
            'loss_logit_ts': loss_logit_ts,
            'loss_doa': loss_doa,
            'total_loss': 0.5 * loss_logit_spatial_semantic 
                + w_sem_eff * loss_logit_semantic + w_doa * loss_doa + w_spatial_eff * loss_logit_spatial
                + w_temp_eff * loss_logit_temporal
                + w_ts_eff * loss_logit_ts

        }
