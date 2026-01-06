import torch
import torch.nn.functional as F
import numpy as np

def compute_angles(pred_doa, gt_doa):
    """
    Compute the angles between predicted and ground truth DOA.
    """
    cos_sim = F.cosine_similarity(pred_doa, gt_doa, dim=-1)
    angle = torch.acos(cos_sim) * 180 / np.pi
    return angle


def _doa_metrics_permutation_invariant_2events(pred_doa: torch.Tensor, gt_doa: torch.Tensor):
    """Permutation-invariant DOA metrics for 2-event case.

    Returns:
        loss_doa: scalar tensor (1 - mean best cosine similarity)
        le_deg: scalar tensor (mean best localization error in degrees)
    """
    # pred/gt: (N,2,3)
    p0, p1 = pred_doa[:, 0, :], pred_doa[:, 1, :]
    g0, g1 = gt_doa[:, 0, :], gt_doa[:, 1, :]

    # cosine similarities (N,)
    c00 = F.cosine_similarity(p0, g0, dim=-1)
    c11 = F.cosine_similarity(p1, g1, dim=-1)
    c01 = F.cosine_similarity(p0, g1, dim=-1)
    c10 = F.cosine_similarity(p1, g0, dim=-1)

    # average over events, then pick better assignment per sample
    cos_no_swap = (c00 + c11) / 2
    cos_swap = (c01 + c10) / 2
    best_cos = torch.maximum(cos_no_swap, cos_swap)
    loss_doa = 1 - best_cos.mean()

    # angles in degrees (N,)
    a00 = torch.acos(torch.clamp(c00, -1.0, 1.0)) * 180 / np.pi
    a11 = torch.acos(torch.clamp(c11, -1.0, 1.0)) * 180 / np.pi
    a01 = torch.acos(torch.clamp(c01, -1.0, 1.0)) * 180 / np.pi
    a10 = torch.acos(torch.clamp(c10, -1.0, 1.0)) * 180 / np.pi

    le_no_swap = (a00 + a11) / 2
    le_swap = (a01 + a10) / 2
    best_le = torch.minimum(le_no_swap, le_swap)
    le_deg = best_le.mean()
    return loss_doa, le_deg


def evaluate(logging, sys_output, logit_scale, dataset_name, task='retrieval', new_idx=None):
    
    if task == 'retrieval':
        audio_features = sys_output['all_audio_features'] # [n_samples, n_dim]
        text_features = sys_output['all_text_features'] # [n_samples (* 5), n_dim]
        logits_per_audio = (logit_scale * audio_features @ text_features.t())
        logits_per_text = logits_per_audio.t()

        logging.info(f"{dataset_name}, logits_per_audio shape: {logits_per_audio.shape}, "
                    f"logits_per_text shape: {logits_per_text.shape}")
        metrics = evaluate_retrieval(logits_per_audio, logits_per_text)
    elif task == 'spatial_retrieval':

        audio_feature_comb = sys_output['all_audio_features'].float() # [n_samples, n_dim]
        text_feature_comb = sys_output['all_text_features'].float() # [n_samples (* 5), n_dim]
        if new_idx: 
            num_caps = text_feature_comb.shape[0] // len(new_idx)
            embed_dim = text_feature_comb.shape[-1]
            text_feature_comb = text_feature_comb.reshape(-1, num_caps, embed_dim)[new_idx]
            text_feature_comb = text_feature_comb.reshape(-1, embed_dim)
        # audio_feature_sed = sys_output['sed_audio_features'] # [n_samples, n_dim]
        # audio_feature_doa = sys_output['doa_audio_features'] # [n_samples, n_dim]
        # text_feature_sed = sys_output['sed_text_features'] # [n_samples (* 5), n_dim]
        
        logits_per_audio_comb = (logit_scale * audio_feature_comb @ text_feature_comb.t())
        logits_per_text_comb = logits_per_audio_comb.t()
        # logits_per_audio_sed = (logit_scale * audio_feature_sed @ text_feature_sed.t())
        # logits_per_text_sed = logits_per_audio_sed.t()
        logging.info(f"{dataset_name}, logits_per_audio_comb shape: {logits_per_audio_comb.shape}, "
                    f"logits_per_text_comb shape: {logits_per_text_comb.shape}")
        metrics = evaluate_retrieval(logits_per_audio_comb, logits_per_text_comb)

        pred_doa = sys_output['pred_doa'].float()
        gt_doa = sys_output['gt_doa'].float()

        # DOA metrics
        # pred/gt may be (N,3) or (N,n_events,3). For n_events=2, use permutation-invariant matching.
        if pred_doa.dim() == 3 and gt_doa.dim() == 2:
            gt_doa = gt_doa.unsqueeze(1).expand(-1, pred_doa.size(1), -1)
        elif pred_doa.dim() == 2 and gt_doa.dim() == 3:
            pred_doa = pred_doa.unsqueeze(1).expand_as(gt_doa)

        if pred_doa.dim() == 3 and gt_doa.dim() == 3 and pred_doa.size(1) == 2 and gt_doa.size(1) == 2:
            loss_doa_t, le_t = _doa_metrics_permutation_invariant_2events(pred_doa, gt_doa)
            loss_doa = float(loss_doa_t.item())
            localization_error = float(le_t.item())
        else:
            # correct dim for cosine similarity: last dim is xyz
            loss_doa = 1 - F.cosine_similarity(pred_doa, gt_doa, dim=-1).mean().item()
            localization_error = compute_angles(pred_doa, gt_doa).mean().item()
        metrics['DOA'] = {'LE': localization_error, 'loss_doa': loss_doa}
        
    return metrics


def evaluate_retrieval(logits_per_audio, logits_per_text):
    """
    num_caps: number of captions per audio clip
    Evaluate the retrieval performance for Clotho and AudioCaps dataset (each audio clip contains num_caps captions).
    1. for text-to-audio retrieval, do num_caps times and average the results
    2. for R@1, R@5, R@10 in audio-to-text retrieval, take the best rank among num_caps text
    3. for map@10 in audio-to-text retrieval:
        3.1: sort the rank of num_caps text
        3.2: exclude the rank >=10 (0-index)
        3.3: compute the map regarding the remaining ranks: np.mean(np.arange(1, len(ranks)+1) / ranks).
        (3.3) That is, take the top ranks of num_caps text that is < 10, and assign the descending number as ground truth.
        (3.3) E.g.: the ground truth of first rank of the num_caps text should be 1, the second rank should be 2, etc.
    """
    metrics = {'stat': {}, 'text_to_audio': {}, 'audio_to_text': {}}
    num_samples = logits_per_audio.shape[0]
    num_caps = logits_per_text.shape[0] // num_samples
    metrics['stat']['num_samples'] = num_samples

    labels = torch.arange(num_samples).long().to(logits_per_audio.device)
    audio_to_text_loss = [
        F.cross_entropy(
            logits_per_audio.reshape(num_samples, num_samples, num_caps)[:, :, d], labels).mean().item() 
            for d in range(num_caps)
    ]
    text_to_audio_loss = [
        F.cross_entropy(
            logits_per_text.reshape(num_samples, num_caps, num_samples)[:, d, :], labels).mean().item() 
            for d in range(num_caps)
    ]
    total_loss = (
            np.mean(audio_to_text_loss) + np.mean(text_to_audio_loss)
    ) / 2
    metrics['stat']["loss_logit"] = total_loss.item()

    # text to audio: do num_caps times
    pred_text = []
    for d in range(num_caps):
        logit = logits_per_text.reshape(num_samples, num_caps, num_samples)[:, d, :]
        ground_truth = torch.arange(len(logit)).view(-1, 1).to(logit.device)
        ranking = torch.argsort(logit, descending=True)  # [num_samples, num_samples]
        preds = torch.where(ranking == ground_truth)[1]
        pred_text.append(preds.cpu().numpy())
    pred_text_concat = np.concatenate(pred_text, axis=0)  # [num_caps * num_samples]
    # metrics['text_to_audio']["mean_rank"] = pred_text_concat.mean() + 1
    # metrics['text_to_audio']["median_rank"] = np.floor(np.median(pred_text_concat)) + 1
    for k in [1, 5, 10]:
        metrics['text_to_audio'][f"R@{k}"] = np.mean(pred_text_concat < k)
    # map@10
    metrics['text_to_audio']["mAP@10"] = np.mean(np.where(pred_text_concat < 10, 1 / (pred_text_concat + 1), 0.0))

    # audio to text: take the best result
    # for audio to text map 10, sort and assign descending ground truth.
    # see https://github.com/XinhaoMei/audio-text_retrieval/blob/main/tools/utils.py#L103
    # map@10
    map_all = []
    pred_audio_all = []
    for d in range(num_samples):
        # logits_per_audio: [num_samples, num_samples * num_caps]
        logit_single = logits_per_audio[d, :]  # [num_caps * num_samples]
        # Ground-truth index: [d*num_caps, d*num_caps+1, ..., d*num_caps+num_caps-1]
        ranking = torch.argsort(logit_single, descending=True)  # [num_caps*num_samples]
        # ranking: the index of first match, second match, ...
        ground_truth = torch.arange(d * num_caps, d * num_caps + num_caps)[None].to(logit_single.device) # [1, num_caps]
        all_pred = torch.where(torch.stack([ranking] * num_caps) == ground_truth.view(-1, 1))[1] # [num_caps]
        min_pred = torch.min(all_pred)
        pred_audio_all.append(min_pred.cpu().numpy())
        all_pred_filter = all_pred[all_pred < 10].cpu().numpy()
        # /num_caps because we have num_caps text, so it means for the text rank >=10 we count as 0.
        map_single = np.sum((np.arange(1, len(all_pred_filter) + 1) / (all_pred_filter + 1))) / num_caps
        map_all.append(map_single)
    for k in [1, 5, 10]:
        metrics['audio_to_text'][f"R@{k}"] = np.mean(np.array(pred_audio_all) < k)
    metrics['audio_to_text']["mAP@10"] = np.mean(map_all)
    
    return metrics