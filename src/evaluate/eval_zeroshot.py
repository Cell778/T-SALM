import numpy as np
import torch


def evaluate_zero_shot(logging, sys_output, dataset_name, audio_key='all_audio_features', 
                       text_key='all_text_features', gt_key='ground_truth'):
    """
    Evaluate the zero-shot classification performance.
    """
    audio_features = torch.cat(sys_output[audio_key], dim=0) # [n_samples, n_dim]
    text_features = sys_output[text_key] # [n_classes, n_dim]
    ground_truth = torch.cat(sys_output[gt_key], dim=0) # [n_samples]

    logging.info(f"{dataset_name}, logits_per_audio shape: {audio_features.shape}, "
                 f"logits_per_text shape: {text_features.shape}")
    
    ranking = torch.argsort(audio_features @ text_features.t(), descending=True)  # [n_samples, n_classes]
    preds = torch.where(ranking == ground_truth.view(-1, 1))[1]
    preds = preds.cpu().numpy()
    print(preds)

    metrics = {}
    metrics[f"mean_rank"] = preds.mean() + 1
    metrics[f"median_rank"] = np.floor(np.median(preds)) + 1
    for k in [1, 5, 10]:
        metrics[f"R@{k}"] = np.mean(preds < k)
    # map@10
    metrics[f"mAP@10"] = np.mean(np.where(preds < 10, 1 / (preds + 1), 0.0))

    metrics = {'zero-shot': metrics}

    return metrics
