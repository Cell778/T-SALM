CUDA_VISIBLE_DEVICES=6 python src/test.py \
    mode=test \
    task=zero_shot_doa \
    task_name=SUBMIT-NEW \
    ckpt_path=/home/hjb/workspace/Spatial-CLAP/logs/SUBMIT-NEW/train/sCLAP_dual_1/checkpoints/last.ckpt \
    experiment_name=zsl-doa-sAudioCaps-doa_feat \
    data=default-spatial-sAudioCaps \
    doa_feature_type=2

CUDA_VISIBLE_DEVICES=6 python src/test.py \
    mode=test \
    task=zero_shot_doa \
    task_name=SUBMIT-NEW \
    ckpt_path=/home/hjb/workspace/Spatial-CLAP/logs/SUBMIT-NEW/train/sCLAP_dual_1/checkpoints/last.ckpt \
    experiment_name=zsl-doa-sAudioCaps-sed_feat \
    data=default-spatial-sAudioCaps \
    doa_feature_type=1

CUDA_VISIBLE_DEVICES=6 python src/test.py \
    mode=test \
    task=zero_shot_doa \
    task_name=SUBMIT-NEW \
    ckpt_path=/home/hjb/workspace/Spatial-CLAP/logs/SUBMIT-NEW/train/sCLAP_dual_1/checkpoints/last.ckpt \
    experiment_name=zsl-doa-sAudioCaps-comb_feat \
    data=default-spatial-sAudioCaps \
    doa_feature_type=0

CUDA_VISIBLE_DEVICES=6 python src/test.py \
    mode=test \
    task=zero_shot_doa \
    task_name=SUBMIT-NEW \
    ckpt_path=/home/hjb/workspace/Spatial-CLAP/logs/SUBMIT-NEW/train/sCLAP_dual_1/checkpoints/last.ckpt \
    experiment_name=zsl-doa-sClotho-doa_feat \
    data=default-spatial-sClotho \
    doa_feature_type=2

CUDA_VISIBLE_DEVICES=6 python src/test.py \
    mode=test \
    task=zero_shot_doa \
    task_name=SUBMIT-NEW \
    ckpt_path=/home/hjb/workspace/Spatial-CLAP/logs/SUBMIT-NEW/train/sCLAP_dual_1/checkpoints/last.ckpt \
    experiment_name=zsl-doa-sClotho-sed_feat \
    data=default-spatial-sClotho \
    doa_feature_type=1

CUDA_VISIBLE_DEVICES=6 python src/test.py \
    mode=test \
    task=zero_shot_doa \
    task_name=SUBMIT-NEW \
    ckpt_path=/home/hjb/workspace/Spatial-CLAP/logs/SUBMIT-NEW/train/sCLAP_dual_1/checkpoints/last.ckpt \
    experiment_name=zsl-doa-sClotho-comb_feat \
    data=default-spatial-sClotho \
    doa_feature_type=0