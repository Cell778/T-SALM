CUDA_VISIBLE_DEVICES=6 python src/test.py \
    experiment=sCLAP_dual \
    task_name=SUBMIT-NEW \
    experiment_name=sCLAP_dual_1 \
    ckpt_path=/home/hjb/workspace/Spatial-CLAP/logs/SUBMIT-NEW/train/sCLAP_dual_1/checkpoints/last.ckpt

CUDA_VISIBLE_DEVICES=6 python src/test.py \
    experiment=sCLAP_dual \
    task_name=SUBMIT-NEW \
    experiment_name=sCLAP_dual_2 \
    ckpt_path=/home/hjb/workspace/Spatial-CLAP/logs/SUBMIT-NEW/train/sCLAP_dual_2/checkpoints/last.ckpt

CUDA_VISIBLE_DEVICES=6 python src/test.py \
    experiment=sCLAP_single \
    task_name=SUBMIT-NEW \
    experiment_name=sCLAP_single \
    ckpt_path=/home/hjb/workspace/Spatial-CLAP/logs/SUBMIT-NEW/train/sCLAP_single/checkpoints/last.ckpt

CUDA_VISIBLE_DEVICES=6 python src/test.py \
    experiment=CLAP \
    task_name=SUBMIT-NEW \
    experiment_name=CLAP \
    ckpt_path=/home/hjb/workspace/Spatial-CLAP/logs/SUBMIT-NEW/train/CLAP/checkpoints/last.ckpt