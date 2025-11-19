CUDA_VISIBLE_DEVICES=6 python src/test.py \
    experiment=sCLAP_dual \
    task_name=SUBMIT-NEW \
    experiment_name=sCLAP_dual_1-ColRIR_test \
    data=default-spatial-all-test \
    ckpt_path=/home/hjb/workspace/Spatial-CLAP/logs/SUBMIT-NEW/train/sCLAP_dual_1/checkpoints/last.ckpt

CUDA_VISIBLE_DEVICES=6 python src/test.py \
    experiment=sCLAP_dual \
    task_name=SUBMIT-NEW \
    experiment_name=sCLAP_dual_1+-ColRIR_test \
    data=default-spatial-all-test \
    ckpt_path=/home/hjb/workspace/Spatial-CLAP/logs/SUBMIT-NEW/train/sCLAP_dual_1+/checkpoints/last.ckpt