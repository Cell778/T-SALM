# CUDA_VISIBLE_DEVICES=6 python src/test.py experiment=sCLAP_dual \
#     task_name=SUBMIT-NEW \
#     experiment_name=sCLAP_dual_swap \
#     ckpt_path=logs/SUBMIT-NEW/train/sCLAP_dual_1/checkpoints/last.ckpt \
#     mode=valid \
#     edit=true

CUDA_VISIBLE_DEVICES=6 python src/test.py experiment=sCLAP_dual \
    task_name=SUBMIT-NEW \
    experiment_name=sCLAP_dual_modify \
    ckpt_path=logs/SUBMIT-NEW/train/sCLAP_dual_1/checkpoints/last.ckpt \
    mode=valid \
    edit=modify