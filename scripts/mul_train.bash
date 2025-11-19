CUDA_VISIBLE_DEVICES=6 python src/train.py \
    experiment=sCLAP_dual \
    task_name=SUBMIT-NEW \
    experiment_name=sCLAP_dual_1

CUDA_VISIBLE_DEVICES=6 python src/train.py \
    experiment=sCLAP_dual \
    task_name=SUBMIT-NEW \
    experiment_name=sCLAP_dual_2 \
    model.loss_weights=\[1.0,0.0\]

CUDA_VISIBLE_DEVICES=6 python src/train.py \
    experiment=sCLAP_single \
    task_name=SUBMIT-NEW \
    experiment_name=sCLAP_single \
    model.batch_size=80

CUDA_VISIBLE_DEVICES=6 python src/train.py \
    experiment=CLAP \
    task_name=SUBMIT-NEW \
    experiment_name=CLAP