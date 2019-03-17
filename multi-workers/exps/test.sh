# python main.py \
#     --arch resnet20 --optimizer adam \
#     --avg_model True --experiment Adam_large_batch_training_baseline_without_lr_decay --debug True \
#     --data cifar10 --pin_memory True \
#     --batch_size 128 --base_batch_size 64 --num_workers 2 --eval_freq 1 \
#     --num_epochs 300 --partition_data True --reshuffle_per_epoch True --stop_criteria epoch \
#     --on_cuda False --blocks 1 --world 0  \
#     --lr 0.1 --lr_scaleup False --lr_warmup False --lr_schedule_scheme custom_multistep --lr_change_epochs 150,225 \
#     --weight_decay 1e-4 --use_nesterov True --momentum_factor 0.9

$HOME/conda/envs/pytorch-py3.6/bin/python run.py \
    --arch resnet20 --optimizer adam \
    --avg_model True --experiment Adam_large_batch_training_baseline_without_lr_decay --debug True \
    --data cifar10 --pin_memory True \
    --batch_size 128 --base_batch_size 64 --num_workers 2 --eval_freq 1 \
    --num_epochs 300 --partition_data True --reshuffle_per_epoch True --stop_criteria epoch \
    --on_cuda True --blocks 1 --world 0  \
    --lr 0.1 --lr_scaleup False --lr_warmup False --lr_schedule_scheme custom_multistep --lr_change_epochs 150,225 \
    --weight_decay 1e-4 --use_nesterov True --momentum_factor 0.9 \
    --is_kube False --hostfile local_hostfile \
    --python_path $HOME/conda/envs/pytorch-py3.6/bin/python --mpi_path $HOME/.openmpi/