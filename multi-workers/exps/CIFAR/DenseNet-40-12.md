```bash
$HOME/conda/envs/pytorch-py3.6/bin/python run.py \
    --arch densenet40 --densenet_growth_rate 12 --densenet_bc_mode False --densenet_compression 1.0 \
    --avg_model True --experiment demo --debug True \
    --data cifar10 --pin_memory True \
    --batch_size 128 --base_batch_size 64 --num_workers 2 --eval_freq 1 \
    --num_epochs 300 --partition_data True --reshuffle_per_epoch True --stop_criteria epoch \
    --on_cuda True --blocks 1 --world 0  \
    --lr 0.1 --lr_scaleup True --lr_warmup True --lr_schedule_scheme custom_multistep --lr_change_epochs 150,225 \
    --weight_decay 1e-4 --use_nesterov True --momentum_factor 0.9 \
    --is_kube False --hostfile local_hostfile \
    --python_path $HOME/conda/envs/pytorch-py3.6/bin/python --mpi_path $HOME/.openmpi/
```