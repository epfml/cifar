```bash
$HOME/conda/envs/pytorch-py3.6/bin/python run.py \
    --arch wideresnet28 --wideresnet_widen_factor 10 \
    --avg_model True --experiment demo --debug True \
    --data cifar10 --pin_memory True \
    --batch_size 128 --base_batch_size 128 --num_workers 2 --eval_freq 1 \
    --num_epochs 250 --partition_data True --reshuffle_per_epoch True --stop_criteria epoch \
    --on_cuda True --blocks 1 --world 0  \
    --lr 0.1 --lr_scaleup True --lr_warmup True --lr_schedule_scheme custom_multistep --lr_change_epochs 125,188 \
    --weight_decay 5e-4 --use_nesterov True --momentum_factor 0.9 \
    --is_kube False --hostfile local_hostfile \
    --python_path $HOME/conda/envs/pytorch-py3.6/bin/python --mpi_path $HOME/.openmpi/ 
```