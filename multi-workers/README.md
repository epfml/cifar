# Getting Started
For the detailed experimental setup of the environment, please refer to the `platform` folder.


## Running the tests
We here describe the general usage of the code. For the detailed usage, please refer to the detailed instruction (template) below.


### Case: single worker
Note the script provided below should support `single worker` case with arbitrary `PyTorch` installed environment. Here we use `ResNet-20` with `CIFAR-10` as an example to explain the meaning of the arguments of the script below.

```bash
python main.py \
    --arch resnet20 --optimizer adam \
    --avg_model True --experiment Adam_large_batch_training_baseline_without_lr_decay --debug True \
    --data cifar10 --pin_memory True \
    --batch_size 128 --base_batch_size 64 --num_workers 2 --eval_freq 1 \
    --num_epochs 300 --partition_data True --reshuffle_per_epoch True --stop_criteria epoch \
    --on_cuda False --blocks 1 --world 0  \
    --lr 0.1 --lr_scaleup False --lr_warmup False --lr_schedule_scheme custom_multistep --lr_change_epochs 150,225 \
    --weight_decay 1e-4 --use_nesterov True --momentum_factor 0.9
```


### Case: multiple workers
We use two notations `world` and `block_size` to describe the topology of the distributed training. For example, by setting `blocks=2,2` and `world=0,1,0,1`, we have two blocks, and each block (with size `2`) has two GPUs with the device id `0,1` and `0,1` respectively. The physical layout of these GPUs is defined through MPI's `hostfile`, where we can have a flexible training topology over the nodes.


#### Example script for different experiments.
Note the scripts provided below are using the environment built by docker. Here we use `ResNet-20` with `CIFAR-10` as an example to explain the meaning of the arguments of the script below.

```bash
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
```

The detailed explanation of some arguments:
* The `lr` and `base_batch_size` determine the learning rate per sample, while the eventual learning rate used for training will be determined by `batch_size`, `lr_scaleup` and the summation of `blocks` (i.e., the number of workers).
* The `lr_warmup` determines if we want to warmup the learning rate or not. If `lr_warmup is True` then the learning rate would gradually increase from `0.1` to the scaled one. Otherwise, it will directly use the scaled one as the initial learning rate.
* The `lr_schedule_scheme` and `lr_change_epochs` define the learning rate schedule, wherein our example is the constant learning rate with a decay factor (i.e., argument `lr_decay`) at predefined epochs.
* The `momentum` decides the momentum scheme used for the distributed training. In this example, we are using Nesterov momentum with factor `0.9`.
