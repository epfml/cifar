#!/bin/bash
set -e  # exit on error

USER=vogels
LAB=mlo
WANDB_API_KEY=`python -c "import wandb; print(wandb.api.api_key)"`
CODE_BUNDLE=`epfml bundle pack .`

for lr in 0.1 0.01 0.001;
do
    # Generate a unique ID for wandb. This makes sure that automatic restarts continue with the same job.
    RUN_ID=`python -c "import wandb; print(wandb.util.generate_id())"`;

    runai submit \
        --name cifar-lr-sweep-$RUN_ID \
        --environment optimizer_learning_rate=$lr \
        --environment WANDB_PROJECT=demo \
        --environment WANDB_RUN_ID=$RUN_ID \
        --environment WANDB_API_KEY=$WANDB_API_KEY \
        --gpu 1 \
        --image ic-registry.epfl.ch/mlo/pytorch:latest \
        --large-shm \
        --host-ipc \
        --environment DATA_DIR=/${LAB}raw1/$USER/data \
        --environment EPFML_LDAP=$USER \
        --environment EPFML_STORE_S3_ACCESS_KEY="$EPFML_STORE_S3_ACCESS_KEY" \
        --environment EPFML_STORE_S3_SECRET_KEY="$EPFML_STORE_S3_SECRET_KEY" \
        --environment EPFML_STORE_S3_BUCKET="$EPFML_STORE_S3_BUCKET" \
        --pvc runai-$LAB-$USER-${LAB}data1:/${LAB}data1 \
        --pvc runai-$LAB-$USER-${LAB}raw1:/${LAB}raw1 \
        --command -- \
            /entrypoint.sh \
            su $USER -c \
            \"epfml bundle exec $CODE_BUNDLE ./start.sh\";
done
