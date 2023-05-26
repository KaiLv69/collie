port=$(shuf -i25000-30000 -n1)

# for tensor trainer with inplace sgd
#CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node 1 --master_port="$port" examples/mcqa/train_tensor.py examples/mcqa/hf_args_tensor.yaml

# for zero trainer with inplace sgd

export PATH="$PATH:/remote-home/share/klv/cudas/cuda-12.0/bin"
export CUDA_HOME="/remote-home/share/klv/cudas/cuda-12.0"
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/remote-home/share/klv/cudas/cuda-12.0/lib64"


#WANDB_MODE=disabled \
#deepspeed --master_port "$port" --include localhost:0,1,2,3 inplace_zero_speed.py hf_args_zero.yaml

WANDB_MODE=disabled \
deepspeed --master_port "$port" --include localhost:0,1,2,3,4,5,6,7 zero_speed.py hf_args_zero.yaml