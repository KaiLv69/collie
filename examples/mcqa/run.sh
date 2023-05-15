set -x
port=$(shuf -i25000-30000 -n1)

# for tensor trainer with inplace sgd
#WANDB_MODE=disabled \
torchrun --nproc_per_node 8 --master_port="$port" train_tensor.py hf_args_tensor.yaml

# for zero trainer with inplace sgd
#WANDB_MODE=disabled \
#deepspeed --master_port "$port" --include localhost:4,5 train_zero.py hf_args_zero.yaml
