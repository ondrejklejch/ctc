#!/bin/bash

export LD_LIBRARY_PATH="/usr/local/cuda/lib64/:$LD_LIBRARY_PATH" 
export PATH="/usr/local/cuda-7.5/bin/:$PATH"
export THEANO_FLAGS="floatX=float32,device=gpu,nvcc.fastmath=True,compile_dir=./compile_dir"
export CUDA_VISIBLE_DEVICES=0 

python run.py
