#!/bin/bash

# Run training for different grid sizes
echo "Starting training for 3x3 grid..."
julia --project=. ude_combined.jl --grid-size 3 --data-path data/n_3.h5 --batch-size 8 --adam-iters 500 --lbfgs-iters 250 --nn-size 16 2>&1 | tee out_3.log

echo "Starting training for 4x4 grid..."
julia --project=. ude_combined.jl --grid-size 4 --data-path data/n_4.h5 --batch-size 8 --adam-iters 500 --lbfgs-iters 250 --nn-size 32 2>&1 | tee out_4.log

echo "Starting training for 5x5 grid..."
julia --project=. ude_combined.jl --grid-size 5 --data-path data/n_5.h5 --batch-size 8 --adam-iters 500 --lbfgs-iters 250 --nn-size 64 2>&1 | tee out_5.log

echo "Starting training for 6x6 grid..."
julia --project=. ude_combined.jl --grid-size 6 --data-path data/n_6.h5 --batch-size 8 --adam-iters 500 --lbfgs-iters 250 --nn-size 64 2>&1 | tee out_6.log

echo "Starting training for 7x7 grid..."
julia --project=. ude_combined.jl --grid-size 7 --data-path data/n_7.h5 --batch-size 8 --adam-iters 500 --lbfgs-iters 250 --nn-size 64 2>&1 | tee out_7.log

echo "Starting training for 8x8 grid..."
julia --project=. ude_combined.jl --grid-size 8 --data-path data/n_8.h5 --batch-size 8 --adam-iters 500 --lbfgs-iters 250 --nn-size 128 2>&1 | tee out_8.log

echo "Starting training for 9x9 grid..."
julia --project=. ude_combined.jl --grid-size 9 --data-path data/n_9.h5 --batch-size 8 --adam-iters 500 --lbfgs-iters 250 --nn-size 128 2>&1 | tee out_9.log

echo "Starting training for 10x10 grid..."
julia --project=. ude_combined.jl --grid-size 10 --data-path data/n_10.h5 --batch-size 8 --adam-iters 500 --lbfgs-iters 250 --nn-size 256 2>&1 | tee out_10.log

echo "Starting training for 11x11 grid..."
julia --project=. ude_combined.jl --grid-size 11 --data-path data/n_11.h5 --batch-size 8 --adam-iters 500 --lbfgs-iters 250 --nn-size 256 2>&1 | tee out_11.log
