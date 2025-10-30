# No-Three-in-Line Problem via Universal Differential Equations

## Problem Statement
Find the maximum number of points that can be placed on an n×n grid such that no three points lie on a straight line.

## SciML Approach
This implementation uses:
- **Energy-based formulation**: E(x) = Σ α·x_i1·x_i2·x_i3 over collinear triplets
- **Neural network approximation**: dx/dt = NN(x; θ) replacing ∂E/∂x
- **Constraint-aware decoding**: Guarantees feasible solutions

## Key Features
- **2-stage training** (simplified from original 3-stage):
  - Stage 1: AdamW (1000 iters) with violation^4 penalty
  - Stage 2: LBFGS (extended) with W_const=1000
- **Binary regularizer**: Forces outputs toward 0 or 1
- **TRBDF2 stiff solver**: Handles sharp gradients safely

## Usage
```julia
# Main training and testing pipeline
main_roadmap_pipeline(n=5, mode="ude", load_data=true, train=true, test=true)

# Available modes:
# - "energy": Pure gradient flow baseline
# - "ude": Neural UDE training
# - "all": Compare methods
```

## Outputs
- Trained models: `ude_energy_n[gridsize].jld2`
- Checkpoints: `checkpoint_[stage]_iter[iter]_n[gridsize].jld2`
- Solutions: `no_three_in_line_*.txt`

## Requirements
- Julia 1.9+
- Packages: Lux, DifferentialEquations, Optimization, SciMLSensitivity

## References
1. Original N3L mathematical formulation
2. SciML universal differential equations paper
3. AdaBelief optimizer (NeurIPS 2020)
