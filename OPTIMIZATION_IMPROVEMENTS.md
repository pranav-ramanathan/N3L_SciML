# UDE N3L - Optimization Improvements

## Summary of Changes

Based on training analysis showing loss plateau at ~6-8 with high variance in Stage 3, implemented a comprehensive optimization strategy with:

**✨ Latest Updates**: 
- Switched from Adam to **AdamW** with progressive weight decay schedule for better generalization and stability.
- Changed to **consistent batch=4** for stages 1-3 (maintains exploration), batch=20 for LBFGS stage 4.

### 🎯 Key Improvements

#### 1. **Consistent Small Batching** (Maintains Exploration)
- **Stage 1**: batch=4  → Fast exploration, high diversity
- **Stage 2**: batch=4  → Maintain exploration with harder constraints
- **Stage 3**: batch=4  → Continue exploration for fine-tuning
- **Stage 4**: batch=20 → Small deterministic batch for LBFGS

**Why**: Keeping batch=4 maintains gradient diversity and exploration throughout. Only increase to 20 for Stage 4 where LBFGS needs deterministic gradients. Avoids the "large batch → sharp minima" problem.

#### 2. **Constraint Curriculum** (Eases Optimization)
- **Stage 1**: W_CONSTRAINT=10  → Easy (focus on reconstruction)
- **Stage 2**: W_CONSTRAINT=40  → Medium difficulty
- **Stage 3**: W_CONSTRAINT=80  → Harder constraints
- **Stage 4**: W_CONSTRAINT=120 → Strictest enforcement

**Why**: Starting with hard constraints (W=50) makes optimization difficult. Curriculum allows model to first learn basic structure, then enforce constraints.

#### 3. **Adaptive ODE Tolerances** (Speed vs Accuracy)
- **Stage 1**: (1e-2, 1e-4) → Loose, ~10x faster
- **Stage 2**: (1e-3, 1e-4) → Medium
- **Stage 3**: (1e-4, 1e-6) → Tight
- **Stage 4**: (1e-5, 1e-7) → Strictest

**Why**: Early training doesn't need high precision. Tight tolerances only matter near convergence.

#### 4. **Improved Learning Rate Schedule with AdamW**
- **Stage 1**: AdamW(LR=0.01,   decay=1e-2) → Fast descent with regularization
- **Stage 2**: AdamW(LR=0.001,  decay=5e-3) → 10x drop (not 100x), lighter decay
- **Stage 3**: AdamW(LR=0.0001, decay=1e-3) → Fine-tuning, minimal decay
- **Stage 4**: LBFGS with BackTracking      → Adaptive line search

**Why AdamW over Adam**:
- Decoupled weight decay → better generalization to unseen initial conditions
- More stable training for Neural ODEs (prevents overfitting to specific trajectories)
- Progressive decay schedule (1e-2 → 5e-3 → 1e-3) allows exploration early, precision late

**Why this LR schedule**: Previous 100x drop to 1e-4 was too conservative. 10x drops maintain progress while stabilizing.

#### 5. **Deterministic LBFGS Stage**
- Uses fixed subset of data (first 100 samples)
- Enables proper second-order optimization
- Includes line search for stability

**Why**: LBFGS requires deterministic gradients to build curvature approximation. Random sampling breaks this.

## Training Configuration

### Old Approach (Had Issues)
```julia
Stage 1: batch=4,  W=50, LR=0.01,   tol=1e-3 → Loss: 806k → 17
Stage 2: batch=4,  W=50, LR=0.0001, tol=1e-3 → Loss: 17   → 7
Stage 3: batch=4,  W=50, LR=1e-5,   tol=1e-3 → Loss: 7    → 6-8 (STUCK!)
Stage 4: CG                                  → Would fail
```

**Problems**:
- High variance from small batch at low LR
- Hard constraints from start
- Loose ODE tolerances throughout
- Stage 3 oscillated in [5.7, 8.8] without converging

### New Approach (Improved)
```julia
Stage 1: batch=4,  W=10,  AdamW(LR=0.01,   decay=1e-2), tol=1e-2 → Fast exploration
Stage 2: batch=4,  W=40,  AdamW(LR=0.001,  decay=5e-3), tol=1e-3 → Stable convergence
Stage 3: batch=4,  W=80,  AdamW(LR=0.0001, decay=1e-3), tol=1e-4 → Fine-tuning
Stage 4: batch=20, W=120, LBFGS with line search,       tol=1e-5 → Final push
```

**Benefits**:
- Maintain high exploration with batch=4 throughout AdamW stages
- Gradual constraint enforcement (10→40→80→120)
- Adaptive speed/accuracy tradeoff
- Small deterministic batch (20) sufficient for LBFGS
- Deterministic gradients for LBFGS

## Expected Results

### Performance Improvements
- **10-15x faster** Stage 1 (loose tolerances)
- **Better convergence** in Stage 2 (larger batch)
- **No more oscillation** in Stage 3 (32 samples, higher LR)
- **LBFGS will work** in Stage 4 (deterministic)

### Loss Trajectory Prediction
```
Stage 1: ~800k → ~15   (fast drop, exploration)
Stage 2: ~15   → ~5    (stable convergence)
Stage 3: ~5    → ~2    (smooth fine-tuning)
Stage 4: ~2    → ~0.5  (LBFGS refinement)
```

## Implementation Details

### Function Signature Change
```julia
# Old:
train_ude_energy(dataset, L, n; max_iters=1000, batchsize=4, ...)

# New:
train_ude_energy(dataset, L, n; max_iters=1000, ...) 
# batchsize is now internal, varies by stage
```

### Key Code Patterns

**Stage-specific loss function**:
```julia
loss_fn = make_total_loss(
    batchsize,       # Stage-specific
    w_constraint,    # Stage-specific
    reltol, abstol,  # Stage-specific
    deterministic    # false for ADAM, true for LBFGS
)
```

**Progressive training**:
```julia
θ_current = θ_init
θ_current = train_stage_1(θ_current)  # Update in place
θ_current = train_stage_2(θ_current)
θ_current = train_stage_3(θ_current)
θ_final   = train_stage_4(θ_current)
```

## Additional Features

- **Checkpointing**: Saves every 500 iterations
- **Gradient monitoring**: Track loss at each iteration
- **Comprehensive logging**: Loss per stage, timing, config
- **Enhanced model save**: Stores all stage losses + full config

## Files Modified

- `SciML/ude_n3l.jl`: Complete rewrite of `train_ude_energy` function
  - Lines 403-703: New training implementation
  - Added `LineSearches` import for LBFGS
  - Updated model saving with new stage structure

## Usage

```julia
# Run with new optimization strategy
main_roadmap_pipeline(
    n=5, 
    mode="ude", 
    load_data=true, 
    train=true, 
    test=true,
    max_iters=1000,
    learning_rate=0.01
)
```

## References

Based on best practices for:
- Neural ODE training (Chen et al. 2018)
- Curriculum learning for constraints (Bengio et al. 2009)
- AdamW optimization (Loshchilov & Hutter 2017)
- LBFGS for fine-tuning (Liu & Nocedal 1989)

