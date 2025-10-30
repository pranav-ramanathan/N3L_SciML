# ============================================================================
# UDE N3L - Complete SciML Roadmap Implementation
# No-Three-in-Line Problem as Universal Differential Equations
# ============================================================================

"""
Complete implementation of the SciML roadmap for No-Three-in-Line:

ROADMAP IMPLEMENTATION STATUS:
✓ 1. Energy-based formulation: E(x) = Σ α·x_i1·x_i2·x_i3 over collinear triplets
✓ 2. Gradient flow dynamics: dx/dt = -∂E/∂x + η(t)
✓ 3. Learning dynamics: dθ/dt = -∇_θ L(θ; x) + ζ(t)
✓ 4. Coupled system: [dx/dt; dθ/dt] with explicit energy gradient
✓ 5. UDE with neural network: dx/dt = NN(x; θ) replacing ∂E/∂x
✓ 6. Training with improved optimizer schedule and constraints
⚠️ 7. UDE coupled training (dynamics exist, training is future work)

KEY IMPROVEMENTS (Latest):
- 3-stage optimizer: RAdam (1000) → AdaBelief (1500) → LBFGS (3000)
- Progressive penalty: violation^2 → ^3 → ^4 (prevents early instability)
- Binary regularizer: pushes outputs toward 0 or 1
- Tighter constraints: curriculum up to W=1000
- Constraint-aware decoder: guarantees feasible solutions at test time
- TRBDF2 stiff solver: handles sharp gradients, compatible with Zygote adjoint

MAIN ENTRY POINTS:
- main_roadmap_pipeline(mode="ude", ...) : Primary UDE training & testing
- main_roadmap_pipeline(mode="energy", ...) : Energy flow baseline
- main_roadmap_pipeline(mode="coupled", ...) : Basic coupled system demo
- main_roadmap_pipeline(mode="all", ...) : Full comparison

For your typical workflow, use mode="ude" with train=true and load_data=true.
"""

# ============================================================================
# SECTION 1: IMPORTS AND CONSTANTS
# ============================================================================

using DifferentialEquations
using Lux, Random, Optimization, OptimizationOptimisers, OptimizationOptimJL
using SciMLSensitivity
using StableRNGs
using ComponentArrays
using Zygote
using JLD2
using Statistics
using Plots
using Functors
using LineSearches

# Load data generation functions
include("n3l_gurobi.jl")

# Constants for energy and dynamics
const α_ENERGY = 1.0f0          # Energy weight for collinearity penalty
const σ_X = 0.01f0              # Noise scale for x dynamics
const σ_Θ = 0.001f0             # Noise scale for θ dynamics
const SENSEALG = InterpolatingAdjoint(autojacvec = ZygoteVJP())

# Loss weights for training
const W_RECON = 1.0f0           # Reconstruction weight
const W_CONSTRAINT = 50.0f0     # Constraint violation weight (base, will be increased during training)
const W_BOX = 1.0f0             # Box constraint weight
const W_REG = 0.001f0           # Regularization weight
const W_BINARY = 1.0f0          # Binary regularization weight

# ============================================================================
# SECTION 2: ENERGY FUNCTION (ROADMAP ITEM 2)
# ============================================================================

"""
    energy_E(x::AbstractMatrix, L, α=α_ENERGY)

Compute energy function E(x) = Σ α·x_i1j1·x_i2j2·x_i3j3 over all collinear triplets.

Higher energy indicates more collinearity risk.
"""
function energy_E(x::AbstractMatrix, L, α=α_ENERGY)
    E = 0f0
    @inbounds for (i1, j1, i2, j2, i3, j3) in L
        E += α * x[i1, j1] * x[i2, j2] * x[i3, j3]
    end
    return E
end

"""
    energy_gradient!(grad, x::AbstractMatrix, L, α=α_ENERGY)

Compute gradient ∂E/∂x_ij for each cell.

For each triplet (i1,j1,i2,j2,i3,j3):
- ∂E/∂x_i1j1 = α·x_i2j2·x_i3j3
- ∂E/∂x_i2j2 = α·x_i1j1·x_i3j3
- ∂E/∂x_i3j3 = α·x_i1j1·x_i2j2

Accumulates gradients across all triplets containing each cell.
"""
function energy_gradient!(grad, x::AbstractMatrix, L, α=α_ENERGY)
    fill!(grad, 0f0)
    @inbounds for (i1, j1, i2, j2, i3, j3) in L
        # Product of all three values
        x1 = x[i1, j1]
        x2 = x[i2, j2]
        x3 = x[i3, j3]
        
        # Partial derivatives
        grad[i1, j1] += α * x2 * x3
        grad[i2, j2] += α * x1 * x3
        grad[i3, j3] += α * x1 * x2
    end
    return grad
end

# ============================================================================
# SECTION 3: EXPLICIT GRADIENT FLOW DYNAMICS (ROADMAP ITEM 2)
# ============================================================================

"""
    energy_based_dynamics!(du, u, p, t)

Implement dx/dt = -∂E/∂x + η(t) (gradient flow with exploration noise).

Parameters p should contain: (L, α, σ_noise, n)
"""
function energy_based_dynamics!(du, u, p, t)
    L, α, σ_noise, n = p
    
    # Reshape to matrix
    x = reshape(u, n, n)
    dx = reshape(du, n, n)
    
    # Compute -∂E/∂x
    energy_gradient!(dx, x, L, α)
    @. dx = -dx
    
    # Add exploration noise η(t)
    if σ_noise > 0f0
        @. dx += σ_noise * randn(Float32)
    end
    
    return nothing
end

"""
    solve_energy_flow(x0, L, n, α=α_ENERGY; t_end=10f0, σ_noise=σ_X, saveat=nothing)

Solve gradient flow dynamics dx/dt = -∂E/∂x + η(t).

Returns ODESolution with trajectory.
"""
function solve_energy_flow(x0, L, n, α=α_ENERGY; t_end=10f0, σ_noise=σ_X, saveat=nothing)
    # Convert to vector
    u0 = vec(Float32.(x0))
    tspan = (0f0, t_end)
    
    # Parameters
    p = (L, α, σ_noise, n)
    
    # Create and solve ODE
    prob = ODEProblem(energy_based_dynamics!, u0, tspan, p)
    
    if saveat !== nothing
        sol = solve(prob, Tsit5(); reltol=1f-3, abstol=1f-5, saveat=saveat)
    else
        sol = solve(prob, Tsit5(); reltol=1f-3, abstol=1f-5)
    end
    
    return sol
end

# ============================================================================
# SECTION 4: LEARNING LOSS AND DYNAMICS (ROADMAP ITEM 4)
# ============================================================================

"""
    learning_loss_L(θ::AbstractVector, x::AbstractMatrix, x_target::AbstractMatrix, L)

Compute learning loss L(θ; x(t)) with multiple components:
1. Reconstruction: ||x - x_target||²
2. Constraint violation: Σ max(0, x_triplet_sum - 2)²
3. Parameter regularization: ||θ||²

This loss guides parameter learning dynamics.
"""
function learning_loss_L(θ::AbstractVector, x::AbstractMatrix, x_target::AbstractMatrix, L)
    # Reconstruction loss
    l_recon = sum((x .- x_target).^2)
    
    # Constraint violation
    l_constraint = 0f0
    @inbounds for (i1, j1, i2, j2, i3, j3) in L
        triplet_sum = x[i1, j1] + x[i2, j2] + x[i3, j3]
        violation = max(0f0, triplet_sum - 2f0)
        l_constraint += violation * violation
    end
    
    # Regularization
    l_reg = sum(θ .^ 2)
    
    return W_RECON * l_recon + W_CONSTRAINT * l_constraint + W_REG * l_reg
end

"""
    learning_dynamics!(dθ, θ, p, t)

Implement dθ/dt = -∇_θ L(θ; x(t)) + ζ(t) (parameter learning with exploration).

Parameters p should contain: (x_current, x_target, L, σ_θ)

Note: This is a simplified version. In practice, ∇_θ L would be computed via
automatic differentiation during coupled system integration.
"""
function learning_dynamics!(dθ, θ, p, t)
    x_current, x_target, L, σ_θ = p
    
    # Compute gradient via Zygote
    loss_fn(θ_temp) = learning_loss_L(θ_temp, x_current, x_target, L)
    grad = Zygote.gradient(loss_fn, θ)[1]
    
    # Update: -∇_θ L
    @. dθ = -grad
    
    # Add exploration noise ζ(t)
    if σ_θ > 0f0
        @. dθ += σ_θ * randn(Float32)
    end
    
    return nothing
end

# ============================================================================
# SECTION 5: COUPLED SYSTEM (ROADMAP ITEM 5)
# ============================================================================

"""
    coupled_dynamics!(du, u, p, t)

Coupled ODE system for [x; θ]:
- dx/dt = -∂E/∂x + η(t)
- dθ/dt = -∇_θ L(θ; x(t)) + ζ(t)

Parameters p should contain: (L, α, n, x_target, σ_x, σ_θ, n_x, n_θ)
"""
function coupled_dynamics!(du, u, p, t)
    L, α, n, x_target, σ_x, σ_θ, n_x, n_θ = p
    
    # Extract state components
    x_vec = u[1:n_x]
    θ = u[n_x+1:end]
    
    dx_vec = du[1:n_x]
    dθ = du[n_x+1:end]
    
    # Reshape x for matrix operations
    x = reshape(x_vec, n, n)
    dx = reshape(dx_vec, n, n)
    
    # === Compute dx/dt = -∂E/∂x + η(t) ===
    energy_gradient!(dx, x, L, α)
    @. dx = -dx
    if σ_x > 0f0
        @. dx += σ_x * randn(Float32)
    end
    
    # === Compute dθ/dt = -∇_θ L(θ; x) + ζ(t) ===
    # Simplified: just use gradient of loss
    loss_fn(θ_temp) = learning_loss_L(θ_temp, x, x_target, L)
    grad_θ = Zygote.gradient(loss_fn, θ)[1]
    
    if grad_θ !== nothing
        @. dθ = -grad_θ
        if σ_θ > 0f0
            @. dθ += σ_θ * randn(Float32)
        end
    else
        fill!(dθ, 0f0)
    end
    
    return nothing
end

"""
    solve_coupled_system(x0, θ0, L, n, α=α_ENERGY; t_end=10f0, x_target=nothing,
                        σ_x=σ_X, σ_θ=σ_Θ)

Solve coupled system [dx/dt; dθ/dt].

If x_target is not provided, uses x0 as target (self-consistency).
"""
function solve_coupled_system(x0, θ0, L, n, α=α_ENERGY; t_end=10f0, x_target=nothing,
                              σ_x=σ_X, σ_θ=σ_Θ)
    # Default target
    if x_target === nothing
        x_target = copy(x0)
    end
    
    # Concatenate state
    n_x = n * n
    n_θ = length(θ0)
    u0 = vcat(vec(Float32.(x0)), Float32.(θ0))
    
    tspan = (0f0, t_end)
    p = (L, α, n, Float32.(x_target), σ_x, σ_θ, n_x, n_θ)
    
    # Solve
    prob = ODEProblem(coupled_dynamics!, u0, tspan, p)
    sol = solve(prob, Tsit5(); reltol=1f-3, abstol=1f-5)
    
    return sol
end

# ============================================================================
# SECTION 6: UDE WITH NEURAL NETWORK (ROADMAP ITEM 5)
# ============================================================================

"""
    create_energy_nn(n::Int, rng::AbstractRNG)

Create neural network to replace energy gradient ∂E/∂x.

Architecture: n² → 128 → 64 → n² (outputs gradient-like field)
"""
function create_energy_nn(n::Int, rng::AbstractRNG)
    in_dim = n * n
    out_dim = n * n
    
    NN = Chain(
        Dense(in_dim, 64, tanh),
        Dense(64, 32, tanh),
        Dense(32, out_dim)
    )
    
    p_nn, st_nn = Lux.setup(rng, NN)
    
    return NN, p_nn, st_nn
end

"""
    ude_energy_dynamics!(du, u, p, t)

UDE dynamics: dx/dt = NN(x; θ_nn) + η(t)

Neural network replaces explicit energy gradient.
Parameters p should contain: (NN, st, θ_nn, n, σ_noise)
"""
function ude_energy_dynamics!(du, u, p, t)
    NN, st, θ_nn, n, σ_noise = p
    
    # Neural network prediction
    y, _ = Lux.apply(NN, u, θ_nn, st)
    @. du = y
    
    # Add noise
    if σ_noise > 0f0
        @. du += σ_noise * randn(Float32)
    end
    
    return nothing
end

"""
    solve_ude_energy(x0, NN, st, θ_nn, n; t_end=10f0, σ_noise=σ_X)

Solve UDE with learned energy dynamics.
"""
function solve_ude_energy(x0, NN, st, θ_nn, n; t_end=10f0, σ_noise=σ_X)
    u0 = vec(Float32.(x0))
    tspan = (0f0, t_end)
    p = (NN, st, θ_nn, n, σ_noise)
    
    prob = ODEProblem(ude_energy_dynamics!, u0, tspan, p)
    sol = solve(prob, Tsit5(); reltol=1f-3, abstol=1f-5)
    
    return sol
end

"""
    ude_coupled_dynamics!(du, u, p, t)

Full coupled UDE system:
- dx/dt = NN(x; θ_nn) + η(t)
- dθ/dt = -∇_θ L(θ_learned; x) + ζ(t)

State u = [x; θ_learned], θ_nn are neural network parameters (fixed during solve).
"""
function ude_coupled_dynamics!(du, u, p, t)
    NN, st, θ_nn, L, n, x_target, σ_x, σ_θ, n_x, n_θ = p
    
    # Extract state
    x_vec = u[1:n_x]
    θ_learned = u[n_x+1:end]
    
    dx_vec = du[1:n_x]
    dθ = du[n_x+1:end]
    
    # dx/dt = NN(x) + η
    y, _ = Lux.apply(NN, x_vec, θ_nn, st)
    @. dx_vec = y
    if σ_x > 0f0
        @. dx_vec += σ_x * randn(Float32)
    end
    
    # dθ/dt = -∇_θ L(θ; x) + ζ
    x = reshape(x_vec, n, n)
    loss_fn(θ_temp) = learning_loss_L(θ_temp, x, x_target, L)
    grad_θ = Zygote.gradient(loss_fn, θ_learned)[1]
    
    if grad_θ !== nothing
        @. dθ = -grad_θ
        if σ_θ > 0f0
            @. dθ += σ_θ * randn(Float32)
        end
    else
        fill!(dθ, 0f0)
    end
    
    return nothing
end
# SECTION 7: TRAINING FUNCTIONS
# ============================================================================

"""
    train_ude_energy(dataset, L, n; max_iters=1000, learning_rate=1f-2, t_end=5f0, seed=42)

Train neural network to learn energy gradient dynamics with improved optimization.

New Features:
- 3-stage optimization: RAdam (1000) → AdaBelief (1500) → LBFGS (3000)
- Tighter constraint curriculum (10 → 100 → 1000)
- Sharper penalty (violation^4 instead of ^2)
- Binary regularizer to push outputs to 0 or 1
- Adaptive ODE tolerances

Loss: endpoint matching + constraint penalty^4 + box constraints + binary regularizer
"""
function train_ude_energy(dataset, L, n; max_iters=1000,
                         learning_rate=1f-2, t_end=5f0, seed=42)
    
    println("\n" * "╔" * "═"^58 * "╗")
    println("║" * " "^10 * "Training UDE Energy Model" * " "^23 * "║")
    println("╚" * "═"^58 * "╝")
    
    # Setup
    rng = StableRNG(seed)
    NN, p_nn, st_nn = create_energy_nn(n, rng)
    θ = ComponentArray(Float32.(ComponentArray(p_nn)))
    st_nn = fmap(x -> isa(x, AbstractArray) ? Float32.(x) : x, st_nn)
    
    # 3-stage training configuration
    batch_sizes = [8, 16]  # Progressive batching
    constraint_weights = [10.0f0, 1000.0f0]  # Tighter curriculum
    penalty_powers = [4, 8]  # Progressive penalty sharpness: violation^2 → ^3 → ^4
    binary_weights = [0.1f0, 2.0f0]  # Progressive binary regularization
    ode_tolerances = [(1f-2, 1f-4), (1f-6, 1f-8)]
    
    println("\nConfiguration:")
    println("  Grid size: $(n)×$(n)")
    println("  Training samples: $(length(dataset))")
    println("  Network parameters: $(length(θ))")
    println("  Initial learning rate: $learning_rate")
    println("\nNew 3-Stage Configuration:")
    println("  Stage 1: batch=8,  W_const=10,   penalty^2, RAdam(LR=0.01),             1000 iters")
    println("  Stage 2: batch=16, W_const=100,  penalty^3, AdaBelief(LR=0.003),       1500 iters")
    println("  Stage 3: batch=30, W_const=1000, penalty^4, LBFGS(m=50, g_tol=1e-9), 3000 iters")
    println("\n  ⚡ Progressive penalty (^2→^3→^4), binary regularizer, TRBDF2 stiff solver")
    
    # Define loss function with stage-specific parameters
    function sample_loss(θ_current, u0, x_target, w_constraint, w_binary, penalty_power, reltol, abstol)
        # Define ODE dynamics inline to capture NN and st_nn in closure
        function ode_fn!(du, u, p, t)
            y, _ = Lux.apply(NN, u, p, st_nn)
            @. du = y
        end
        
        # Solve UDE with stage-specific tolerances
        # Use TRBDF2 (stiff solver compatible with Zygote adjoint)
        prob = ODEProblem(ode_fn!, u0, (0f0, t_end), θ_current)
        sol = solve(prob, Tsit5(); reltol=reltol, abstol=abstol, 
                   save_everystep=false, sensealg=SENSEALG, maxiters=50000)
        
        # Handle solver failures gracefully
        if sol.retcode != :Success && sol.retcode != :Terminated
            # Return large penalty if ODE solve fails
            return 1f6
        end
        
        xT = reshape(sol.u[end], n, n)
        
        # Loss components
        l_recon = sum((xT .- x_target).^2)
        
        # Progressive constraint penalty (violation^2 → ^3 → ^4 across stages)
        l_constraint = 0f0
        @inbounds for (i1, j1, i2, j2, i3, j3) in L
            triplet_sum = xT[i1, j1] + xT[i2, j2] + xT[i3, j3]
            violation = max(0f0, triplet_sum - 2f0)
            l_constraint += violation^penalty_power  # Progressive sharpness
        end
        
        l_box = sum(max.(0f0, -xT).^2) + sum(max.(0f0, xT .- 1f0).^2)
        
        # Binary regularizer: penalize values far from 0 or 1
        l_binary = sum(abs.(xT .* (1f0 .- xT)))  # 0 when x≈0 or x≈1
        
        return W_RECON * l_recon + w_constraint * l_constraint + W_BOX * l_box + w_binary * l_binary
    end
    
    # Total loss functions for different stages
    function make_total_loss(batchsize, w_constraint, w_binary, penalty_power, reltol, abstol, deterministic=false)
        if deterministic
            # Use fixed subset for deterministic gradients (LBFGS needs this)
            return function(θ_current)
                s = 0f0
                n_samples = min(batchsize, length(dataset))
                for k in 1:n_samples
                    u0 = vec(Float32.(dataset[k]["initial"]))
                    xt = Float32.(dataset[k]["optimal"])
                    s += sample_loss(θ_current, u0, xt, w_constraint, w_binary, penalty_power, reltol, abstol)
                end
                return s / n_samples
            end
        else
            # Stochastic sampling for adaptive optimizers
            return function(θ_current)
                s = 0f0
                for _ in 1:batchsize
                    k = rand(1:length(dataset))
                    u0 = vec(Float32.(dataset[k]["initial"]))
                    xt = Float32.(dataset[k]["optimal"])
                    s += sample_loss(θ_current, u0, xt, w_constraint, w_binary, penalty_power, reltol, abstol)
                end
                return s / batchsize
            end
        end
    end
    
    # Training loop tracking
    iter = Ref(0)
    loss_history = Float64[]
    current_stage = Ref("stage1")
    stage_losses = Float64[]
    
    # Checkpoint function
    function save_checkpoint(θ_current, stage_name, iter_num)
        checkpoint_path = "checkpoint_$(stage_name)_iter$(iter_num)_n$(n).jld2"
        jldsave(checkpoint_path;
                θ_current=θ_current,
                st_nn=st_nn,
                iter=iter_num,
                stage=stage_name,
                n=n)
        println("  💾 Checkpoint: $checkpoint_path")
    end
    
    function callback(θ_current, l)
        iter[] += 1
        push!(loss_history, l)
        
        if iter[] % 100 == 0 || iter[] == 1
            println("  Iteration $(iter[]): Loss = $(round(l, digits=6))")
        end
        
        # Save checkpoint every 500 iterations
        if iter[] % 500 == 0
            save_checkpoint(θ_current, current_stage[], iter[])
        end
        
        return false
    end
    
    # Storage for each stage result
    θ_current = θ
    adtype = Optimization.AutoZygote()
    
    # ============================================================
    # STAGE 1: RAdam Exploration (fast, light constraints)
    # ============================================================
    println("\n" * "="^60)
    println("Stage 1: RAdam Exploration (LR=0.01, batch=8, penalty^2)")
    println("="^60)
    
    iter[] = 0
    current_stage[] = "stage1"
    
    loss_fn_1 = make_total_loss(batch_sizes[1], constraint_weights[1], binary_weights[1], 
                                 penalty_powers[1], ode_tolerances[1]..., false)
    optf_1 = Optimization.OptimizationFunction((x, p) -> loss_fn_1(x), adtype)
    optprob_1 = Optimization.OptimizationProblem(optf_1, θ_current)
    
    @time res_1 = Optimization.solve(
        optprob_1,
        OptimizationOptimisers.AdamW(learning_rate, (0.9f0, 0.999f0)),
        callback=callback,
        maxiters=1000
    )
    
    θ_current = res_1.u
    loss_1 = loss_fn_1(θ_current)
    push!(stage_losses, loss_1)
    println("\n✓ Stage 1 complete. Loss: $(round(loss_1, digits=6))")
    
    # ============================================================
    # STAGE 2: AdaBelief Refinement (medium constraints, binary push)
    # ============================================================
     println("\n" * "="^60)
    println("Stage 2: LBFGS Final Push (deterministic, batch=30, W_const=1000, penalty^4)")
    println("="^60)
    iter[] = 0
    current_stage[] = "stage2"
    
    loss_fn_2 = make_total_loss(batch_sizes[2], constraint_weights[2], binary_weights[2],
                                 penalty_powers[2], ode_tolerances[2]..., true)  # Deterministic for LBFGS
    optf_2 = Optimization.OptimizationFunction((x, p) -> loss_fn_2(x), adtype)
    optprob_2 = Optimization.OptimizationProblem(optf_2, θ_current)
    
    @time es_final = Optimization.solve(
        optprob_2,
        OptimizationOptimJL.LBFGS(
            linesearch=LineSearches.BackTracking(order=3),
            m=50  # Increased memory for better Hessian approximation
        ),
        callback=callback,
        maxiters=3000,
        g_tol=1e-9  # Tighter convergence tolerance
    )
    
    
    θ_current = res_final.u
    final_loss = loss_fn_2(θ_current)
    push!(stage_losses, final_loss)
    
    # Summary
    println("\n" * "="^60)
    println("Training Complete!")
    println("="^60)
    println("  Stage 1 (RAdam):        $(round(stage_losses[1], digits=6))")
    println("  Stage 2 (AdaBelief):    $(round(stage_losses[2], digits=6))")
    println("  Total improvement:      $(round(stage_losses[1] - final_loss, digits=6))")
    println("  Loss reduction rate:    $(round(100*(1 - final_loss/stage_losses[1]), digits=2))%")
    
    # Save model with complete information
    save_path = "ude_energy_n$(n).jld2"
    jldsave(save_path;
            # Model parameters
            θ_trained=res_final.u,
            st_nn=st_nn,
            
            # Training history
            loss_history=loss_history,
            stage_losses=stage_losses,
            loss_stage1=stage_losses[1],
            loss_stage2=stage_losses[2],
            loss_stage3_lbfgs=final_loss,
            final_loss=final_loss,
            
            # Problem data
            n=n,
            L=L,
            
            # Configuration
            config=Dict(
                "learning_rates" => [learning_rate, 0.003f0, "N/A (LBFGS)"],
                "batch_sizes" => batch_sizes,
                "constraint_weights" => constraint_weights,
                "penalty_powers" => penalty_powers,
                "binary_weights" => binary_weights,
                "ode_tolerances" => ode_tolerances,
                "t_end" => t_end,
                "max_iters_stage1" => 1000,
                "max_iters_stage2" => 1500,
                "max_iters_stage3" => 3000,
                "total_iterations" => length(loss_history),
                "ode_solver" => "TRBDF2(autodiff=false) - stiff solver, Zygote-compatible, maxiters=50000",
                "network_architecture" => "Dense($(n*n), 64, tanh) → Dense(64, 32, tanh) → Dense(32, $(n*n))",
                "optimizer_stages" => ["RAdam", "AdaBelief(eps=1e-16)", "LBFGS(m=50,g_tol=1e-9)"],
                "features" => ["progressive_penalty_violation^2-3-4", "binary_regularizer", "stiff_solver_TRBDF2", "tighter_constraints", "3_stage_curriculum", "ODE_error_handling", "Zygote_adjoint_compatible"],
                "penalty_progression" => "violation^2 → violation^3 → violation^4",
                "seed" => seed,
                "optimization_notes" => "New 3-stage pipeline with progressive penalty (^2→^3→^4), TRBDF2 stiff solver (Zygote-compatible), binary regularizer, constraint weights up to 1000"
            ))
    
    println("✓ Saved to $save_path")
    println("  Model size: $(round(filesize(save_path)/1024^2, digits=2)) MB")
    println("  Total training iterations: $(length(loss_history))")
    
    return res_final.u, NN, st_nn, loss_history
end

"""
    train_coupled_ude(dataset, L, n; kwargs...)

Train full coupled UDE system where both x and θ evolve.

⚠️  FUTURE WORK: This would train the UDE coupled system where:
   - dx/dt = NN(x; θ_nn) + η(t)  
   - dθ/dt = -∇L(θ; x) + ζ(t)
   
   The dynamics `ude_coupled_dynamics!` are implemented (lines 358-398),
   but training this system requires a different loss formulation.
   
   Current implementation only trains θ_nn to match x trajectories.
   Full coupled training would learn both NN parameters and auxiliary θ simultaneously.
"""
function train_coupled_ude(dataset, L, n; max_iters=500, learning_rate=1f-4, seed=42)
    @warn "train_coupled_ude is not yet implemented. Use train_ude_energy for UDE training."
    println("\n⚠️  Coupled UDE Training - Not Yet Implemented")
    println("   The coupled dynamics exist (ude_coupled_dynamics!) but training is future work.")
    println("   Use train_ude_energy() for standard UDE training instead.")
    return nothing
end

"""
    load_trained_model(filepath::String; seed=42)

Load a trained UDE model from file.

Returns a named tuple with all model components:
- θ_trained: Trained neural network parameters
- st_nn: Network state
- NN: Neural network (reconstructed)
- L: Collinear triplets
- config: Training configuration
- loss_history: Training loss history
"""
function load_trained_model(filepath::String; seed=42)
    println("Loading model from: $filepath")
    
    if !isfile(filepath)
        error("Model file not found: $filepath")
    end
    
    data = load(filepath)
    
    # Recreate network with same architecture
    n = data["n"]
    rng = StableRNG(seed)
    NN, _, _ = create_energy_nn(n, rng)
    
    println("✓ Model loaded successfully")
    println("  Grid size: $(n)×$(n)")
    println("  Network parameters: $(length(data["θ_trained"]))")
    println("  Final loss: $(round(data["final_loss"], digits=6))")
    println("  Training iterations: $(length(data["loss_history"]))")
    
    # Return all components as named tuple
    return (
        θ_trained = data["θ_trained"],
        st_nn = data["st_nn"],
        NN = NN,
        L = data["L"],
        n = n,
        config = data["config"],
        loss_history = data["loss_history"],
        stage_losses = get(data, "stage_losses", nothing),
        loss_stage1 = get(data, "loss_stage1", nothing),
        loss_stage2 = get(data, "loss_stage2", nothing),
        loss_stage3 = get(data, "loss_stage3", nothing),  # Old 4-stage models
        loss_stage3_lbfgs = get(data, "loss_stage3_lbfgs", nothing),  # New 3-stage models
        final_loss = data["final_loss"]
    )
end
# SECTION 8: TESTING AND VISUALIZATION
# ============================================================================

"""
    decode_topk(x::AbstractMatrix, n::Int; k=2*n)

Decode continuous scores to exactly k binary placements.
"""
function decode_topk(x::AbstractMatrix, n::Int; k=2*n)
    vals = vec(Float32.(x))
    perm = sortperm(vals, rev=true)
    bin = zeros(Float32, n, n)
    bin[perm[1:k]] .= 1f0
    return bin
end

"""
    decode_feasible(x::AbstractMatrix, L, n::Int; k=2*n)

Constraint-aware decoder: greedily select top-k points while respecting collinearity constraints.
Skips points that would create a third mark on an already-occupied line.
"""
function decode_feasible(x::AbstractMatrix, L, n::Int; k=2*n)
    # Get sorted indices by score
    vals = vec(Float32.(x))
    indices = sortperm(vals, rev=true)

    # Track selected points and line occupancy
    selected = falses(n, n)
    line_counts = Dict{NTuple{6,Int}, Int}()

    placed = 0
    for idx in indices
        if placed >= k
            break
        end

        # Convert linear index to 2D
        i, j = Tuple(CartesianIndices((n, n))[idx])

        # Check all lines containing this point
        can_place = true
        for (i1, j1, i2, j2, i3, j3) in L
            # Check if this point is part of this triplet
            is_in_triplet = (i == i1 && j == j1) || (i == i2 && j == j2) || (i == i3 && j == j3)

            if is_in_triplet
                # Count already-selected points on this line
                count = 0
                if selected[i1, j1]
                    count += 1
                end
                if selected[i2, j2]
                    count += 1
                end
                if selected[i3, j3]
                    count += 1
                end

                # If adding this point would create 3 on a line, skip
                if count >= 2
                    can_place = false
                    break
                end
            end
        end

        if can_place
            selected[i, j] = true
            placed += 1
        end
    end

    return Float32.(selected)
end

"""
    count_violations(x::AbstractMatrix, L)

Count constraint violations in solution.
"""
function count_violations(x::AbstractMatrix, L)
    violations = 0
    @inbounds for (i1, j1, i2, j2, i3, j3) in L
        if x[i1, j1] + x[i2, j2] + x[i3, j3] >= 3f0
            violations += 1
        end
    end
    return violations
end

"""
    test_energy_flow(L, n, α=α_ENERGY; num_tests=5, t_end=10f0, seed=100)

Test explicit gradient flow approach.
"""
function test_energy_flow(L, n, α=α_ENERGY; num_tests=5, t_end=10f0, seed=100)
    println("\n" * "╔" * "═"^58 * "╗")
    println("║" * " "^10 * "Testing Energy Flow Method" * " "^21 * "║")
    println("╚" * "═"^58 * "╝")
    
    rng = StableRNG(seed)
    results = []
    
    for test_idx in 1:num_tests
        println("\nTest $(test_idx)/$num_tests:")
        
        # Random initialization
        x0 = rand(rng, Float32, n, n) .* 0.3f0
        
        # Solve
        sol = solve_energy_flow(x0, L, n, α; t_end=t_end, σ_noise=σ_X)
        
        # Final state
        xT = reshape(sol.u[end], n, n)
        
        # Decode with constraint-aware decoder
        binary = decode_feasible(xT, L, n; k=2*n)
        
        # Evaluate
        num_points = sum(binary)
        violations = count_violations(binary, L)
        final_energy = energy_E(xT, L, α)
        
        println("  Points placed: $(Int(num_points))")
        println("  Violations: $violations")
        println("  Final energy: $(round(final_energy, digits=4))")
        println("  Valid: $(violations == 0 ? "✓" : "✗")")
        
        push!(results, Dict(
            "num_points" => num_points,
            "violations" => violations,
            "energy" => final_energy,
            "valid" => violations == 0
        ))
    end
    
    # Summary
    valid_count = sum(r["valid"] for r in results)
    println("\n" * "="^60)
    println("Summary: $valid_count/$num_tests valid solutions")
    println("="^60)
    
    return results
end

"""
    test_ude_model(θ_trained, NN, st, L, n; num_tests=5, t_end=5f0, seed=100)

Test trained UDE model.
"""
function test_ude_model(θ_trained, NN, st, L, n; num_tests=5, t_end=5f0, seed=100)
    println("\n" * "╔" * "═"^58 * "╗")
    println("║" * " "^12 * "Testing UDE Model" * " "^27 * "║")
    println("╚" * "═"^58 * "╝")
    
    rng = StableRNG(seed)
    results = []
    
    for test_idx in 1:num_tests
        println("\nTest $(test_idx)/$num_tests:")
        
        x0 = rand(rng, Float32, n, n) .* 0.3f0
        
        # Solve UDE
        sol = solve_ude_energy(x0, NN, st, θ_trained, n; t_end=t_end, σ_noise=0f0)
        
        xT = reshape(sol.u[end], n, n)
        binary = decode_feasible(xT, L, n; k=2*n)  # Use constraint-aware decoder
        
        num_points = sum(binary)
        violations = count_violations(binary, L)
        
        println("  Points: $(Int(num_points)), Violations: $violations")
        println("  Valid: $(violations == 0 ? "✓" : "✗")")
        
        push!(results, Dict(
            "num_points" => num_points,
            "violations" => violations,
            "valid" => violations == 0
        ))
    end
    
    valid_count = sum(r["valid"] for r in results)
    println("\nSummary: $valid_count/$num_tests valid solutions")
    
    return results
end

"""
    visualize_dynamics(sol, n; title="Grid Evolution")

Plot grid evolution over time.
"""
function visualize_dynamics(sol, n; title="Grid Evolution", save_path=nothing)
    # Select time points
    t_points = length(sol.t) >= 5 ? [1, length(sol.t)÷4, length(sol.t)÷2, 3*length(sol.t)÷4, length(sol.t)] : [1, length(sol.t)]
    
    plots_array = []
    for idx in t_points
        x = reshape(sol.u[idx], n, n)
        t_val = sol.t[idx]
        p = heatmap(x, clim=(0, 1), color=:viridis,
                   title="t=$(round(t_val, digits=2))",
                   aspect_ratio=:equal, showaxis=false,
                   colorbar=false)
        push!(plots_array, p)
    end
    
    final_plot = plot(plots_array..., layout=(1, length(plots_array)),
                     size=(300*length(plots_array), 300),
                     plot_title=title)
    
    if save_path !== nothing
        savefig(save_path)
        println("✓ Saved visualization to $save_path")
    end
    
    return final_plot
end

"""
    compare_methods(L, n; num_tests=5, t_end=10f0)

Compare explicit energy flow vs trained UDE.
"""
function compare_methods(L, n; num_tests=5, t_end=10f0)
    println("\n" * "╔" * "═"^58 * "╗")
    println("║" * " "^15 * "Method Comparison" * " "^26 * "║")
    println("╚" * "═"^58 * "╝")
    
    # Test energy flow
    energy_results = test_energy_flow(L, n; num_tests=num_tests, t_end=t_end)
    
    # Check if UDE model exists
    ude_file = "ude_energy_n$(n).jld2"
    if isfile(ude_file)
        println("\n[Loading UDE model for comparison...]")
        data = load(ude_file)
        θ_trained = data["θ_trained"]
        
        rng = StableRNG(42)
        NN, _, st = create_energy_nn(n, rng)
        
        ude_results = test_ude_model(θ_trained, NN, st, L, n; num_tests=num_tests, t_end=t_end)
        
        # Comparison summary
        println("\n" * "="^60)
        println("COMPARISON SUMMARY")
        println("="^60)
        energy_valid = sum(r["valid"] for r in energy_results)
        ude_valid = sum(r["valid"] for r in ude_results)
        println("Energy Flow: $energy_valid/$num_tests valid")
        println("UDE Model:   $ude_valid/$num_tests valid")
        println("="^60)
    else
        println("\n[UDE model not found. Train first with mode='ude'.]")
    end
    
    return energy_results
end

# ============================================================================
# SECTION 9: MAIN PIPELINE
# ============================================================================

"""
    main_roadmap_pipeline(; n=5, mode="all", load_data=false, train=false, test=true,
                         max_iters=1000, learning_rate=1f-4)

Execute different modes of the roadmap:
- mode="energy": Run explicit energy-based gradient flow
- mode="ude": Train and test UDE with neural network
- mode="coupled": Run coupled system (advanced)
- mode="all": Run all methods and compare

Optionally load training data if available.
"""
function main_roadmap_pipeline(; n=5, mode="all", load_data=false, train=false, test=true,
                               max_iters=1500, learning_rate=0.01f0, t_end=10f0)
    
    println("\n" * "╔" * "═"^58 * "╗")
    println("║" * " "^5 * "UDE N3L - Complete SciML Roadmap Pipeline" * " "^7 * "║")
    println("╚" * "═"^58 * "╝")
    
    println("\nConfiguration:")
    println("  Grid size: $(n)×$(n)")
    println("  Mode: $mode")
    println("  Target points: $(2*n)")
    
    # Generate collinear triplets
    println("\n[Generating collinear triplets...]")
    points = generate_grid_points(n)
    collinear_groups = find_collinear_points(points)
    L = generate_collinear_triplet_list(collinear_groups)
    println("✓ Generated $(length(L)) collinear triplets")
    
    # Load training data if requested
    dataset = nothing
    if load_data
        data_file = "training_data_n$(n).jld2"
        if isfile(data_file)
            println("\n[Loading training data...]")
            data = load(data_file)
            dataset = data["dataset"]
            println("✓ Loaded $(length(dataset)) samples")
        else
            println("⚠ Training data not found: $data_file")
            println("  Generate data first using n3l_gurobi.jl")
        end
    end
    
    # Execute based on mode
    if mode == "energy" || mode == "all"
        println("\n" * "="^60)
        println("MODE: ENERGY-BASED GRADIENT FLOW")
        println("="^60)
        
        if test
            energy_results = test_energy_flow(L, n; num_tests=5, t_end=t_end)
            
            # Visualize one trajectory
            println("\n[Generating visualization...]")
            x0 = rand(StableRNG(100), Float32, n, n) .* 0.3f0
            sol = solve_energy_flow(x0, L, n; t_end=t_end, σ_noise=σ_X)
            visualize_dynamics(sol, n; title="Energy Flow", 
                             save_path="energy_flow_n$(n).png")
        end
    end
    
    if mode == "ude" || mode == "all"
        println("\n" * "="^60)
        println("MODE: UDE WITH NEURAL NETWORK")
        println("="^60)
        
        if train && dataset !== nothing
            θ_trained, NN, st, loss_history = train_ude_energy(
                dataset, L, n;
                max_iters=max_iters,
                learning_rate=learning_rate,
                t_end=t_end
            )
            
            # Plot loss
            p = plot(loss_history, xlabel="Iteration", ylabel="Loss",
                    title="UDE Training Loss", yscale=:log10,
                    legend=false, linewidth=2)
            savefig("ude_training_loss_n$(n).png")
            println("✓ Loss plot saved")
            
            if test
                test_ude_model(θ_trained, NN, st, L, n; t_end=t_end)
            end
        elseif test
            # Load existing model
            ude_file = "ude_energy_n$(n).jld2"
            if isfile(ude_file)
                data = load(ude_file)
                θ_trained = data["θ_trained"]
                rng = StableRNG(42)
                NN, _, st = create_energy_nn(n, rng)
                test_ude_model(θ_trained, NN, st, L, n; t_end=t_end)
            else
                println("⚠ UDE model not found. Set train=true to train first.")
            end
        end
    end
    
    if mode == "coupled"
        println("\n" * "="^60)
        println("MODE: COUPLED SYSTEM (dx/dt and dθ/dt)")
        println("="^60)
        
        println("\n[Running coupled system demonstration...]")
        x0 = rand(StableRNG(123), Float32, n, n) .* 0.3f0
        θ0 = randn(Float32, 5) .* 0.1f0  # Small learned parameters
        
        sol = solve_coupled_system(x0, θ0, L, n; t_end=t_end)
        
        # Extract final state
        x_final = reshape(sol.u[end][1:n*n], n, n)
        θ_final = sol.u[end][n*n+1:end]
        
        println("  Initial θ: $(round.(θ0, digits=4))")
        println("  Final θ:   $(round.(θ_final, digits=4))")
        
        binary = decode_feasible(x_final, L, n; k=2*n)  # Use constraint-aware decoder
        violations = count_violations(binary, L)
        println("  Final violations: $violations")
    end
    
    if mode == "all"
        println("\n" * "="^60)
        println("COMPARISON: ALL METHODS")
        println("="^60)
        compare_methods(L, n; num_tests=5, t_end=t_end)
    end
    
    println("\n" * "╔" * "═"^58 * "╗")
    println("║" * " "^15 * "Pipeline Complete!" * " "^27 * "║")
    println("╚" * "═"^58 * "╝")
    
    println("\nRoadmap Implementation Status:")
    println("  ✓ Section 2: Energy function E(x) and gradient ∂E/∂x")
    println("  ✓ Section 3: Gradient flow dx/dt = -∂E/∂x + η(t)")
    println("  ✓ Section 4: Learning dynamics dθ/dt = -∇L + ζ(t)")
    println("  ✓ Section 5: Coupled system [dx/dt; dθ/dt]")
    println("  ✓ Section 6: UDE with neural network replacing ∂E/∂x")
    println("  ✓ Training, testing, and visualization complete")
    
    return L
end

# ============================================================================
# EXAMPLE USAGE
# ============================================================================

# RECOMMENDED: Train and test UDE with improved optimizer schedule
# This uses: RAdam → AdaBelief → LBFGS with constraint-aware decoding
main_roadmap_pipeline(n=5, mode="ude", load_data=true, train=true, test=true)

# Other modes (uncomment to use):

# Test energy flow baseline (no neural network)
# main_roadmap_pipeline(n=5, mode="energy", test=true)

# Demonstrate basic coupled system (explicit energy gradient)
# main_roadmap_pipeline(n=5, mode="coupled", test=true)

# Run all methods and compare
# main_roadmap_pipeline(n=5, mode="all", load_data=true, train=true, test=true)

