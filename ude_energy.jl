"""
Universal Differential Equations for No-Three-in-Line Problem
Simplified implementation following SciML best practices
"""

# ============================================================================
# IMPORTS
# ============================================================================

using HDF5, JLD2
using Lux, DiffEqFlux, DifferentialEquations
using Optimization, OptimizationOptimisers, OptimizationOptimJL
using ComponentArrays
using Random, Statistics
using Plots
using Zygote
using Functors
using LineSearches
using Dates
using StableRNGs
# using NNlib

# ============================================================================
# DATA LOADING
# ============================================================================

"""
Load training data from HDF5 file
Returns n (grid size), triplets (collinear constraints), and samples
"""
function load_data(path::String)
    h5open(path, "r") do f
        n = Int32(read(f["n"]))
        # Transpose triplets - Python stores (N, 6), Julia reads as (6, N)
        triplets = read(f["triplets"])'
        
        ds = f["dataset"]
        samples = []
        
        for k in keys(ds)
            g = ds[k]
            # Transpose grids because Python uses row-major, Julia uses column-major
            push!(samples, (
                initial = Float32.(read(g["initial_grid"])'),
                target = Float32.(read(g["target_grid"])'),
                mask = Float32.(read(g["mask_top2n"]))'
            ))
        end
        
        return n, triplets, samples
    end
end

# Load data
data_path = "data/n_5.h5"
n, triplets, samples = load_data(data_path)
println("Loaded $(length(samples)) samples for $(n)×$(n) grid")

# Convert triplets to 1-based indexing
triplets = Int32.(triplets)

# Build unique lines (1-based (row,col) indices) once for differentiable loss
function _build_lines(n::Int, triplets::AbstractMatrix{<:Integer})
    # Triplets are already in Julia format: (i,j) where i=row, j=col (1-based)
    # We compute canonical line equations directly from these coordinates
    canon(A::Int, B::Int, C::Int) = begin
        g = gcd(gcd(abs(A), abs(B)), abs(C))
        if g > 0; A ÷= g; B ÷= g; C ÷= g; end
        if A < 0 || (A == 0 && B < 0) || (A == 0 && B == 0 && C < 0)
            A = -A; B = -B; C = -C
        end
        (A, B, C)
    end
    linekey_from_rc(i1, j1, i2, j2) = begin
        # Direct computation using (i,j) as (row, col) coordinates
        # Must match Python: A = y1-y2, B = x2-x1, C = x1*y2-x2*y1
        # where x=row=i, y=col=j
        i1 = Int(i1); j1 = Int(j1); i2 = Int(i2); j2 = Int(j2)
        A = j1 - j2  # col difference (matches y1 - y2)
        B = i2 - i1  # row difference (matches x2 - x1)
        C = i1*j2 - i2*j1  # matches x1*y2 - x2*y1
        canon(A, B, C)
    end
    lp = Dict{NTuple{3,Int}, Base.Set{Tuple{Int,Int}}}()
    @inbounds for row in eachrow(triplets)
        i1, j1, i2, j2, i3, j3 = row
        i1 = Int(i1); j1 = Int(j1); i2 = Int(i2); j2 = Int(j2); i3 = Int(i3); j3 = Int(j3)
        k = linekey_from_rc(i1, j1, i2, j2)
        s = get!(lp, k, Base.Set{Tuple{Int,Int}}())
        push!(s, (i1, j1)); push!(s, (i2, j2)); push!(s, (i3, j3))
    end
    # Convert to sorted vectors of (i,j) with length ≥ 3
    lines = Vector{Vector{Tuple{Int,Int}}}()
    for s in values(lp)
        if length(s) >= 3
            v = collect(s)
            sort!(v, by = x -> (x[2], x[1]))  # sort by col then row for consistent endpoints
            push!(lines, v)
        end
    end
    return lines
end

const LINES = _build_lines(Int(n), triplets)


# ============================================================================
# NEURAL NETWORK SETUP  
# ============================================================================

# Initialize random number generator
rng = Xoshiro(42)

# Grid dimensions
F = n * n

# Define neural networks for UDE terms (similar to epidemiology.jl style)
# Simplified to single network for dynamics
NN_dynamics = Chain(
    Dense(F, 128, tanh),
    Dense(128, 64, tanh),
    Dense(64, F)
)

# Setup parameters
p_nn, st_nn = Lux.setup(rng, NN_dynamics)
# Convert parameters and state to Float32 for consistency with data
p_nn = fmap(x -> isa(x, AbstractArray) ? Float32.(x) : x, p_nn)
st_nn = fmap(x -> isa(x, AbstractArray) ? Float32.(x) : x, st_nn)
θ0 = ComponentArray(p_nn)

println("Network parameters: $(length(θ0))")

# ============================================================================
# UDE DYNAMICS
# ============================================================================

"""
UDE dynamics function: dx/dt = NN(x; θ)
Includes soft constraints for box bounds and collinearity
"""

# @inline sp(z::Float32; β::Float32=10f0) = (1f0/β) * log1p(exp(β*z))
# lower = sp(-u[i])
# upper = sp(u[i] - 1f0)

function ude_dynamics!(du, u, p, t)
    # Neural network prediction
    y, _ = Lux.apply(NN_dynamics, u, p, st_nn)
    
    # Box constraint correction (keep values in [0,1])
    penalty_strength = 10.0f0
    
    # Compute derivatives element-wise without mutation
    for i in eachindex(du)
        lower = (u[i] < 0) * u[i]
        upper = (u[i] > 1) * (u[i] - 1)
        du[i] = y[i] - penalty_strength * (lower + upper)
    end
    
    return nothing
end

# ============================================================================
# PREDICTION AND LOSS FUNCTIONS
# ============================================================================

"""
Solve ODE forward to get prediction at final time
"""
function predict(θ, u0; t_end=5.0f0)
    prob = ODEProblem(ude_dynamics!, u0, (0.0f0, t_end), θ)
    sol = solve(prob, Tsit5(); 
                reltol=1f-3, abstol=1f-5, 
                save_everystep=false,
                sensealg=InterpolatingAdjoint(autojacvec=ZygoteVJP()))
    
    return sol.u[end]
end

"""
Compute collinearity violations for soft penalty
"""
function compute_violations(x::AbstractVector, n)
    x_grid = reshape(x, n, n)
    viol = 0.0f0
    @inbounds for line in LINES
        s = 0.0f0
        for (i, j) in line
            s += x_grid[i, j]
        end
        v = max(0.0f0, s - 2.0f0)
        viol += v * v
    end
    return viol
end

"""
Loss function combining reconstruction and constraint violations
"""


# ============================================================================
# TRAINING
# ============================================================================

BATCH_SIZE = 8
MAX_ITER_ADAM = 1500
MAX_ITER_LBFGS = 1500
LR = 0.01f0

const ALPHA_POINTS   = 1.0f0    # reward for placing points
const BETA_VIOLATION = 50.0f0   # weight on 3-in-line violations
const GAMMA_BOX      = 10.0f0   # weight on box [0,1] penalty
# const ETA_SURPLUS    = 10.0f0   # penalty on points beyond 2n

# Loss weights
const LAMBDA_E       = 1.0f0    # weight on final energy
const LAMBDA_DELTAE  = 0.5f0    # weight on energy improvement (E_pred - E0)
const LAMBDA_RECON   = 0.5f0    # small reconstruction anchor (set 0f0 to disable)
const LAMBDA_WD      = 1e-4   # weight decay (set 0f0 to disable)

const ETA_LOW  = 10.0f0   # penalty when underfilling
const ETA_HIGH = 10.0f0  # penalty when overfilling

# Tiny L2 over params (ComponentArray-friendly)
function l2_params(θ)
    # θ is a ComponentVector; iterate its flattened entries directly
    s = 0.0f0
    @inbounds @simd for v in θ
        s += float(v) * float(v)
    end
    return s
end


function grid_energy(x::AbstractVector{<:Real}, n::Integer)
    # (1) Violation penalty
    viol = compute_violations(x, n)

    # (2) Box penalty (same as before)
    box = sum(max.(0.0f0, -x).^2 .+ max.(0.0f0, x .- 1.0f0).^2)

    # (3) Point equilibrium: penalty on deviation from 2n
    P = sum(x)
    diff_low  = max(0.0f0, 2.0f0 * Float32(n) - P)  # too few points
    diff_high = max(0.0f0, P - 2.0f0 * Float32(n))  # too many points

    # You can use asymmetric weights if you want to penalize surplus more heavily

    points_t = ETA_LOW * diff_low^2 + ETA_HIGH * diff_high^2

    return BETA_VIOLATION * viol + GAMMA_BOX * box + points_t
end

@inline softplus(z::Float32; β::Float32=10f0) = (1f0/β) * log1p(exp(β*z))

function loss(θ, batch_data)
    total = 0.0f0

    @inbounds for sample in batch_data
        u0     = vec(sample.initial)
        target = vec(sample.target)
        mask   = vec(sample.mask)

        # Forward prediction
        pred = predict(θ, u0)

        # Energy terms
        E_pred = grid_energy(pred, n)
        E0     = grid_energy(u0, n)     # encourages energy decrease

        # Optional: small reconstruction anchor for stability
        weight = 1.0f0 .+ 5.0f0 .* mask
        L_rec  = sum(weight .* (pred .- target).^2)

        ΔE = E_pred - E0
        L_dec = max(0.0, ΔE)


        # Combine
        total += LAMBDA_E*E_pred + LAMBDA_DELTAE*L_dec + LAMBDA_RECON*L_rec
        # total += LAMBDA_E*E_pred + L_rec
    end

    # Mean over batch + tiny weight decay
    return total / length(batch_data) + LAMBDA_WD * l2_params(θ)
end


println("\n" * "="^60)
println("Training Configuration:")
println("  Batch size: $BATCH_SIZE")
println("  Adam iterations: $MAX_ITER_ADAM (LR=$LR)")
println("  LBFGS iterations: $MAX_ITER_LBFGS")
println("="^60)

# Create minibatch function
function get_batch(samples, batch_size)
    idx = rand(1:length(samples), batch_size)
    return samples[idx]
end

# Loss wrapper for optimization
function loss_wrapper(θ)
    batch = get_batch(samples, BATCH_SIZE)
    return loss(θ, batch)
end

# Callback for monitoring training
iter = 0
loss_history = Float32[]

function callback(θ_cur, l)
    global iter
    iter += 1
    push!(loss_history, Float32(l))
    
    if iter % 50 == 0
        println("Iteration $iter: Loss = $(round(l, digits=4))")
    end
    return false
end

# Stage 1: ADAM optimization
println("\nStage 1: ADAM Optimization")
println("-"^30)

adtype = Optimization.AutoZygote()
optf = Optimization.OptimizationFunction((x, p) -> loss_wrapper(x), adtype)
optprob = Optimization.OptimizationProblem(optf, θ0)

@time result_adam = Optimization.solve(
    optprob,
    OptimizationOptimisers.ADAM(LR);
    callback = callback,
    maxiters = MAX_ITER_ADAM
)

θ_adam = result_adam.u
# println("Adam complete. Final loss: $(round(loss_wrapper(θ_adam), digits=4))")

# Stage 2: LBFGS fine-tuning
println("\nStage 2: LBFGS Fine-tuning")
println("-"^30)

# Reset iteration counter
iter = 0

# Create deterministic loss for LBFGS (use fixed subset)
function loss_lbfgs(θ, lo, hi)
    # Use first 8 samples for deterministic gradient
    batch = samples[lo:min(hi, length(samples))]
    return loss(θ, batch)
end

optf2 = Optimization.OptimizationFunction((x, p) -> loss_lbfgs(x, 1, 50), adtype)
optprob2 = Optimization.OptimizationProblem(optf2, θ_adam)

@time result_lbfgs_1 = Optimization.solve(
    optprob2,
    Optim.LBFGS(linesearch=LineSearches.BackTracking());
    callback = callback,
    maxiters = MAX_ITER_LBFGS
)

θ_lbfgs_1 = result_lbfgs_1.u


θ_final = result_lbfgs_1.u
final_loss = loss_lbfgs(θ_final, 1, 50)

println("LBFGS complete. Final loss: $(round(final_loss, digits=4))")

# ============================================================================
# EVALUATION
# ============================================================================

println("\n" * "="^60)
println("Training Complete!")
println("="^60)
println("  Total iterations: $(length(loss_history))")
println("  Initial loss: $(round(loss_history[1], digits=4))")
println("  Final loss: $(round(final_loss, digits=4))")
println("  Improvement: $(round(100*(1 - final_loss/loss_history[1]), digits=2))%")

# Test on a sample
test_idx = 1
test_sample = samples[test_idx]
u0_test = vec(test_sample.initial)
pred_test = predict(θ_final, u0_test)

# Decode to binary (top-2n points)
function decode_topk(x_vec, n; k=2*n)
    idx = sortperm(x_vec; rev=true)
    binary = zeros(Float32, length(x_vec))
    binary[idx[1:k]] .= 1.0f0
    return reshape(binary, n, n)
end

pred_binary = decode_topk(pred_test, n)
violations = compute_violations(vec(pred_binary), n)

println("\nTest Sample $test_idx:")
println("  Points placed: $(Int(sum(pred_binary)))")
println("  Violations: $(round(violations, digits=2))")

# Visualization
p1 = heatmap(test_sample.initial, title="Initial", color=:grays, aspect_ratio=:equal)
p2 = heatmap(reshape(pred_test, n, n), title="Predicted (continuous)", color=:grays, aspect_ratio=:equal)
p3 = heatmap(pred_binary, title="Decoded (binary)", color=:grays, aspect_ratio=:equal)
p4 = heatmap(test_sample.target, title="Target", color=:grays, aspect_ratio=:equal)

plot(p1, p2, p3, p4, layout=(2,2), size=(800, 800))

# Create unique id based on time
unique_id = Dates.format(now(), dateformat"yymmddHHMMSS")

# Make a directory in models which is unique_id followed by n
dirpath = "out/$(unique_id)_$(n)"
mkpath(dirpath)

savefig(dirpath * "/sample_n$(n).png")

# Save trained model
save_path = dirpath * "/energy_model.jld2"
JLD2.@save save_path θ_final st_nn n triplets loss_history
println("\n✓ Model saved to $save_path")
