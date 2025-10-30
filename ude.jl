using HDF5
using Lux, DiffEqFlux, DifferentialEquations
using Optimization, OptimizationOptimJL, OptimizationOptimisers
using ComponentArrays
using Statistics, Random, StableRNGs
using Plots
using Zygote

data_path = "data/n_5.jld2"

const SampleNT = NamedTuple{
    (:initial_grid, :target_grid, :mask_top2n,
     :num_points, :num_violations, :deficit, :score, :n),
    Tuple{Matrix{Float32}, Matrix{Float32}, Matrix{Float32},
          Int32, Int32, Int32, Float32, Int32}
}

const SampleDict = Dict{String, Any}

function load_data(path::AbstractString)
    h5open(path, "r") do f
        n = Int32(read(f["n"]))
        triplets = read(f["triplets"])

        ds = f["dataset"]
        ks = sort(collect(keys(ds)), by = x -> parse(Int, x))
        samples = Vector{SampleNT}(undef, length(ks))
        for (i, k) in enumerate(ks)
            g = ds[k]
            samples[i] = (
                initial_grid = read(g["initial_grid"]),
                target_grid = read(g["target_grid"]),
                mask_top2n = read(g["mask_top2n"]),
                num_points = Int32(read(g["num_points"])),
                num_violations = Int32(read(g["num_violations"])),
                deficit = Int32(read(g["deficit"])),
                score = Float32(read(g["score"])),
                n = n,
            )
        end
        return (n, triplets, samples)
    end
end



n, triplets, samples = load_data(data_path)

_get(s, sym::Symbol) = getproperty(s, sym)
_get(s::AbstractDict, sym::Symbol) = s[String(sym)]

function minibatch(samples::Vector{SampleNT}, batch_size::Int, rng=default_rng())
    N = length(samples)
    @assert 1 <= batch_size <= N "B must be between 1 and $(N)"

    idxs = randperm(rng, N)[1:batch_size]

    n = size(_get(samples[1], :initial_grid), 1)
    F = n * n

    u0_batch = Array{Float32}(undef, batch_size, F)
    target_batch = Array{Float32}(undef, batch_size, F)
    mask2n_batch = Array{Float32}(undef, batch_size, F)

    score          = Array{Float32}(undef, batch_size)
    num_points     = Array{Int32}(undef, batch_size)
    num_violations = Array{Int32}(undef, batch_size)
    deficit        = Array{Int32}(undef, batch_size)

    @inbounds for (i, idx) in enumerate(idxs)
        s = samples[idx]
        u0_batch[i, :]     = vec(_get(s, :initial_grid))
        target_batch[i, :] = vec(_get(s, :target_grid))
        mask2n_batch[i, :] = vec(_get(s, :mask_top2n))

        score[i]          = Float32(_get(s, :score))
        num_points[i]     = Int32(_get(s, :num_points))
        num_violations[i] = Int32(_get(s, :num_violations))
        deficit[i]        = Int32(_get(s, :deficit))
    end

    return (
        u0_batch=u0_batch,
        target_batch=target_batch,
        mask2n_batch=mask2n_batch,
        score=score,
        num_points=num_points,
        num_violations=num_violations,
        deficit=deficit,
        indices=idxs,
    )    
end

rng = Random.default_rng()

function init_network(rng::AbstractRNG, F::Int32)
    NN_add = Chain(Dense(F, 128, relu), Dense(128, F))
    p_add, st_add = Lux.setup(rng, NN_add)

    NN_block = Chain(Dense(F, 128, relu), Dense(128, F))
    p_block, st_block = Lux.setup(rng, NN_block)

    NN_box = Chain(Dense(F, 64, relu), Dense(64, F))
    p_box, st_box = Lux.setup(rng, NN_box)

    θ0 = ComponentArray((
            add = p_add, 
            block = p_block, 
            box = p_box
        ))
    
    st = (
        add = st_add,
        block = st_block,
        box = st_box,
    )

    return NN_add, NN_block, NN_box, θ0, st
end

NN_add, NN_block, NN_box, θ0, st = init_network(rng, n * n)

function dudt_pred!(du, u, p, t)
    raw_add,   _ = Lux.apply(NN_add,   u, p.add,   st.add)
    raw_block, _ = Lux.apply(NN_block, u, p.block, st.block)
    raw_box,   _ = Lux.apply(NN_box,   u, p.box,   st.box)

    add_term   = max.(raw_add,  0f0)
    block_term = max.(raw_block, 0f0)

    # Non-mutating "box" correction: push back toward [0,1]
    mag   = abs.(raw_box)
    above = clamp.(u .- 1f0, 0f0, typemax(Float32))     # positive when u>1
    below = clamp.(0f0 .- u, 0f0, typemax(Float32))     # positive when u<0
    box_term = -mag .* above .+ mag .* below            # push down / up

    du .= add_term .- block_term .+ box_term
end

# ================================================================
# 4. Example ODE solve
# ================================================================

F = n * n
u0 = vec(samples[1].initial_grid)
tspan = (0.0f0, 5.0f0)

prob = ODEProblem(dudt_pred!, u0, tspan, θ0)

sol = solve(prob, Tsit5(); saveat=0.0f0:0.5f0:5.0f0,
            reltol=1f-3, abstol=1f-5)

initial_grid = reshape(u0, n, n)
final_grid = reshape(sol.u[end], n, n)

plot(
    heatmap(initial_grid, title="Initial Grid", color=:grays, aspect_ratio=:equal),
    heatmap(final_grid, title="Final Grid", color=:grays, aspect_ratio=:equal),
    layout=(1,2)
)

# ================================================================
# 5) Triplets utils, prediction, loss, training, evaluation
# ================================================================

# ---- Triplets to (num_triplets, 6) Int32 with 1-based indices ----
function normalize_triplets(triplets_raw)
    T = ndims(triplets_raw) == 2 ? copy(triplets_raw) : reduce(hcat, triplets_raw)
    # Expect either (6, M) or (M, 6). Make it (M, 6).
    if size(T, 1) == 6 && size(T, 2) > 6
        T = permutedims(T)  # (M, 6)
    end
    T = Int32.(T) .+ 1  # Python saved 0-based; Julia is 1-based
    return T
end

T_trip = normalize_triplets(triplets)  # use loaded `triplets` from earlier

# ---- Collinearity soft-residual on a continuous grid vector u (size F) ----
@inline function triplet_residual(u_vec::AbstractVector{<:Real}, T::AbstractMatrix{<:Integer}, n::Integer)
    x = reshape(u_vec, n, n)
    s = 0.0f0
    @inbounds for r in axes(T, 1)
        i1 = T[r, 1]; j1 = T[r, 2]
        i2 = T[r, 3]; j2 = T[r, 4]
        i3 = T[r, 5]; j3 = T[r, 6]
        y = (x[i1, j1] + x[i2, j2] + x[i3, j3]) - 2f0
        if y > 0f0
            s += y*y
        end
    end
    return s
end

# ---- Box penalty to keep outputs in [0,1] ----
@inline function box_penalty(u_vec::AbstractVector{<:Real})
    lo = max.(0f0 .- Float32.(u_vec), 0f0)
    hi = max.(Float32.(u_vec) .- 1f0, 0f0)
    return sum(lo .* lo) + sum(hi .* hi)
end

# ---- One-sample forward solve to final time ----
const SENSEALG = InterpolatingAdjoint(autojacvec = ZygoteVJP())

function forward_solve_final(u0_vec::Vector{Float32}, θ, t_end::Float32)
    prob = ODEProblem(dudt_pred!, u0_vec, (0f0, t_end), θ)
    sol = solve(prob, Tsit5();
                reltol=1f-3, abstol=1f-5,
                save_everystep=false,
                sensealg=SENSEALG)
    return Array(sol.u[end])::Vector{Float32}
end

# ---- Batched forward solve ----
function predict_batch(θ, batch; t_end::Float32 = 5f0)
    u0_batch = batch.u0_batch
    B, F = size(u0_batch)
    # Avoid mutation by collecting results and stacking
    pred_rows = [forward_solve_final(vec(u0_batch[i, :]), θ, t_end)' for i in 1:B]
    return vcat(pred_rows...)
end

# ---- Weighted MSE + penalties ----
struct LossConfig
    λ_mask::Float32
    λ_box::Float32
    λ_size::Float32
    λ_trip::Float32
    t_end::Float32
end

const LOSS_CFG = LossConfig(3f0, 0.05f0, 0.02f0, 0.5f0, 5f0)

function batch_loss(θ, batch, n::Int, T::AbstractMatrix{<:Integer};
                    cfg::LossConfig = LOSS_CFG)
    preds = predict_batch(θ, batch; t_end=cfg.t_end)
    targets = batch.target_batch
    masks   = batch.mask2n_batch

    B, F = size(preds)
    # mask-weighted MSE
    w = 1f0 .+ cfg.λ_mask .* masks
    mse = 0f0
    box = 0f0
    szp = 0f0
    trp = 0f0

    @inbounds for i in 1:B
        p = view(preds, i, :)
        t = view(targets, i, :)
        w_i = view(w, i, :)

        # weighted MSE per sample
        mse += sum(w_i .* (p .- t).^2)

        # box penalty
        box += box_penalty(p)

        # size penalty: push total mass toward 2n
        szp += (sum(p) - 2f0*n)^2

        # triplet residual (soft violation)
        trp += triplet_residual(p, T, n)
    end
    mse /= B; box /= B; szp /= B; trp /= B

    return mse + cfg.λ_box*box + cfg.λ_size*szp + cfg.λ_trip*trp
end

# ---- Minibatch sampler wrapper (uses your minibatch function) ----
mutable struct DataStream
    samples::Vector{SampleNT}
    batch_size::Int
    rng::AbstractRNG
end

function next_batch!(ds::DataStream)
    return minibatch(ds.samples, ds.batch_size, ds.rng)
end

# Mark data sampling functions as non-differentiable
Zygote.@nograd minibatch
Zygote.@nograd next_batch!

# ================================================================
# 6) Training
# ================================================================

# Training config
BATCH_SIZE   = 16
MAXITERS_ADAM = 1000
LR_ADAM       = 1f-2
MAXITERS_LBFGS = 1000

ds = DataStream(samples, BATCH_SIZE, StableRNG(42))

# Parameter vector (already created earlier as θ0)
θ = θ0                      # ComponentArray with fields add, block, box
adtype = Optimization.AutoZygote()

# Define OptimizationFunction over θ with fresh minibatches each call
function loss_wrapper(θ_vec)
    batch = Zygote.ignore() do
        next_batch!(ds)   # draws & mutates arrays safely, outside AD
    end
    return batch_loss(θ_vec, batch, Int(n), T_trip; cfg=LOSS_CFG)
end

optf = Optimization.OptimizationFunction((x, p) -> loss_wrapper(x), adtype)
optprob = Optimization.OptimizationProblem(optf, θ)

# Callback & history
loss_hist = Float32[]

function cb(θ_cur, l)
    push!(loss_hist, Float32(l))
    if length(loss_hist) % 50 == 0
        @info "iter=$(length(loss_hist)) loss=$(round(l, digits=4))"
    end
    return false
end

# ---- Stage 1: ADAM ----
res_adam = Optimization.solve(
    optprob,
    OptimizationOptimisers.ADAM(LR_ADAM);
    callback = cb,
    maxiters = MAXITERS_ADAM
)

θ_adam = res_adam.u

# ---- Stage 2: LBFGS fine-tune (optional) ----
optprob2 = Optimization.OptimizationProblem(optf, θ_adam)
res_lbfgs = Optimization.solve(
    optprob2,
    OptimizationOptimJL.LBFGS();  # from Optim.jl via OptimizationOptimJL
    callback = cb,
    maxiters = MAXITERS_LBFGS
)

θ_trained = res_lbfgs.u
@info "Training complete. Final loss = $(round(loss_wrapper(θ_trained), digits=4))"

# ================================================================
# 7) Evaluation helpers
# ================================================================

# top-k decode to exactly 2n points
function decode_topk(x_vec::AbstractVector{<:Real}, n::Int; k::Int = 2n)
    idx = sortperm(x_vec; rev=true)
    bin = zeros(Float32, length(x_vec))
    @inbounds for i in 1:min(k, length(idx))
        bin[idx[i]] = 1f0
    end
    return reshape(bin, n, n)
end

# hard violation count
function count_violations(bin_grid::AbstractMatrix{<:Real}, T::AbstractMatrix{<:Integer})
    v = 0
    @inbounds for r in axes(T, 1)
        i1 = T[r,1]; j1 = T[r,2]
        i2 = T[r,3]; j2 = T[r,4]
        i3 = T[r,5]; j3 = T[r,6]
        s = bin_grid[i1,j1] + bin_grid[i2,j2] + bin_grid[i3,j3]
        v += (s ≥ 3) ? 1 : 0
    end
    return v
end

# quick eval on a random small batch
function evaluate_random(ds::DataStream, θ_eval; t_end::Float32 = LOSS_CFG.t_end)
    batch = minibatch(ds.samples, min(8, length(ds.samples)), ds.rng)
    preds = predict_batch(θ_eval, batch; t_end)
    B, F = size(preds)
    nn = Int(sqrt(F))
    total_v = 0
    total_pts = 0
    @inbounds for i in 1:B
        bin = decode_topk(view(preds, i, :), nn)
        total_v += count_violations(bin, T_trip)
        total_pts += round(Int, sum(bin))
    end
    return (; avg_points = total_pts / B, avg_violations = total_v / B)
end

metrics = evaluate_random(ds, θ_trained)
@info "Eval: avg_points=$(metrics.avg_points), avg_violations=$(metrics.avg_violations)"

# ================================================================
# 8) Visualize a sample prediction vs target
# ================================================================
using StatsBase: sample

# pick a random sample
rng_vis = StableRNG(7)
idx = rand(rng_vis, 1:length(samples))
s = samples[idx]

u0_vec = vec(s.initial_grid)
pred_vec = forward_solve_final(u0_vec, θ_trained, LOSS_CFG.t_end)
pred_bin = decode_topk(pred_vec, Int(n))

p_init = heatmap(reshape(u0_vec, n, n), title="Initial Grid", color=:grays, aspect_ratio=:equal)
p_pred = heatmap(reshape(pred_vec, n, n), title="Final (continuous)", color=:grays, aspect_ratio=:equal)
p_bin  = heatmap(pred_bin, title="Decoded Top-2n (binary)", color=:grays, aspect_ratio=:equal)
p_tgt  = heatmap(s.target_grid, title="Target Grid", color=:grays, aspect_ratio=:equal)

plot(p_init, p_pred, p_bin, p_tgt, layout=(2,2))
println("Violations (decoded): ", count_violations(pred_bin, T_trip))
println("Points (decoded): ", sum(pred_bin))
