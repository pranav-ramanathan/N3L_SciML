using HDF5, JLD2, Lux, DiffEqFlux, DifferentialEquations, Optimization, OptimizationOptimisers, OptimizationOptimJL, Random, Plots
using ComponentArrays, Dates, LineSearches
using ArgParse
using StableRNGs

f64(model) = Lux.f64(model)

rnd_seed = abs(rand(UInt32))
println("Random seed: $rnd_seed")
rng = StableRNG(rnd_seed)

function parse_commandline()
    s = ArgParseSettings(description = "Train Lotka-Volterra style UDE for N3L")
    @add_arg_table! s begin
        "--grid-size", "-n"
            help = "Grid size (n×n)"
            arg_type = Int64   
            default = 5
        "--data-path"
            help = "Path to HDF5 data file"
            arg_type = String
            default = "data/n_5.h5"
        "--adam-iters"
            help = "ADAM iterations"
            arg_type = Int64
            default = 500
        "--lbfgs-iters"
            help = "LBFGS iterations"
            arg_type = Int64
            default = 300
        "--batch-size"
            help = "Batch size"
            arg_type = Int64
            default = 4
        "--lr"
            help = "Learning rate"
            arg_type = Float64
            default = 0.01
        
        "--nn-size"
            help = "Hidden layer size of the network"
            arg_type = Int64
            default = 64
    end
    return parse_args(s)
end

args = parse_commandline()
n_grid = args["grid-size"]
data_path = args["data-path"]
BATCH_SIZE = args["batch-size"]
MAX_ITER_ADAM = args["adam-iters"]
MAX_ITER_LBFGS = args["lbfgs-iters"]
LR = args["lr"]
NN_SIZE = args["nn-size"]

function load_data(path::String)
    h5open(path, "r") do f
        n = Int64(read(f["n"]))
        triplets = read(f["triplets"])'
        ds = f["dataset"]
        samples = []
        for k in keys(ds)
            g = ds[k]
            push!(samples, (
                initial = Float64.(read(g["initial_grid"])'),
                target  = Float64.(read(g["target_grid"])'),
                mask    = Float64.(read(g["mask_top2n"]))')
            )
        end
        return Int64(n), triplets, samples
    end
end

n, triplets, samples = load_data(data_path)
@info "Loaded $(length(samples)) samples for $(n)×$(n) grid"
const F = n*n

function _build_lines(n::Int, triplets::AbstractMatrix{<:Integer})
    canon(A::Int, B::Int, C::Int) = begin
        g = gcd(gcd(abs(A), abs(B)), abs(C))
        if g > 0; A ÷= g; B ÷= g; C ÷= g; end
        if A < 0 || (A == 0 && B < 0) || (A == 0 && B == 0 && C < 0)
            A = -A; B = -B; C = -C
        end
        (A,B,C)
    end
    
    linekey_from_rc(i1, j1, i2, j2) = begin
        i1, j1, i2, j2 = Int(i1), Int(j1), Int(i2), Int(j2)
        A = j1 - j2
        B = i2 - i1
        C = i1*j2 - i2*j1
        canon(A,B,C)
    end
    
    lp = Dict{NTuple{3,Int}, Set{Tuple{Int,Int}}}()
    @inbounds for row in eachrow(triplets)
        i1,j1,i2,j2,i3,j3 = row
        k = linekey_from_rc(i1,j1,i2,j2)
        s = get!(lp, k, Set{Tuple{Int,Int}}())
        push!(s, (Int(i1),Int(j1)), (Int(i2),Int(j2)), (Int(i3),Int(j3)))
    end
    
    lines = Vector{Vector{Tuple{Int,Int}}}()
    for s in values(lp)
        if length(s) >= 3
            v = sort(collect(s), by = x -> (x[2], x[1]))
            push!(lines, v)
        end
    end
    return lines
end

const LINES = _build_lines(n, triplets)

const NN_xij = Lux.Chain(
    Lux.Dense(F, NN_SIZE, tanh),
    Lux.Dense(NN_SIZE, F)
)



p_xij, st_xij = Lux.setup(rng, NN_xij)

θ0 = ComponentArrays.ComponentArray{Float64}(p_xij)

function ude_dynamics!(du, u, p, t)
    # Learned correction η(t)
    η = NN_xij(u, p, st_xij)[1]

    penalty_strength = 10.0

    # Initialize du with learned term + box constraints (no separate zero-fill needed)
    @inbounds for i in eachindex(du)
        lower = (u[i] < 0) * u[i]
        upper = (u[i] > 1) * (u[i] - 1)
        box = penalty_strength * (lower + upper)
        du[i] = η[i] - box
    end
    
    # Add energy gradient: -∂E/∂x from violations
    u_grid = reshape(u, n, n)
    for line in LINES
        line_sum = sum(u_grid[i, j] for (i, j) in line)
        
        if line_sum > 2.0
            violation = line_sum - 2.0
            ΔE = 2.0 * violation
            
            for (i, j) in line
                idx = (j - 1) * n + i
                du[idx] -= ΔE
            end
        end
    end

    return nothing
end


function predict(θ, u0; t_end=5.0)
    prob = ODEProblem{true}(ude_dynamics!, u0, (0.0, t_end), θ)

    sol = solve(prob, Vern7();
                reltol=1e-3, abstol=1e-5,
                save_everystep=false,
                sensealg=InterpolatingAdjoint(autojacvec=ZygoteVJP()))

    return sol.u[end]
end

function violations(x::AbstractVector, n::Int64)
    x_grid = reshape(x, n, n)
    viol = 0.0
    for line in LINES
        line_sum = sum(x_grid[i, j] for (i, j) in line)
        if line_sum > 2.0
            viol += line_sum - 2.0
        end
    end
    return viol
end

function box(x::AbstractVector)
    sum(max.(0.0, -x).^2 .+ max.(0.0, x .- 1.0).^2)
end

function points(x::AbstractVector, n::Int64)
    P = sum(x)
    target = 2.0 * Float64(n)
    10.0 * (P - target)^2
end

function total_energy(x::AbstractVector, n::Int64)
    return violations(x, n) + box(x) + points(x, n)
end

const LAMBDA_E = 1.0
const LAMBDA_DELTA = 0.5
const LAMBDA_RECON = 0.3
const LAMBDA_WD = 1e-4

# Energy function coefficients
const BETA_VIOLATION = 1.0
const GAMMA_BOX = 1.0
const ETA_LOW = 10.0
const ETA_HIGH = 10.0

# Tiny L2 over params (ComponentArray-friendly)
function l2_params(θ)
    # θ is a ComponentVector; iterate its flattened entries directly
    s = 0.0
    @inbounds @simd for v in θ
        s += Float64(v) * Float64(v)
    end
    return s
end

function grid_energy(x::AbstractVector{<:Real}, n::Int64)
    # (1) Violation penalty
    viol = violations(x, n)

    # (2) Box penalty (same as before)
    box = sum(max.(0.0, -x).^2 .+ max.(0.0, x .- 1.0).^2)

    # (3) Point equilibrium: penalty on deviation from 2n
    P = sum(x)
    diff_low  = max(0.0, 2.0 * Float64(n) - P)  # too few points
    diff_high = max(0.0, P - 2.0 * Float64(n))  # too many points

    # You can use asymmetric weights if you want to penalize surplus more heavily

    points_t = ETA_LOW * diff_low^2 + ETA_HIGH * diff_high^2

    return BETA_VIOLATION * viol + GAMMA_BOX * box + points_t
end

function loss(θ, batch_data)
    total = 0.0

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
        weight = 1.0 .+ 5.0 .* mask
        L_rec  = sum(weight .* (pred .- target).^2)

        ΔE = E_pred - E0
        L_dec = max(0.0, ΔE)


        # Combine
        total += LAMBDA_E*E_pred + LAMBDA_DELTA*L_dec + LAMBDA_RECON*L_rec
        # total += LAMBDA_E*E_pred + L_rec
    end

    # Mean over batch + tiny weight decay
    return total / length(batch_data) + LAMBDA_WD * l2_params(θ)
end

function get_batch(samples, batch_size)
    idx = rand(1:length(samples), batch_size)
    return samples[idx]
end

iter = 0
loss_history = Float64[]

function callback(θ_cur, l)
    global iter
    iter += 1
    push!(loss_history, Float64(l))
    
    if iter % 50 == 0
        println("Iteration $iter: Loss = $(round(l, digits=4))")
    end
    return false
end

println("Stage 1: ADAM Optimization")
println("-"^70)
adtype = Optimization.AutoZygote()
optf = Optimization.OptimizationFunction(
    (x, p) -> loss(x, get_batch(samples, BATCH_SIZE)), 
    adtype
)

optprob = Optimization.OptimizationProblem(optf, θ0)

@time res1 = Optimization.solve(
    optprob, 
    OptimizationOptimisers.Adam(LR); 
    maxiters=MAX_ITER_ADAM, 
    callback=callback
)

println("\n✓ ADAM complete. Loss: $(round(loss_history[end], digits=4))")

println("\nStage 2: LBFGS Optimization")
println("-"^70)
iter = 0
fixed_batch = get_batch(samples, BATCH_SIZE * 2)
optf2 = Optimization.OptimizationFunction(
    (x, p) -> loss(x, fixed_batch), 
    adtype
)
optprob2 = Optimization.OptimizationProblem(optf2, res1.u)

@time res2 = Optimization.solve(
    optprob2, 
    Optim.LBFGS(linesearch=LineSearches.BackTracking()); 
    maxiters=MAX_ITER_LBFGS, 
    callback=callback
)

θ_trained = res2.u
final_loss = loss(θ_trained, get_batch(samples, 20))
println("\n✓ LBFGS complete. Final loss: $(round(final_loss, digits=4))")

println("\n" * "="^60)
println("Training Complete!")
println("="^60)
println("  Total iterations: $(length(loss_history))")
println("  Initial loss: $(round(loss_history[1], digits=4))")
println("  Final loss: $(round(final_loss, digits=4))")
println("  Improvement: $(round(100*(1 - final_loss/loss_history[1]), digits=2))%")

# Decode to binary (top-2n points)
function decode_topk(x_vec, n; k=2*n)
    idx = sortperm(x_vec; rev=true)
    binary = zeros(Float64, length(x_vec))
    binary[idx[1:k]] .= 1.0
    return reshape(binary, n, n)
end

# Function to get violating lines
function violating_lines(x::AbstractVector, n)
    x_grid = reshape(x, n, n)
    viols = Vector{Vector{Tuple{Int,Int}}}()
    @inbounds for line in LINES
        points_on_line = 0
        for (i, j) in line
            if x_grid[i, j] > 0.5
                points_on_line += 1
            end
        end
        if points_on_line >= 3
            push!(viols, line)
        end
    end
    return viols
end

# Plotting function (matching viz.jl style)
function plot_grid_with_points(grid_matrix, title_text=""; show_grid=true, overlay_lines=Vector{Vector{Tuple{Int,Int}}}())
    # Create base plot
    p = plot(aspect_ratio=:equal, legend=false, title=title_text,
             xlims=(0.5, n+0.5), ylims=(0.5, n+0.5),
             framestyle=:box, size=(300, 300),
             xticks=1:n, yticks=1:n)
    
    # Draw grid lines
    if show_grid
        for i in 0.5:(n+0.5)
            plot!([i, i], [0.5, n+0.5], color=:lightgray, linewidth=0.5, label="")
            plot!([0.5, n+0.5], [i, i], color=:lightgray, linewidth=0.5, label="")
        end
    end

    # Overlay violating lines (in red)
    for line in overlay_lines
        first_pt = line[1]; last_pt = line[end]
        xs = [first_pt[2], last_pt[2]]  # j
        ys = [n - first_pt[1] + 1, n - last_pt[1] + 1]  # inverted i
        plot!(xs, ys, color=:red, linewidth=2, alpha=0.6, label="")
    end
    
    # Plot points where grid_matrix has value 1
    for i in 1:n
        for j in 1:n
            if grid_matrix[i, j] > 0.5  # Binary threshold
                scatter!([j], [n - i + 1], color=:black, markersize=12, label="")
            end
        end
    end
    
    return p
end

# Evaluate best 10 samples from dataset
println("\n" * "="^60)
println("Evaluating Best 10 Samples from Dataset")
println("="^60)

results = []
for (i, sample) in enumerate(samples)
    u0 = vec(sample.initial)
    pred = predict(θ_trained, u0)
    pred_binary = decode_topk(pred, n)
    
    vlines = violating_lines(vec(pred_binary), n)
    viols = Float64(length(vlines))
    energy = total_energy(vec(pred_binary), n)
    
    push!(results, (idx=i, violations=viols, energy=energy, pred=pred, pred_binary=pred_binary, vlines=vlines, initial=sample.initial, target=sample.target))
end

# Sort by violations (lower is better)
sort!(results, by=r->r.violations)

println("\nTop 10 Results (by violations):")
for (rank, r) in enumerate(results[1:min(10, length(results))])
    println("  $rank. Sample $(r.idx): Violations=$(Int(r.violations)), Energy=$(round(r.energy, digits=2))")
end

# Visualize best 10
plots_best = []
for (rank, r) in enumerate(results[1:min(10, length(results))])
    p = plot_grid_with_points(r.pred_binary, 
                              "Sample $(r.idx): $(Int(r.violations)) violations";
                              overlay_lines=r.vlines)
    push!(plots_best, p)
end

plot(plots_best..., layout=(5, 2), size=(700, 1600))

# Create unique id based on time
unique_id = Dates.format(now(), dateformat"yymmddHHMMSS")

# Make a directory in models which is unique_id followed by n
dirpath = "out/$(unique_id)_n$(n)"
mkpath(dirpath)

savefig(dirpath * "/best_10_samples.png")
println("\n✓ Best 10 visualization saved")

# Random initialization evaluation
println("\n" * "="^60)
println("Evaluating Random Initializations")
println("="^60)

random_results = []
for i in 1:20
    # Random initialization with approximately 2n points
    u0_random = rand(Float64, F) .* 0.5
    u0_random ./= sum(u0_random)
    u0_random .*= 2.0 * Float64(n)
    
    pred = predict(θ_trained, u0_random)
    pred_binary = decode_topk(pred, n)
    
    vlines = violating_lines(vec(pred_binary), n)
    viols = Float64(length(vlines))
    energy = total_energy(vec(pred_binary), n)
    
    push!(random_results, (idx=i, violations=viols, energy=energy, pred=pred, pred_binary=pred_binary, vlines=vlines, initial=u0_random))
end

# Sort by violations
sort!(random_results, by=r->r.violations)

println("\nTop 10 Random Results (by violations):")
for (rank, r) in enumerate(random_results[1:10])
    println("  $rank. Random $(r.idx): Violations=$(Int(r.violations)), Energy=$(round(r.energy, digits=2))")
end

# Visualize best 10 random
plots_random = []
for (rank, r) in enumerate(random_results[1:10])
    p = plot_grid_with_points(r.pred_binary, 
                              "Random $(r.idx): $(Int(r.violations)) violations";
                              overlay_lines=r.vlines)
    push!(plots_random, p)
end

plot(plots_random..., layout=(5, 2), size=(700, 1600))
savefig(dirpath * "/best_10_random.png")
println("✓ Random initialization visualization saved")

# Save trained model
save_path = dirpath * "/energy_model.jld2"
JLD2.@save save_path θ_trained n triplets loss_history
println("\n✓ Model saved to $save_path")