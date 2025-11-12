"""
Evaluate and visualize UDE model on all available samples
"""

# ============================================================================
# IMPORTS
# ============================================================================

using HDF5, JLD2
using Lux, DifferentialEquations
using ComponentArrays
using Random
using Plots
using Zygote
using SciMLSensitivity
using Statistics
using StableRNGs

# ============================================================================
# DATA LOADING
# ============================================================================

"""
Load training data from HDF5 file
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

# ============================================================================
# MODEL SETUP
# ============================================================================

# Load data
data_path = "data/n_5.h5"
n, triplets, samples = load_data(data_path)
println("Loaded $(length(samples)) samples for $(n)×$(n) grid")

# Load trained model
model_path = "out/251107223125_5/lvstyle_one_net.jld2"
if !isfile(model_path)
    error("Model file not found: $model_path. Please train the model first.")
end

JLD2.@load model_path θ_trained st n triplets loss_hist
println("✓ Model loaded from $model_path")

# Recreate neural network architecture (must match training in test.jl)
F = n * n
const OUT = 2 + F
const U = Lux.Chain(
    Lux.Dense(2, 128, tanh),
    Lux.Dense(128, 128, tanh),
    Lux.Dense(128, OUT)
)
const _st = st

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

# Build unique lines for violation computation
function _build_lines(n::Int, triplets::AbstractMatrix{<:Integer})
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
    lines = Vector{Vector{Tuple{Int,Int}}}()
    for s in values(lp)
        if length(s) >= 3
            v = collect(s)
            sort!(v, by = x -> (x[2], x[1]))
            push!(lines, v)
        end
    end
    return lines
end

const LINES = _build_lines(Int(n), triplets)

# Continuous energies (match test.jl)
function E_viol(x::AbstractVector{<:Real}, n::Integer)
    xg = reshape(x, n, n)
    viol = 0.0f0
    @inbounds for line in LINES
        s = 0.0f0
        @inbounds for (i,j) in line
            s += xg[i,j]
        end
        z = s - 2.0f0
        viol += (z > 0.0f0 ? z*z : 0.0f0)
    end
    return viol
end

E_box(x) = sum(max.(0.0f0, -x).^2 .+ max.(0.0f0, x .- 1.0f0).^2)

const ETA_LOW  = 10.0f0
const ETA_HIGH = 10.0f0
function E_pts(x::AbstractVector{<:Real}, n::Integer)
    P = sum(x); two_n = 2.0f0*Float32(n)
    low  = max(0.0f0, two_n - P)
    high = max(0.0f0, P - two_n)
    ETA_LOW*low^2 + ETA_HIGH*high^2
end
L_box_pts(x, n) = E_box(x) + E_pts(x, n)

"""
LV-style latent UDE (u ∈ R^2):
  y = U(u; θ) ∈ R^{2+F}, x̂ = y[3:end]
  u̇₁ = -α * E_viol(x̂) + y[1]
  u̇₂ = -β * L_box_pts(x̂) + y[2]
"""
const ALPHA = 3.0f0
const BETA  = 1.0f0
function ude_dynamics!(du, u, θ, t)
    y, _ = Lux.apply(U, u, θ, _st)
    û1, û2 = y[1], y[2]
    x̂ = view(y, 3:length(y))
    du[1] = -ALPHA * E_viol(x̂, n) + û1
    du[2] = -BETA  * L_box_pts(x̂, n) + û2
    return nothing
end

@inline function decode_from_u(u::AbstractVector{<:Real}, θ)
    y, _ = Lux.apply(U, u, θ, _st)
    return view(y, 3:length(y))
end

"""
Predict using the trained model (embed grid → latent, integrate, decode x̂)
"""
tspan = (0.0f0, 6.0f0)
@inline function embed_u0(g::AbstractMatrix{<:Real})
    μ  = Float32(mean(g))
    δP = Float32((sum(g) - 2.0f0*n) / F)
    Float32[μ, δP]
end

function predict(θ, grid0; T=tspan[2])
    u0 = embed_u0(grid0)
    prob = ODEProblem(ude_dynamics!, u0, (0.0f0, Float32(T)), θ)
    sol = solve(prob, Tsit5();
                reltol=1f-3, abstol=1f-5,
                save_everystep=false,
                sensealg=InterpolatingAdjoint(autojacvec=ZygoteVJP()))
    uT = sol.u[end]
    x̂T = decode_from_u(uT, θ)
    return x̂T
end

"""
Decode continuous predictions to binary (top-k points)
"""
function decode_topk(x_vec, n; k=2*n)
    idx = sortperm(x_vec; rev=true)
    binary = zeros(Float32, length(x_vec))
    binary[idx[1:k]] .= 1.0f0
    return reshape(binary, n, n)
end

"""
Return the list of violating lines (each a vector of (i,j)) for a given binary grid vector
"""
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

"""
Compute collinearity violations (count discrete violations)
"""
compute_violations(x::AbstractVector, triplets, n) = Float32(length(violating_lines(x, n)))

# ============================================================================
# EVALUATE ALL SAMPLES
# ============================================================================

println("\n" * "="^60)
println("Evaluating Model on All Samples")
println("="^60)

# Store results
results = []

for (idx, sample) in enumerate(samples)
    pred = predict(θ_trained, sample.initial)
    pred_binary = decode_topk(pred, n)
    vlines = violating_lines(vec(pred_binary), n)
    violations = Float32(length(vlines))
    points_placed = Int(sum(pred_binary))
    
    push!(results, (
        idx = idx,
        pred = pred,
        pred_binary = pred_binary,
        vlines = vlines,
        violations = violations,
        points_placed = points_placed
    ))
    
    println("Sample $idx: Points=$(points_placed), Violations=$(Int(violations))")
end

# Summary statistics
all_violations = [r.violations for r in results]
all_points = [r.points_placed for r in results]

println("\n" * "="^60)
println("Summary Statistics")
println("="^60)
println("  Total samples evaluated: $(length(samples))")
println("  Average violations: $(round(mean(all_violations), digits=2))")
println("  Min violations: $(round(minimum(all_violations), digits=2))")
println("  Max violations: $(round(maximum(all_violations), digits=2))")
println("  Samples with 0 violations: $(sum(all_violations .== 0))")
println("  Average points placed: $(round(mean(all_points), digits=2))")

# ============================================================================
# VISUALIZATION
# ============================================================================

println("\nCreating visualizations...")

"""
Plot a grid with points placed
"""
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
        # endpoints after sorting by column then row (already sorted when built)
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

# Sort results by violations (best first)
sorted_indices = sortperm([r.violations for r in results])
best_10_indices = sorted_indices[1:min(10, length(results))]
best_results = results[best_10_indices]

println("\nBest 10 samples (by violations):")
for idx in best_10_indices
    r = results[idx]
    println("  Sample $(r.idx): Points=$(r.points_placed), Violations=$(Int(r.violations))")
end

# Create combined plot showing only the predicted outputs
plots_grid = []

for result in best_results
    p = plot_grid_with_points(result.pred_binary, 
                              "Sample $(result.idx): $(Int(result.violations)) violations";
                              overlay_lines=result.vlines)
    push!(plots_grid, p)
end

# Create grid layout (2 columns, 5 rows for 10 samples)
plot(plots_grid..., layout=(5, 2), size=(700, 1600))
savefig("best_10_samples.png")

println("✓ Best 10 samples saved as best_10_samples.png")

# ============================================================================
# RANDOM INITIALIZATION MODE (optional)
# ============================================================================

function random_init_best(N_RANDOM::Int=200; seed::Int=42, k::Int=10)
    rng = Xoshiro(seed)
    println("\n" * "="^60)
    println("Random Initialization Evaluation")
    println("="^60)

    runs = Any[]
    for r in 1:N_RANDOM
        # Random continuous init in [0,1]
        grid0 = rand(rng, Float32, n, n)
        pred = predict(θ_trained, grid0)
        pred_binary = decode_topk(pred, n)
        vlines_r = violating_lines(vec(pred_binary), n)
        vcount = length(vlines_r)
        push!(runs, (
            idx = r,
            grid0 = grid0,
            pred = pred,
            pred_binary = pred_binary,
            vlines = vlines_r,
            violations = Float32(vcount),
            points_placed = Int(sum(pred_binary))
        ))
    end

    if isempty(runs)
        println("No random runs evaluated.")
        return nothing, Any[]
    end

    ord = sortperm(runs; by = x -> Int(x.violations))
    topk = runs[ord[1:min(k, length(runs))]]

    best_random = topk[1]
    println("Best random init: run $(best_random.idx), Points=$(best_random.points_placed), Violations=$(Int(best_random.violations))")
    p_best = plot_grid_with_points(best_random.pred_binary,
                                   "Random best: $(Int(best_random.violations)) violations";
                                   overlay_lines=best_random.vlines)
    savefig(p_best, "random_best.png")
    println("✓ Random best saved as random_best.png")

    println("\nTop $(length(topk)) random runs (by violations):")
    for r in topk
        println("  Run $(r.idx): Points=$(r.points_placed), Violations=$(Int(r.violations))")
    end

    plots_grid2 = Any[]
    for r in topk
        push!(plots_grid2, plot_grid_with_points(r.pred_binary,
                          "Run $(r.idx): $(Int(r.violations)) violations";
                          overlay_lines=r.vlines))
    end
    cols = 2
    rows = ceil(Int, length(plots_grid2)/cols)
    plot(plots_grid2..., layout=(rows, cols), size=(700, 1600))
    outname = "random_best_$(length(topk)).png"
    savefig(outname)
    println("✓ Top $(length(topk)) random runs saved as $(outname)")

    return best_random, topk
end

# Configure and run random search (set N_RANDOM=0 to skip)
N_RANDOM = 1000
random_init_best(N_RANDOM; seed=2005)

println("\n" * "="^60)
println("Evaluation Complete!")
println("="^60)