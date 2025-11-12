"""
Evaluate and visualize UDE model trained with ude_combined.jl
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
using ArgParse

# ============================================================================
# ARGUMENT PARSING
# ============================================================================

function parse_commandline()
    s = ArgParseSettings(description = "Visualize trained UDE model")
    @add_arg_table! s begin
        "--model-path"
            help = "Path to trained model JLD2 file"
            arg_type = String
            required = true
        "--data-path"
            help = "Path to HDF5 data file"
            arg_type = String
            default = "data/n_5.h5"
        "--random-runs"
            help = "Number of random initialization runs"
            arg_type = Int
            default = 200

        "--nn-size"
            help = "Hidden layer size of the network"
            arg_type = Int
            default = 64
    end
    return parse_args(s)
end

args = parse_commandline()
model_path = args["model-path"]
data_path = args["data-path"]
N_RANDOM = args["random-runs"]
NN_SIZE = args["nn-size"]
# ============================================================================
# DATA LOADING
# ============================================================================

"""
Load training data from HDF5 file
"""
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
                target = Float64.(read(g["target_grid"])'),
                mask = Float64.(read(g["mask_top2n"]))'
            ))
        end
        
        return n, triplets, samples
    end
end

# Load data
n_data, triplets, samples = load_data(data_path)
println("Loaded $(length(samples)) samples for $(n_data)×$(n_data) grid")

# Load trained model
if !isfile(model_path)
    error("Model file not found: $model_path")
end

JLD2.@load model_path θ_trained n triplets loss_history
println("✓ Model loaded from $model_path")

# Verify grid size matches
if n != n_data
    error("Grid size mismatch: model trained on $(n)×$(n), data is $(n_data)×$(n_data)")
end

# ============================================================================
# MODEL SETUP
# ============================================================================

const F = n * n

# Recreate neural network architecture (must match training)
const NN_xij = Lux.Chain(
    Lux.Dense(F, NN_SIZE, tanh),
    Lux.Dense(NN_SIZE, F)
)

rng = StableRNG(123)
p_xij, st_xij = Lux.setup(rng, NN_xij)
const _st_xij = st_xij

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
        i1, j1, i2, j2 = Int(i1), Int(j1), Int(i2), Int(j2)
        A = j1 - j2
        B = i2 - i1
        C = i1*j2 - i2*j1
        canon(A, B, C)
    end
    
    lp = Dict{NTuple{3,Int}, Set{Tuple{Int,Int}}}()
    @inbounds for row in eachrow(triplets)
        i1, j1, i2, j2, i3, j3 = row
        k = linekey_from_rc(i1, j1, i2, j2)
        s = get!(lp, k, Set{Tuple{Int,Int}}())
        push!(s, (Int(i1), Int(j1)), (Int(i2), Int(j2)), (Int(i3), Int(j3)))
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

const LINES = _build_lines(Int(n), triplets)

# ============================================================================
# UDE DYNAMICS (must match training)
# ============================================================================

function ude_dynamics!(du, u, p, t)
    # Learned correction η(t)
    η = NN_xij(u, p, _st_xij)[1]

    penalty_strength = 10.0

    # Initialize du with learned term + box constraints
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

# ============================================================================
# EVALUATION FUNCTIONS
# ============================================================================

"""
Decode continuous predictions to binary (top-k points)
"""
function decode_topk(x_vec, n; k=2*n)
    idx = sortperm(x_vec; rev=true)
    binary = zeros(Float64, length(x_vec))
    binary[idx[1:k]] .= 1.0
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

# ============================================================================
# EVALUATE ALL SAMPLES
# ============================================================================

println("\n" * "="^60)
println("Evaluating Model on All Samples")
println("="^60)

# Store results
results = []

for (idx, sample) in enumerate(samples)
    u0 = vec(sample.initial)
    pred = predict(θ_trained, u0)
    pred_binary = decode_topk(pred, n)
    vlines = violating_lines(vec(pred_binary), n)
    violations = Float64(length(vlines))
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
# VISUALIZATION - BEST 10 FROM DATASET
# ============================================================================

println("\nCreating visualizations...")

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

# Create grid layout (5 rows, 2 columns for 10 samples)
plot(plots_grid..., layout=(5, 2), size=(700, 1600))
savefig("best_10_samples_combined.png")

println("✓ Best 10 samples saved as best_10_samples_combined.png")

# ============================================================================
# RANDOM INITIALIZATION MODE
# ============================================================================

println("\n" * "="^60)
println("Random Initialization Evaluation")
println("="^60)

runs = Any[]
for r in 1:N_RANDOM
    # Random continuous init in [0,1]
    u0_random = rand(Float64, F) .* 0.5
    u0_random ./= sum(u0_random)
    u0_random .*= 2.0 * Float64(n)
    
    pred = predict(θ_trained, u0_random)
    pred_binary = decode_topk(pred, n)
    vlines_r = violating_lines(vec(pred_binary), n)
    vcount = length(vlines_r)
    push!(runs, (
        idx = r,
        pred = pred,
        pred_binary = pred_binary,
        vlines = vlines_r,
        violations = Float64(vcount),
        points_placed = Int(sum(pred_binary))
    ))
end

if isempty(runs)
    println("No random runs evaluated.")
else
    ord = sortperm(runs; by = x -> Int(x.violations))
    topk = runs[ord[1:min(10, length(runs))]]

    best_random = topk[1]
    println("Best random init: run $(best_random.idx), Points=$(best_random.points_placed), Violations=$(Int(best_random.violations))")
    
    println("\nTop 10 random runs (by violations):")
    for r in topk
        println("  Run $(r.idx): Points=$(r.points_placed), Violations=$(Int(r.violations))")
    end

    plots_grid2 = Any[]
    for r in topk
        push!(plots_grid2, plot_grid_with_points(r.pred_binary,
                          "Run $(r.idx): $(Int(r.violations)) violations";
                          overlay_lines=r.vlines))
    end
    
    plot(plots_grid2..., layout=(5, 2), size=(700, 1600))
    savefig("random_best_10_combined.png")
    println("✓ Top 10 random runs saved as random_best_10_combined.png")
    
    # Random statistics
    random_violations = [r.violations for r in runs]
    println("\nRandom Initialization Statistics:")
    println("  Average violations: $(round(mean(random_violations), digits=2))")
    println("  Min violations: $(round(minimum(random_violations), digits=2))")
    println("  Max violations: $(round(maximum(random_violations), digits=2))")
    println("  Runs with 0 violations: $(sum(random_violations .== 0))")
end

println("\n" * "="^60)
println("Evaluation Complete!")
println("="^60)

