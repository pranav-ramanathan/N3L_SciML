using JuMP, Gurobi
using Random
using LinearAlgebra, Random
import MathOptInterface as MOI
using JLD2
using Plots


"""
Symmetry types for the no-three-in-line problem.

Each symmetry type defines how points must be selected to maintain symmetry.
Compound symmetries like `BOTH_AXES` combine multiple individual transformations.

# Values
- `NONE`: No symmetry constraints
- `HORIZONTAL`: Horizontal reflection symmetry
- `VERTICAL`: Vertical reflection symmetry
- `BOTH_AXES`: Both horizontal and vertical symmetry
- `DIAGONAL`: Main diagonal symmetry (top-left to bottom-right)
- `ANTI_DIAGONAL`: Anti-diagonal symmetry (top-right to bottom-left)
- `BOTH_DIAGONALS`: Both diagonal symmetries
- `ROTATIONAL_90`: 90° rotational symmetry (includes 270°)
- `ROTATIONAL_180`: 180° rotational symmetry
- `ROTATIONAL_270`: 270° rotational symmetry
- `ALL`: All symmetry types combined
"""
@enum SymmetryType NONE HORIZONTAL VERTICAL BOTH_AXES DIAGONAL BOTH_DIAGONALS ANTI_DIAGONAL ROTATIONAL_90 ROTATIONAL_180 ROTATIONAL_270 ALL

function generate_grid_points(n::Int)
    return [(i, j) for i in 1:n for j in 1:n]
end

"""
normalise_line_equation(a, b, c)

Normalise the coefficients of a line equation ax + by = c to a standard form.

The function divides by the GCD of the absolute values and ensures the first
non-zero coefficient is positive.
"""
function normalise_line_equation(a::Int, b::Int, c::Int)
    g = gcd(gcd(abs(a), abs(b)), abs(c))
    if g > 0
        a, b, c = a ÷ g, b ÷ g, c ÷ g
    end
    if a < 0 || (a == 0 && b < 0)
        a, b, c = -a, -b, -c
    end
    return (a, b, c)
end

function get_line_through_points(p1::Tuple{Int, Int}, p2::Tuple{Int, Int})
    x1, y1 = p1
    x2, y2 = p2
    a = y1 - y2
    b = x2 - x1
    c = x1 * y2 - y1 * x2
    return normalise_line_equation(a, b, c)
end

"""
Group points by the lines they lie on.

For each pair of points, calculate the line passing through them
and add both points to that line's point set.
"""
function find_collinear_points(points::Vector{Tuple{Int, Int}})
    lines_to_points = Dict{Tuple{Int, Int, Int}, Set{Tuple{Int, Int}}}()

    for p1 in points
        for p2 in points
            if p1 != p2
                line = get_line_through_points(p1, p2)
                if !haskey(lines_to_points, line)
                    lines_to_points[line] = Set{Tuple{Int, Int}}()
                end
                push!(lines_to_points[line], p1)
                push!(lines_to_points[line], p2)
            end
        end
    end

    return lines_to_points
end


# Get the symmetric counterpart of a point for a given symmetry transformation.

function get_symmetric_point(point::Tuple{Int, Int}, n::Int, symmetry_type::SymmetryType)
    i, j = point
    if symmetry_type == HORIZONTAL
        return (i, n + 1 - j)
    elseif symmetry_type == VERTICAL
        return (n + 1 - i, j)
    elseif symmetry_type == DIAGONAL
        return (j, i)
    elseif symmetry_type == ANTI_DIAGONAL
        return (n + 1 - j, n + 1 - i)
    elseif symmetry_type == ROTATIONAL_90
        return (j, n + 1 - i)
    elseif symmetry_type == ROTATIONAL_180
        return (n + 1 - i, n + 1 - j)
    elseif symmetry_type == ROTATIONAL_270
        return (n + 1 - j, i)
    else
        throw(ArgumentError("Invalid symmetry type: $symmetry_type"))
    end
end

"""
Get the list of symmetry transformations required for a given symmetry type.

For compound symmetries like `BOTH_AXES`, returns multiple individual transformations
that need to be applied together to achieve the desired symmetry.
"""
function get_symmetry_transforms(symmetry_type::SymmetryType)
    if symmetry_type == NONE
        return SymmetryType[]
    elseif symmetry_type == HORIZONTAL
        return [HORIZONTAL]
    elseif symmetry_type == VERTICAL
        return [VERTICAL]
    elseif symmetry_type == DIAGONAL
        return [DIAGONAL]
    elseif symmetry_type == ANTI_DIAGONAL
        return [ANTI_DIAGONAL]
    elseif symmetry_type == BOTH_DIAGONALS
        return [DIAGONAL, ANTI_DIAGONAL]
    elseif symmetry_type == ROTATIONAL_90
        return [ROTATIONAL_90, ROTATIONAL_270]
    elseif symmetry_type == ROTATIONAL_180
        return [ROTATIONAL_180]
    elseif symmetry_type == BOTH_AXES
        return [HORIZONTAL, VERTICAL]
    elseif symmetry_type == ALL
        return [HORIZONTAL, VERTICAL, DIAGONAL, ANTI_DIAGONAL, ROTATIONAL_90, ROTATIONAL_180]
    else
        return SymmetryType[]
    end
end

"""
Generate symmetry constraints for the optimisation model.

For each symmetry transformation, finds pairs of points that must have the same
selection status. Removes duplicates and ensures each pair is represented only once.
"""
function get_symmetry_constraints(points::Vector{Tuple{Int, Int}}, n::Int, symmetry_type::SymmetryType)
    transforms = get_symmetry_transforms(symmetry_type)
    constraints = Set{Tuple{Tuple{Int, Int}, Tuple{Int, Int}}}()

    for transform in transforms
        processed = Set{Tuple{Int, Int}}()

        for p in points
            p in processed && continue

            sp = get_symmetric_point(p, n, transform)

            # Only add constraint if symmetric point exists and is different
            if sp in points && p != sp
                # Normalize the pair so (p1,p2) and (p2,p1) are considered the same
                pair = p < sp ? (p, sp) : (sp, p)
                push!(constraints, pair)
                push!(processed, p)
                push!(processed, sp)
            end
        end
    end

    return collect(constraints)
end

"""
Build the ILP model for the no-three-in-line problem using JuMP and Gurobi.
"""
function build_optimisation_model(
    points::Vector{Tuple{Int, Int}},
    collinear_groups::Dict{Tuple{Int, Int, Int}, Set{Tuple{Int, Int}}},
    n:: Int,
    symmetry_type::SymmetryType = NONE,
    threads::Union{Int, Nothing} = nothing
)
    min_points = 2 * n
    max_points = 2 * n
    # Create JuMP model with Gurobi optimizer
    model = Model(Gurobi.Optimizer)

    # Set parameters (equivalent to Python's model.setParam)
    set_optimizer_attribute(model, "OutputFlag", 1)  # Show output for debugging

    # Set thread count if specified
    if threads !== nothing
        set_optimizer_attribute(model, "Threads", threads)
        println("Using $threads threads")
    end

    # For problems with symmetry, set helpful parameters
    if symmetry_type != NONE
        set_optimizer_attribute(model, "Symmetry", 2)  # Aggressive symmetry detection
        set_optimizer_attribute(model, "MIPFocus", 1)  # Focus on finding feasible solutions quickly
    end

    # Create binary variables for each point as a Dict
    point_vars = Dict{Tuple{Int, Int}, VariableRef}()
    for point in points
        point_vars[point] = @variable(model, binary=true, base_name="select_$(point[1])_$(point[2])")
    end

    # Objective: maximise number of selected points
    @objective(model, Max, sum(values(point_vars)))

    # Constraint: at most 2 points on any line
    for (line, line_points) in collinear_groups
        if length(line_points) > 2  # Only constrain lines with 3+ points
            @constraint(model,
                sum(point_vars[p] for p in line_points) <= 2,
                base_name = "no_three_on_line"
            )
        end
    end

    # Optional constraint: require at least min_points
    if min_points > 0
        @constraint(model,
            sum(values(point_vars)) >= min_points,
            base_name = "minimum_points"
        )
    end

    # Optional constraint: cap the number of points
    if max_points > 0
        @constraint(model,
            sum(values(point_vars)) <= max_points,
            base_name = "maximum_points"
        )
    end

    # Add symmetry constraints
    if symmetry_type != NONE
        symmetry_constraints = get_symmetry_constraints(points, n, symmetry_type)

        constraint_count = 0
        for (point1, point2) in symmetry_constraints
            @constraint(model,
                point_vars[point1] == point_vars[point2],
                base_name = "symmetry_$constraint_count"
            )
            constraint_count += 1
        end

        println("Added $constraint_count symmetry constraints for $symmetry_type symmetry")
    end

    return model, point_vars
end


function display_solution(n::Int, point_vars::Dict{Tuple{Int, Int}, VariableRef},
                         objective_value::Float64, show_symmetry_analysis::Bool = true)
    println("Maximum number of points selected: $(Int(objective_value))")
    println("This achieves the theoretical maximum of 2n = $(2*n) points")

    # Print grid with 'O' for selected points, '.' for unselected
    for i in 1:n
        row = String[]
        for j in 1:n
            if value(point_vars[(i, j)]) > 0.5  # Point is selected
                push!(row, "O")
            else
                push!(row, ".")
            end
        end
        println(join(row, " "))
    end

    if show_symmetry_analysis
        println("\nSymmetry Analysis:")
        symmetries = check_solution_symmetry(n, point_vars)
        for (sym_type, has_symmetry) in symmetries
            status = has_symmetry ? "✓" : "✗"
            sym_name = replace(string(sym_type), "_" => " ")
            println("  $status $(titlecase(sym_name))")
        end
    end
end

function save_solution_to_file(n::Int, point_vars::Dict{Tuple{Int, Int}, VariableRef},
                              objective_value::Float64, filename::Union{String, Nothing} = nothing,
                              symmetry_type::SymmetryType = NONE)
    if filename === nothing
        sym_suffix = symmetry_type != NONE ? "_$(string(symmetry_type))" : ""
        filename = "no_three_in_line_n$(n)$(sym_suffix)_solution.txt"
    end

    open(filename, "w") do f
        write(f, "No-three-in-line solution for $(n)x$(n) grid\n")
        write(f, "Symmetry constraint: $(string(symmetry_type))\n")
        write(f, "Maximum number of points selected: $(Int(objective_value))\n")
        write(f, "Theoretical maximum of 2n = $(2*n) points\n\n")

        # Write grid
        for i in 1:n
            row = String[]
            for j in 1:n
                if value(point_vars[(i, j)]) > 0.5
                    push!(row, "O")
                else
                    push!(row, ".")
                end
            end
            write(f, join(row, " ") * "\n")
        end

        # Write selected points as coordinates
        write(f, "\nSelected points (coordinates):\n")
        selected_points = Tuple{Int, Int}[]
        for i in 1:n
            for j in 1:n
                if value(point_vars[(i, j)]) > 0.5
                    push!(selected_points, (i, j))
                end
            end
        end

        for point in selected_points
            write(f, "$(point)\n")
        end

        # Write symmetry analysis
        write(f, "\nSymmetry Analysis:\n")
        symmetries = check_solution_symmetry(n, point_vars)
        for (sym_type, has_symmetry) in symmetries
            status = has_symmetry ? "Yes" : "No"
            sym_name = replace(string(sym_type), "_" => " ")
            write(f, "$(titlecase(sym_name)): $status\n")
        end
    end

    println("Solution saved to $filename")
end

function check_solution_symmetry(n::Int, point_vars::Dict{Tuple{Int, Int}, VariableRef})
    symmetries = Dict{SymmetryType, Bool}()

    # Get selected points
    selected_points = Set{Tuple{Int, Int}}()
    for i in 1:n
        for j in 1:n
            if value(point_vars[(i, j)]) > 0.5
                push!(selected_points, (i, j))
            end
        end
    end

    # Check each symmetry type
    for sym_type in instances(SymmetryType)
        if sym_type == NONE
            symmetries[sym_type] = true  # NONE is always satisfied
            continue
        end

        has_symmetry = true
        transforms = get_symmetry_transforms(sym_type)

        for transform in transforms
            for p in selected_points
                sp = get_symmetric_point(p, n, transform)
                if sp ∉ selected_points
                    has_symmetry = false
                    break
                end
            end
            if !has_symmetry
                break
            end
        end

        symmetries[sym_type] = has_symmetry
    end

    return symmetries
end

function solution_to_matrix(point_vars::Dict{Tuple{Int,Int}, VariableRef}, n::Int)
    grid = zeros(Float32, n, n)
    for i in 1:n
        for j in 1:n
            grid[i, j] = value(point_vars[(i, j)]) > 0.5 ? 1.0f0 : 0.0f0
        end
    end
    return grid
end

function augment_solution(grid::Matrix{Float32})
    augmented = Matrix{Float32}[]
    
    # Original
    push!(augmented, copy(grid))
    
    # Rotations (using rot180, rotr90, rotl90)
    push!(augmented, rotr90(grid))   # 90° clockwise
    push!(augmented, rot180(grid))    # 180°
    push!(augmented, rotl90(grid))   # 270° clockwise
    
    # Flips
    push!(augmented, reverse(grid, dims=2))  # horizontal flip
    push!(augmented, reverse(grid, dims=1))  # vertical flip
    push!(augmented, reverse(reverse(grid, dims=1), dims=2))  # both
    
    # Transpose (diagonal flip)
    push!(augmented, permutedims(grid, (2, 1)))
    
    return augmented
end

function create_trajectory(x_optimal::Matrix{Float32}; T::Int=20, rng::AbstractRNG=Random.GLOBAL_RNG)
    n = size(x_optimal, 1)

    # Random initialization with values in [0, 0.3]
    x_init = rand(rng, Float32, n, n) .* 0.3f0

    # Pre-allocate trajectory array
    trajectory = zeros(Float32, T, n, n)

    # Linear interpolation from x_init to x_optimal
    for t in 1:T
        α = Float32((t - 1) / (T - 1))
        trajectory[t, :, :] = (1f0 - α) .* x_init .+ α .* x_optimal
    end

    return x_init, trajectory, x_optimal
end

# Convenience overload to support positional T with optional keyword rng
function create_trajectory(x_optimal::Matrix{Float64}, T::Int; rng::AbstractRNG=Random.GLOBAL_RNG)
    return create_trajectory(Float32.(x_optimal); T=T, rng=rng)
end

function create_trajectory(x_optimal::Matrix{Float32}, T::Int; rng::AbstractRNG=Random.GLOBAL_RNG)
    return create_trajectory(x_optimal; T=T, rng=rng)
end


function generate_collinear_triplet_list(collinear_groups::Dict{Tuple{Int, Int, Int}, Set{Tuple{Int, Int}}})
    L = Tuple{Int,Int,Int,Int,Int,Int}[]
    
    for (line, line_points) in collinear_groups
        if length(line_points) >= 3
            points_vec = collect(line_points)
            # Generate all triplets from this line
            for i in 1:length(points_vec)-2
                for j in i+1:length(points_vec)-1
                    for k in j+1:length(points_vec)
                        p1, p2, p3 = points_vec[i], points_vec[j], points_vec[k]
                        push!(L, (p1[1], p1[2], p2[1], p2[2], p3[1], p3[2]))
                    end
                end
            end
        end
    end
    
    return L
end


function generate_training_dataset(n::Int=5; 
                                  num_base_solutions::Int=5,
                                  T::Int=20,
                                  save_path::String="training_data_n$(n).jld2",
                                  threads::Union{Int,Nothing}=nothing,
                                  seed::Int=42)
    
    # Set up random seed for reproducibility
    rng = MersenneTwister(seed)
    
    dataset = []
    symmetry_types = [NONE, HORIZONTAL, VERTICAL, DIAGONAL, BOTH_AXES]
    
    println("=" ^ 60)
    println("Generating Training Data for n=$n")
    println("Following Roadmap Action Items 1-2")
    println("=" ^ 60)
    
    # Generate collinear groups once (same for all symmetries)
    points = generate_grid_points(n)
    collinear_groups = find_collinear_points(points)
    L = generate_collinear_triplet_list(collinear_groups)
    
    println("\nGrid info:")
    println("  Points: $(length(points))")
    println("  Lines with 3+ points: $(count(length(pts) > 2 for (_, pts) in collinear_groups))")
    println("  Collinear triplets: $(length(L))")
    
    for idx in 1:num_base_solutions
        # Cycle through symmetry types, then use NONE with random objectives
        symmetry = idx <= length(symmetry_types) ? symmetry_types[idx] : NONE
        
        println("\n[$idx/$num_base_solutions] Solving with symmetry: $symmetry")
        
        # Build and solve model
        model, point_vars = build_optimisation_model(
            points, collinear_groups, n, symmetry, threads
        )
        
        # For solutions beyond the symmetry types, add random objective perturbation
        if idx > length(symmetry_types)
            Random.seed!(rng, idx * 100)
            obj_weights = 1.0 .+ 0.15 .* randn(rng, n, n)
            @objective(model, Max, sum(obj_weights[i,j] * point_vars[(i,j)] 
                                      for i in 1:n for j in 1:n))
            println("  Using randomized objective (seed=$(idx*100))")
        end
        
        # Suppress output for cleaner logs
        set_optimizer_attribute(model, "OutputFlag", 0)
        
        optimize!(model)
        
        if termination_status(model) != MOI.OPTIMAL
            @warn "Failed to find optimal solution for configuration $idx, skipping..."
            continue
        end
        
        # Count actual points (not objective value, which may be weighted)
        num_points = sum(value(point_vars[(i,j)]) > 0.5 for i in 1:n for j in 1:n)
        println("  ✓ Found optimal solution with $(Int(num_points)) points")
        
        # Convert to matrix
        x_optimal = solution_to_matrix(point_vars, n)
        
        # Augment with rotations/flips
        augmented_grids = augment_solution(x_optimal)
        println("  ✓ Generated $(length(augmented_grids)) augmented versions")
        
        # Create trajectory for each augmented version
        for (aug_idx, aug_grid) in enumerate(augmented_grids)
            x_init, trajectory, x_final = create_trajectory(aug_grid, T; rng=rng)
            
            push!(dataset, Dict(
                "initial" => x_init,
                "trajectory" => trajectory,
                "optimal" => x_final,
                "source_symmetry" => string(symmetry),
                "augmentation_idx" => aug_idx,
                "num_points" => sum(x_final),
                "n" => n,
                "T" => T
            ))
        end
    end
    
    println("\n" * "=" ^ 60)
    println("Dataset Generation Complete")
    println("=" ^ 60)
    println("Total samples: $(length(dataset))")
    println("Expected: $(num_base_solutions) symmetries × 8 augmentations = $(num_base_solutions * 8)")
    
    # Save to file with metadata
    jldsave(save_path; 
            dataset=dataset, 
            n=n, 
            T=T, 
            seed=seed,
            collinear_triplets=L,
            num_samples=length(dataset))
    
    println("✓ Saved to $save_path")
    println("\nNext step: Train UDE using this data (Action Item 3)")
    
    return dataset, L
end

function validate_dataset(dataset_path::String)
    println("\n" * "=" ^ 60)
    println("Validating Dataset")
    println("=" ^ 60)
    
    # Load data
    data = load(dataset_path)
    dataset = data["dataset"]
    n = data["n"]
    T = data["T"]
    L = data["collinear_triplets"]
    
    println("\nDataset Info:")
    println("  Grid size: $(n)×$(n)")
    println("  Time steps: $T")
    println("  Number of samples: $(length(dataset))")
    println("  Collinear triplets: $(length(L))")
    
    # Validation checks
    all_valid = true
    
    println("\nValidating samples...")
    for (i, sample) in enumerate(dataset)
        # Check shapes
        @assert size(sample["initial"]) == (n, n) "Sample $i: Wrong initial shape"
        @assert size(sample["trajectory"]) == (T, n, n) "Sample $i: Wrong trajectory shape"
        @assert size(sample["optimal"]) == (n, n) "Sample $i: Wrong optimal shape"
        
        # Check trajectory endpoints
        if !isapprox(sample["trajectory"][1, :, :], sample["initial"], atol=1e-10)
            @warn "Sample $i: Trajectory doesn't start at initial"
            all_valid = false
        end
        
        if !isapprox(sample["trajectory"][end, :, :], sample["optimal"], atol=1e-10)
            @warn "Sample $i: Trajectory doesn't end at optimal"
            all_valid = false
        end
        
        # Check for NaN/Inf
        if any(isnan, sample["trajectory"]) || any(isinf, sample["trajectory"])
            @warn "Sample $i: Contains NaN or Inf values"
            all_valid = false
        end
        
        # Check optimal solution satisfies constraints
        optimal = sample["optimal"]
        violations = 0
        for (i1, j1, i2, j2, i3, j3) in L
            if optimal[i1, j1] + optimal[i2, j2] + optimal[i3, j3] > 2.5  # Allow small numerical error
                violations += 1
            end
        end
        
        if violations > 0
            @warn "Sample $i: Optimal solution has $violations constraint violations!"
            all_valid = false
        end
    end
    
    if all_valid
        println("✓ All validation checks passed!")
    else
        println("✗ Some validation checks failed - see warnings above")
    end
    
    # Display sample statistics
    println("\nSample Statistics (first 3 samples):")
    for (i, sample) in enumerate(dataset[1:min(3, length(dataset))])
        println("\nSample $i:")
        println("  Source symmetry: $(sample["source_symmetry"])")
        println("  Augmentation: $(sample["augmentation_idx"])")
        println("  Num points: $(Int(sample["num_points"]))")
        println("  Initial range: [$(round(minimum(sample["initial"]), digits=3)), $(round(maximum(sample["initial"]), digits=3))]")
        println("  Optimal range: [$(round(minimum(sample["optimal"]), digits=3)), $(round(maximum(sample["optimal"]), digits=3))]")
    end
    
    println("\n" * "=" ^ 60)
    println("Validation Complete")
    println("=" ^ 60)
    
    return dataset, L
end

function visualize_sample(sample::Dict; save_path::Union{String,Nothing}=nothing)
    n = sample["n"]
    T = sample["T"]
    
    # Select time points to display
    time_points = [1, T÷4, T÷2, 3T÷4, T]
    
    plots = []
    for t in time_points
        p = heatmap(sample["trajectory"][t, :, :], 
                    clim=(0, 1), 
                    color=:viridis,
                    title="t=$t",
                    aspect_ratio=:equal,
                    showaxis=false)
        push!(plots, p)
    end
    
    plot(plots..., layout=(1, 5), size=(1500, 300))
    
    if save_path !== nothing
        savefig(save_path)
    end
    
    println("Visualization requires Plots.jl - function body commented out")
end

function main_generate_data(; n=5, num_base_solutions=5, T=20, threads=12, seed=42)
    println("\n" * "╔" * "═"^58 * "╗")
    println("║" * " "^10 * "No-Three-In-Line Data Generation" * " "^16 * "║")
    println("║" * " "^15 * "Following Scientific ML Roadmap" * " "^13 * "║")
    println("╚" * "═"^58 * "╝")
    
    save_path = "training_data_n$(n).jld2"
    
    println("\nConfiguration:")
    println("  Grid size: $(n)×$(n)")
    println("  Base solutions: $num_base_solutions")
    println("  Time steps: $T")
    println("  Threads: $threads")
    println("  Random seed: $seed")
    println("  Output file: $save_path")
    # Generate dataset
    println("\nStep 1: Generating training dataset...")
    dataset, L = generate_training_dataset(
        n, 
        num_base_solutions=num_base_solutions,
        T=T,
        save_path=save_path,
        threads=threads,
        seed=seed
    )
    
    # Validate dataset
    println("\nStep 2: Validating dataset...")
    validate_dataset(save_path)
    
    println("\n" * "╔" * "═"^58 * "╗")
    println("║" * " "^10 * "✓ Data Generation Complete!" * " "^20 * "║")
    println("╚" * "═"^58 * "╝")
    
    println("\nGenerated files:")
    println("  • $save_path  ($(round(filesize(save_path)/1024^2, digits=2)) MB)")
    
    println("\nNext steps:")
    println("  1. Review the validation output above")
    println("  2. Proceed to Action Item 3: Train UDE model")
    println("  3. Implement Equation 6: dx/dt = NN(x; θ) + η(t)")
    
    return dataset, L
end

# Uncomment to run:
# dataset, L = main_generate_data(n=5, num_base_solutions=20, T=80, threads=12)

# N = 25
# symmetry_type = BOTH_DIAGONALS
# points = generate_grid_points(N)
# @time collinear_groups = find_collinear_points(points)
# @time model, point_vars = build_optimisation_model(points, collinear_groups, N, symmetry_type)
# @time optimise!(model)

# # Check if optimisation was successful
# if termination_status(model) == MOI.OPTIMAL
#     println("\nOptimisation successful!")
#     obj_val = objective_value(model)
#     @time display_solution(N, point_vars, obj_val)
#     @time save_solution_to_file(N, point_vars, obj_val)
# elseif termination_status(model) == MOI.INFEASIBLE
#     println("\nModel is INFEASIBLE - no solution exists with the given constraints")
#     println("Termination status: $(termination_status(model))")
# else
#     println("\nOptimisation failed or didn't find optimal solution")
#     println("Termination status: $(termination_status(model))")
#     if has_values(model)
#         println("But found a feasible solution with objective value: $(objective_value(model))")
#         obj_val = objective_value(model)
#         @time display_solution(N, point_vars, obj_val)
#         @time save_solution_to_file(N, point_vars, obj_val)
#     end
# end