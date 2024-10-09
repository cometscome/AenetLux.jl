using Lux, Random, Zygote, CUDA
using ComponentArrays

# Define the NetAtom structure using Lux
struct NetAtom
    input_size::Vector{Int}
    hidden_size::Vector{Vector{Int}}
    species::Vector{String}
    activations::Vector{Vector{Function}}
    alpha::Vector{Float64}
    device
    functions::Vector{Lux.Model}
end

# Activation functions mapping
activation_map = Dict("linear" => x -> x, "tanh" => Lux.Tanh(), "sigmoid" => Lux.Sigmoid())

function NetAtom(input_size::Vector{Int}, hidden_size::Vector{Vector{Int}}, species::Vector{String},
    activations::Vector{Vector{String}}, alpha::Vector{Float64}, device)

    # Create the activation function layers
    activations_list = [map(act_name -> activation_map[act_name], activations[i]) for i in 1:length(species)]

    # Create function chains (Dense + Activation) for each species
    function_chains = []
    for i in 1:length(species)
        layers = []
        push!(layers, Lux.Chain(Dense(input_size[i], hidden_size[i][1]), activations_list[i][1]))
        for j in 2:length(hidden_size[i])
            push!(layers, Lux.Chain(Dense(hidden_size[i][j-1], hidden_size[i][j]), activations_list[i][j]))
        end
        push!(layers, Dense(hidden_size[i][end], 1))
        push!(function_chains, Lux.Chain(layers...))
    end

    return NetAtom(input_size, hidden_size, species, activations_list, alpha, device, function_chains)
end

# Forward pass for energy training
function forward(net::NetAtom, grp_descrp, logic_reduce, ps)
    partial_E_ann = [lux_apply(net.functions[i], ps[i], grp_descrp[i]) for i in 1:length(net.species)]

    list_E_ann = CUDA.zeros(length(logic_reduce[1]))
    for i in 1:length(net.species)
        list_E_ann += sum(partial_E_ann[i] .* logic_reduce[i], dims=2)[:]
    end
    return list_E_ann
end

# Forward pass for force training
function forward_F(net::NetAtom, group_descrp, group_sfderiv_i, group_sfderiv_j, group_indices_F, grp_indices_F_i,
    logic_reduce, input_size, max_nnb, ps)

    E_atomic_ann = []
    aux_F_i = CUDA.zeros(0, 3)
    aux_F_j = CUDA.zeros(0, max_nnb, 3)

    for i in 1:length(net.species)
        group_descrp[i] = group_descrp[i] |> gpu
        partial_E_ann = lux_apply(net.functions[i], ps[i], group_descrp[i])
        dE_dG = gradient(() -> sum(partial_E_ann), group_descrp[i])[1]

        push!(E_atomic_ann, partial_E_ann)
        aux_F_j = vcat(aux_F_j, sum(dE_dG[:, :, None] .* group_sfderiv_j[i], dims=1))
        aux_F_i = vcat(aux_F_i, sum(dE_dG[:, :, None] .* group_sfderiv_i[i], dims=1))
    end

    aux_F_j = vcat(aux_F_j, zeros(1, size(aux_F_j, 2), 3))
    aux_F_flat = reshape(aux_F_j, :, 3)

    F_ann = -aux_F_i[grp_indices_F_i, :] .- sum(aux_F_flat[group_indices_F, :], dims=1)

    list_E_ann = CUDA.zeros(length(logic_reduce[1]))
    for i in 1:length(net.species)
        list_E_ann += sum(E_atomic_ann[i] .* logic_reduce[i], dims=2)[:]
    end

    return list_E_ann, F_ann
end

# Loss function for RMSE (energy training)
function get_loss_RMSE(net::NetAtom, grp_descrp, grp_energy, logic_reduce, grp_N_atom, ps)
    list_E_ann = forward(net, grp_descrp, logic_reduce, ps)
    N_data = length(list_E_ann)
    differences = (list_E_ann .- grp_energy)
    l2 = sum(differences .^ 2 ./ grp_N_atom .^ 2)
    return l2, N_data
end

# Loss function for RMSE (force training)
function get_loss_RMSE_F(net::NetAtom, group_energy, group_N_atom, group_descrp, group_sfderiv_i, group_sfderiv_j,
    group_indices_F, grp_indices_F_i, logic_reduce, group_forces, input_size, max_nnb, E_scaling, ps)
    list_E_ann, list_F_ann = forward_F(net, group_descrp, group_sfderiv_i, group_sfderiv_j, group_indices_F, grp_indices_F_i, logic_reduce, input_size, max_nnb, ps)

    N_data_E = length(list_E_ann)
    N_data_F = length(list_F_ann)

    diff_E = list_E_ann .- group_energy
    diff_F = list_F_ann .- group_forces

    l2_E = sum(diff_E .^ 2 ./ group_N_atom .^ 2)
    l_F = sqrt(sum(diff_F .^ 2) / N_data_F)

    return l2_E, l_F, N_data_E, N_data_F
end

# Helper function to apply the model using Lux
function lux_apply(f, ps, x)
    Lux.apply(f, ps, Lux.init(f)[2], x)
end