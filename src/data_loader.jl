using Random
#using CUDA
using BSON: @save, @load

function custom_collate(batch)
    return batch
end

mutable struct PrepDataloader{T}
    dataset::T
    train_forces::Bool
    batch_size::Int
    N_batch::Int
    sampler::Vector{Int}
    memory_mode::String
    device::String
    dataname::String
    #indexes::Vector{Tuple{Int,Int}}
    indexes::Matrix{Int64}
    group_energy
    group_descrp
    logic_tensor_reduce
    index_from_database
    group_N_atom
    group_sfderiv_i
    group_sfderiv_j
    group_forces
    group_indices_F
    group_indices_F_i
    batch_names

    function PrepDataloader(dataset; train_forces=false, batch_size=1, N_batch=1,
        sampler=nothing, memory_mode="cpu", device="cpu",
        dataname::String="", generate::Bool=true)
        if sampler === nothing
            sampler = collect(1:length(dataset))
            Random.shuffle!(sampler)
        end

        if memory_mode == "cpu"
            device = "cpu"
        end

        # Initialize
        indexes = Matrix{Int64}(undef, 0, 0) #Vector{Tuple{Int,Int}}()
        group_energy = nothing
        group_descrp = nothing
        logic_tensor_reduce = nothing
        index_from_database = nothing
        group_N_atom = nothing
        group_sfderiv_i = nothing
        group_sfderiv_j = nothing
        group_forces = nothing
        group_indices_F = nothing
        group_indices_F_i = nothing
        batch_names = [joinpath("tmp_batches", dataname * "batch_energy$(ibatch).bson") for ibatch in 1:N_batch]

        if generate
            # Generate batches
            if N_batch == 0
                indexes = Matrix{Int64}(undef, 0, 0)
            else
                indexes = get_batch_indexes_N_batch(dataset, sampler, N_batch)
            end
        end

        new{typeof(dataset)}(dataset, train_forces, batch_size, N_batch, sampler, memory_mode, device, dataname,
            indexes, group_energy, group_descrp, logic_tensor_reduce, index_from_database,
            group_N_atom, group_sfderiv_i, group_sfderiv_j, group_forces, group_indices_F,
            group_indices_F_i, batch_names)
    end
end

function get_batch_indexes_N_batch(dataset, sampler, N_batch)
    base, extra = divrem(length(sampler), N_batch)
    N_per_batch = [base + (i < extra ? 1 : 0) for i in 1:N_batch]

    Random.shuffle!(N_per_batch)

    finish = 0
    indexes = Vector{Int64}[]#Vector{Tuple{Int,Int}}()
    for i in 1:N_batch
        start = finish
        finish = start + N_per_batch[i]
        #push!(indexes, (start, finish))
        push!(indexes, [start, finish])
    end

    return vcat(transpose(indexes)...)
end

function save_batch(batch_name, save_data)
    @save batch_name save_data
end

function load_batch(batch_name)
    data = BSON.load(batch_name)
    return data
end

function delete_batch!(batch)
    batch = nothing
end

function prepare_batches(dataset, indexes, sampler, N_batch, device)
    group_descrp = [[zeros(0, dataset.input_size[iesp]) for iesp in 1:dataset.N_species] for ibatch in 1:N_batch]
    group_energy = [[] for ibatch in 1:N_batch]
    index_from_database = [[] for ibatch in 1:N_batch]
    group_N_atom = [[] for ibatch in 1:N_batch]

    for ibatch in 1:N_batch
        index = indexes[ibatch]
        ind_start_struc = 0

        for istruc in index[1]:index[2]
            index_struc = sampler[istruc]
            append!(group_energy[ibatch], dataset[index_struc].energy)
            append!(group_N_atom[ibatch], dataset[index_struc].N_atom)
            append!(index_from_database[ibatch], (dataset[index_struc].name, dataset[index_struc].E_atomic_structure))

            for iesp in 1:dataset.N_species
                group_descrp[ibatch][iesp] = vcat(group_descrp[ibatch][iesp], dataset[index_struc].descriptor[iesp])
            end
        end

        group_energy[ibatch] = CUDA.convert(Vector{Float64}, group_energy[ibatch])
        group_N_atom[ibatch] = CUDA.convert(Vector{Float64}, group_N_atom[ibatch])
    end

    return group_energy, group_descrp, index_from_database, group_N_atom
end