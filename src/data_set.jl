#using CUDA

struct StructureDataset{T1,T2}
    list_struc::Vector{T1}
    sys_species::Vector{T2}
    N_species::Int
    input_size::Vector{Int64}
    max_nnb::Int
    device::String

end

function StructureDataset(list_structures, sys_species, input_size, max_nnb, device="cpu")
    T1 = eltype(list_structures)
    T2 = eltype(sys_species)
    println(typeof(list_structures))

    #println(list_structures)
    println(typeof(sys_species))
    #println(sys_species)
    println(typeof(input_size))
    println(input_size)
    println(typeof(max_nnb))
    StructureDataset{T1,T2}(
        list_structures,
        sys_species,
        length(sys_species),
        input_size,
        max_nnb,
        device
        #CUDA.has_cuda() ? "cuda" : "cpu"
    )
end

function Base.length(dataset::StructureDataset)
    return length(dataset.list_struc)
end

function Base.getindex(dataset::StructureDataset, index::Int)
    return dataset.list_struc[index]
end

function get_names_and_E_atomic_structure(dataset::StructureDataset)
    dataset_names = String[]
    dataset_E_atomic_structure = Float64[]
    for struc in dataset.list_struc
        push!(dataset_names, struc.name)
        push!(dataset_E_atomic_structure, struc.E_atomic_structure)
    end
    return dataset_names, dataset_E_atomic_structure
end

function get_species(dataset::StructureDataset)
    return dataset.sys_species
end

function normalize_E!(dataset::StructureDataset, E_scaling, E_shift)
    for struc in dataset.list_struc
        struc.energy = (struc.energy - struc.N_atom * E_shift) * E_scaling
    end
end

function normalize_F!(dataset::StructureDataset, E_scaling, E_shift)
    for struc in dataset.list_struc
        struc.forces .*= E_scaling
    end
end

function normalize_stp!(dataset::StructureDataset, sfval_avg, sfval_cov)
    shift = []
    scale = []
    for iesp in 1:dataset.N_species
        sh = deepcopy(sfval_avg[iesp])
        #println(sh)
        #println(sfval_cov[iesp])
        sc = 1.0 ./ sqrt.(sfval_cov[iesp] - sh .^ 2)

        # Check if scale is finite. If infinite, it means some values are always 0

        sc = map(x -> ifelse(isinf(x), 0.0, x), sc)
        #if isinf(sc)
        #    sc = 0.0
        #end

        push!(shift, sh)
        push!(scale, sc)
    end

    for struc in dataset.list_struc
        for iesp in 1:dataset.N_species
            #display(struc.descriptor[iesp])
            #display(shift[iesp])
            #display(scale[iesp])
            #error("dd")
            n1, n2 = size(struc.descriptor[iesp])
            for i2 = 1:n2
                for i1 = 1:n1
                    struc.descriptor[iesp][i1, i2] = (struc.descriptor[iesp][i1, i2] - shift[iesp][i2]) * scale[iesp][i2]
                end
            end
            #for i = 1:length(struc.descriptor[iesp])
            #    struc.descriptor[iesp][i] = (struc.descriptor[iesp][i] .- shift[iesp]) .* scale[iesp]
            #end
            #struc.descriptor[iesp] = (struc.descriptor[iesp] .- shift[iesp]) .* scale[iesp]

            if struc.train_forces
                #display(struc.list_sfderiv_i[iesp]) #8x56x3
                #display(scale[iesp]) #56
                #println(size(struc.list_sfderiv_i[iesp]))
                #println(size(struc.list_sfderiv_j[iesp]))
                #error("d")
                n1, n2, n3 = size(struc.list_sfderiv_i[iesp])
                for i3 = 1:n3
                    for i2 = 1:n2
                        for i1 = 1:n1
                            struc.list_sfderiv_i[iesp][i1, i2, i3] *= scale[iesp][i2]
                        end
                    end
                end
                n1, n2, n3, n4 = size(struc.list_sfderiv_j[iesp])
                for i4 = 1:n4
                    for i3 = 1:n3
                        for i2 = 1:n2
                            for i1 = 1:n1
                                struc.list_sfderiv_j[iesp][i1, i2, i3, i4] *= scale[iesp][i3]
                            end
                        end
                    end
                end
                #struc.list_sfderiv_i[iesp] .= struc.list_sfderiv_i[iesp] .* scale[iesp]
                #struc.list_sfderiv_j[iesp] .= struc.list_sfderiv_j[iesp] .* scale[iesp]
            end
        end
    end

    return shift, scale
end

struct GroupedDataset
    memory_mode::String
    device::String
    dataname::String
    train_energy::Bool
    train_forces::Bool
    N_batch::Int

    F_group_descrp::Vector
    F_group_energy::Vector
    F_logic_tensor_reduce::Vector
    F_index_from_database::Vector
    F_group_N_atom::Vector
    F_group_forces::Vector
    F_group_sfderiv_i::Vector
    F_group_sfderiv_j::Vector
    F_group_indices_F::Vector
    F_group_indices_F_i::Vector

    E_group_descrp::Vector
    E_group_energy::Vector
    E_logic_tensor_reduce::Vector
    E_index_from_database::Vector
    E_group_N_atom::Vector

    function GroupedDataset(energy_data, forces_data; generate=true, memory_mode="cpu", device="cpu", dataname="")
        train_energy = !isnothing(energy_data)
        train_forces = !isnothing(forces_data)
        N_batch = ifelse(train_energy, energy_data.N_batch, forces_data.N_batch)

        new(
            memory_mode,
            device,
            dataname,
            train_energy,
            train_forces,
            N_batch,
            Vector{Any}(undef, N_batch),
            Vector{Any}(undef, N_batch),
            Vector{Any}(undef, N_batch),
            Vector{Any}(undef, N_batch),
            Vector{Any}(undef, N_batch),
            Vector{Any}(undef, N_batch),
            Vector{Any}(undef, N_batch),
            Vector{Any}(undef, N_batch),
            Vector{Any}(undef, N_batch),
            Vector{Any}(undef, N_batch),
            Vector{Any}(undef, N_batch),
            Vector{Any}(undef, N_batch),
            Vector{Any}(undef, N_batch),
            Vector{Any}(undef, N_batch)
        )
    end
end

function Base.length(dataset::GroupedDataset)
    return dataset.N_batch
end

function Base.getindex(dataset::GroupedDataset, index::Int)
    data = [
        dataset.F_group_descrp[index],
        dataset.F_group_energy[index],
        dataset.F_logic_tensor_reduce[index],
        dataset.F_index_from_database[index],
        dataset.F_group_N_atom[index],
        dataset.F_group_forces[index],
        dataset.F_group_sfderiv_i[index],
        dataset.F_group_sfderiv_j[index],
        dataset.F_group_indices_F[index],
        dataset.F_group_indices_F_i[index],
        dataset.E_group_descrp[index],
        dataset.E_group_energy[index],
        dataset.E_logic_tensor_reduce[index],
        dataset.E_index_from_database[index],
        dataset.E_group_N_atom[index]
    ]

    if dataset.device == "cuda"
        data = batch_data_cpu_to_gpu(dataset, data)
    end

    return data
end

function batch_data_cpu_to_gpu(dataset::GroupedDataset, data)
    data_gpu = [[] for _ in 1:length(data)]
    if dataset.train_forces
        data_gpu[3] = data[3]
        data_gpu[1] = CUDA.convert(CuArray, data[1])
        data_gpu[4] = CUDA.convert(CuArray, data[4])
        data_gpu[5] = CUDA.convert(CuArray, data[5])
        data_gpu[8] = CUDA.convert(CuArray, data[8])
        data_gpu[9] = CUDA.convert(CuArray, data[9])

        for iesp in 1:length(data[0])
            push!(data_gpu[0], CUDA.convert(CuArray, data[0][iesp]))
            push!(data_gpu[2], CUDA.convert(CuArray, data[2][iesp]))
            push!(data_gpu[6], CUDA.convert(CuArray, data[6][iesp]))
            push!(data_gpu[7], CUDA.convert(CuArray, data[7][iesp]))
        end
    end

    if dataset.train_energy
        data_gpu[13] = data[13]
        data_gpu[11] = CUDA.convert(CuArray, data[11])
        data_gpu[14] = CUDA.convert(CuArray, data[14])

        for iesp in 1:length(data[10])
            push!(data_gpu[10], CUDA.convert(CuArray, data[10][iesp]))
            push!(data_gpu[12], CUDA.convert(CuArray, data[12][iesp]))
        end
    end

    return data_gpu
end