using Random

mutable struct TrainSetParameter
    filename::String
    normalized::Bool
    E_scaling::Float64
    E_shift::Float64
    N_species::Int64
    sys_species::Vector{String}
    E_atomic::Vector{Float64}
    N_atom::Int64
    N_struc::Int64
    E_min::Float64
    E_max::Float64
    E_avg::Float64
    E_min_norm::Float64
    E_max_norm::Float64
    E_avg_norm::Float64
end

mutable struct FPSetupParameter
    N_species::Int64
    description::Vector{String}
    atomtype::Vector{String}
    nenv::Vector{Int64}
    envtypes::Vector{Vector{String}}
    rcmin::Vector{Float64}
    rcmax::Vector{Float64}
    sftype::Vector{String}
    nsfparam::Vector{Int64}
    nsf::Vector{Int64}
    sf::Vector{Vector{Float64}}
    sfparam::Vector{Vector{Float64}}
    sfenv::Vector{Vector{Float64}}
    neval::Vector{Int64}
    sfval_min::Vector{Vector{Float64}}
    sfval_max::Vector{Vector{Float64}}
    sfval_avg::Vector{Vector{Float64}}
    sfval_cov::Vector{Vector{Float64}}
end


mutable struct InputParameters
    train_file::String
    numpy_seed::Int64
    pytorch_seed::Int64
    epoch_write::Int64
    batch_size::Int64
    mode::String
    max_energy::Union{Bool,Nothing}
    max_forces::Union{Bool,Nothing}
    save_energies::Bool
    save_forces::Bool
    train_forces::Bool
    memory_mode::String
    alpha::Float64
    verbose::Bool
    save_batches::Bool
    load_batches::Bool
    test_split::Float64
    epoch_size::Int64
    method::String
    lr::Float64
    N_species::Int64
    sys_species::Vector{String}
    method_param::Dict{String,String}
    forces_param::Dict{String,String}
    original_batch_size::Int64
    train_forces_file::String
    networks_param::Dict{String,Any}
    regularization::Union{Nothing,Float64}
    trainset_params::Union{Nothing,TrainSetParameter}
    setup_params::Union{Nothing,FPSetupParameter}
end

function InputParameters()
    return InputParameters("", 11, 22, 1, 256, "train", nothing, nothing, false, false, false, "cpu", 0.0, false, false, false,
        0.0, 0, "", 0.0, 0, String[], Dict(), Dict(), 0, "", Dict(), nothing, nothing, nothing)
end

function initialize!(params::InputParameters)
    params.test_split /= 100
    params.epoch_size = Int(params.epoch_size)
    params.method = lowercase(params.method_param["method"])
    params.lr = parse(Float64, params.method_param["lr"])
    params.N_species = length(params.sys_species)
    params.original_batch_size = params.batch_size

    if params.train_forces
        params.train_forces_file = params.train_file[1:end-6] * ".forces"
        params.alpha = parse(Float64, params.forces_param["alpha"])
    end
end



function FPSetupParameter(N_species::Int)
    return FPSetupParameter(N_species,
        Vector{String}(undef, N_species),
        fill("", N_species), fill(0, N_species), Vector{Vector{String}}(undef, N_species),
        fill(0.0, N_species), fill(0.0, N_species), fill("", N_species), fill(0, N_species), fill(0, N_species),
        Vector{Vector{Float64}}(undef, N_species), Vector{Vector{Float64}}(undef, N_species),
        Vector{Vector{Float64}}(undef, N_species), fill(0, N_species), Vector{Vector{Float64}}(undef, N_species),
        Vector{Vector{Float64}}(undef, N_species), Vector{Vector{Float64}}(undef, N_species), Vector{Vector{Float64}}(undef, N_species))
end

function add_specie!(params::FPSetupParameter, iesp_0, description, atomtype, nenv, envtypes,
    rcmin, rcmax, sftype, nsf, nsfparam, sf,
    sfparam, sfenv, neval, sfval_min, sfval_max, sfval_avg, sfval_cov)
    iesp = iesp_0 + 1
    params.description[iesp] = description
    params.atomtype[iesp] = atomtype
    params.nenv[iesp] = nenv
    params.envtypes[iesp] = envtypes
    params.rcmin[iesp] = rcmin
    params.rcmax[iesp] = rcmax
    params.sftype[iesp] = sftype
    params.nsf[iesp] = nsf
    params.nsfparam[iesp] = nsfparam
    params.sf[iesp] = sf
    params.sfparam[iesp] = sfparam
    params.sfenv[iesp] = sfenv
    params.neval[iesp] = neval
    #println("sfval_min, $sfval_min")
    params.sfval_min[iesp] = sfval_min
    params.sfval_max[iesp] = sfval_max
    params.sfval_avg[iesp] = sfval_avg
    params.sfval_cov[iesp] = sfval_cov
end


function TrainSetParameter(filename, normalized,
    E_scaling, E_shift, N_species,
    sys_species, E_atomic, N_atom,
    N_struc, E_min,
    E_max, E_avg)
    return TrainSetParameter(filename, normalized, E_scaling, E_shift, N_species, sys_species, E_atomic, N_atom, N_struc, E_min, E_max, E_avg, 0, 0, 0)
end

function get_E_normalization!(trainset::TrainSetParameter)

    E_scaling = 2 / (trainset.E_max - trainset.E_min)
    E_shift = 0.5 * (trainset.E_max + trainset.E_min)

    trainset.E_min_norm = (trainset.E_min - E_shift) * E_scaling
    trainset.E_max_norm = (trainset.E_max - E_shift) * E_scaling
    trainset.E_avg_norm = (trainset.E_avg - E_shift) * E_scaling

    trainset.E_scaling = E_scaling
    trainset.E_shift = E_shift
    trainset.normalized = true

    return E_scaling, E_shift
end

mutable struct Structure
    name::String
    energy::Float64
    E_atomic_structure::Float64
    train_forces::Bool
    N_species::Int64
    N_ions::Vector{Int64}
    N_atom::Int64
    species::Vector{Int64}
    descriptor::Vector{Vector{Vector{Float64}}}
    forces::Vector{Vector{Float64}}
    coords::Vector{Vector{Float64}}
    max_nb_struc::Union{Nothing,Int64}
    list_nblist::Union{Nothing,Vector{Vector{Vector{Int64}}}}
    list_sfderiv_i::Union{Nothing,Vector{Array{Float64,3}}}
    list_sfderiv_j::Union{Nothing,Vector{Array{Float64,4}}}
    order::Union{Nothing,Array{Int64}}


end

function Structure(name, species, descriptor, energy,
    energy_atom_struc, sys_species, coords,
    forces, input_size, train_forces=false,
    list_nblist=nothing, list_sfderiv_i=nothing, list_sfderiv_j=nothing)

    N_species = length(sys_species)
    N_ions = fill(0, N_species)

    for iesp in species
        N_ions[iesp+1] += 1
    end

    N_atom = sum(N_ions)
    #descriptor_data = [Vector{Vector{Float64}}(undef, N_ions[iesp]) for iesp in 1:N_species]
    descriptor_data = [[Float64[] for iat in 1:N_ions[iesp]] for iesp in 1:N_species]
    #println(typeof(descriptor_data))
    #descriptor_data = [zeros(input_size[i]) for i in 1:N_species]
    forces_tensor = [zeros(3) for iesp in 1:N_atom]#zeros(N_atom)
    coords_tensor = [zeros(3) for iesp in 1:N_atom]#zeros(N_atom)

    cont_iesp = fill(0, N_species)
    cont = 1
    for iesp in 1:N_species
        for iat in 1:N_atom
            if species[iat] == iesp - 1
                #println(descriptor[iat])
                #println(descriptor_data[iesp][cont_iesp[iesp]+1])
                #println(typeof(descriptor[iat]))
                #println(typeof(descriptor_data[iesp][cont_iesp[iesp]+1]))
                descriptor_data[iesp][cont_iesp[iesp]+1] = descriptor[iat]
                #push!(descriptor_data[iesp], descriptor[iat])
                forces_tensor[cont] = forces[iat]
                coords_tensor[cont] = coords[iat]
                cont_iesp[iesp] += 1
                cont += 1
            end
        end
    end
    self = Structure(name, energy, energy_atom_struc, train_forces, N_species, N_ions, N_atom, species, descriptor_data, forces_tensor, coords_tensor, nothing, nothing, nothing, nothing, nothing)

    if train_forces

        self.max_nb_struc = maximum([length(x) for x in list_nblist])

        self.order = fill(0, self.N_atom)
        self.list_nblist = [Vector{Vector{Int64}}(undef, 0) for iesp in 1:self.N_species]
        #self.list_nblist = [Vector{Vector{Int64}}(undef, 0) for iesp in 1:self.N_species]
        #error(typeof(self.list_nblist))
        #self.list_nblist = [[Int64[] for iat in 1:N_ions[iesp]] for iesp in 1:N_species]
        self.list_sfderiv_i = [zeros(self.N_ions[iesp], input_size[iesp], 3) for iesp in 1:self.N_species]
        self.list_sfderiv_j = [zeros(self.N_ions[iesp], self.max_nb_struc, input_size[iesp], 3) for iesp in 1:self.N_species]

        cont_iesp = fill(0, self.N_species)
        cont = 0

        for iesp in 1:self.N_species
            for iat in 1:self.N_atom
                if species[iat] == iesp - 1
                    nnb = length(list_nblist[iat])
                    self.list_sfderiv_i[iesp][cont_iesp[iesp]+1, :, :] = list_sfderiv_i[iat]
                    self.list_sfderiv_j[iesp][cont_iesp[iesp]+1, 1:nnb, :, :] = list_sfderiv_j[iat]
                    #println(list_nblist[iat])
                    #println(typeof(list_nblist[iat]))
                    #println(self.list_nblist[iesp])
                    #println("t ", typeof(self.list_nblist[iesp]))
                    #println(list_nblist[iat])
                    #println(self.list_nblist[iesp])
                    push!(self.list_nblist[iesp], list_nblist[iat]) #
                    #println(self.list_nblist[iesp])
                    self.order[iat] = cont

                    cont_iesp[iesp] += 1
                    cont += 1
                end
            end
            #error("w")
        end
    end
    return self

end

function padding!(structure::Structure, max_nnb::Int, input_size::Vector{Int})
    """
    Add trailing zeros to make all tensors of the same size
    max_nb_struc :: Maximum number of neighbors in the whole data set
    input_size   :: Size of the descriptors for each species
    """
    for iesp in 1:structure.N_species
        if structure.N_ions[iesp] == 0
            structure.descriptor[iesp] = zeros(0, input_size[iesp])
        end
    end

    if structure.train_forces
        for iesp in 1:structure.N_species
            if structure.N_ions[iesp] == 0
                structure.list_sfderiv_i[iesp] = zeros(structure.N_ions[iesp], input_size[iesp], 3)
                structure.list_sfderiv_j[iesp] = zeros(structure.N_ions[iesp], max_nnb, input_size[iesp], 3)
            end
        end

        aux_nblist = [[fill(-1000000000, max_nnb) for k = 1:structure.N_ions[iesp]] for iesp in 1:structure.N_species]
        #aux_nblist = [fill(-1000000000, structure.N_ions[iesp], max_nnb) for iesp in 1:structure.N_species]
        aux_sfderiv_j = [zeros(structure.N_ions[iesp], max_nnb, input_size[iesp], 3) for iesp in 1:structure.N_species]

        for iesp in 1:structure.N_species
            for iat in 1:structure.N_ions[iesp]
                nnb = length(structure.list_nblist[iesp][iat])
                #println("dd ", structure.list_nblist[iesp][iat])
                #aux_nblist[iesp][iat, 1:nnb] = structure.list_nblist[iesp][iat] .- 1
                aux_nblist[iesp][iat][1:nnb] = structure.list_nblist[iesp][iat] .- 1
                #println("iesp,iat $iesp ", iat)
                #println(typeof(iat))
                #println([iat, 1:nnb, :, :])
                #println(typeof(aux_sfderiv_j[iesp]))
                #println(size(aux_sfderiv_j[iesp]))
                #println(size(structure.list_sfderiv_j[iesp]))
                #println(aux_sfderiv_j[iesp][iat, 1:nnb, :, :])
                #println(size(structure.list_sfderiv_j[iesp][iat][1:nnb, :, :]))
                aux_sfderiv_j[iesp][iat, 1:nnb, :, :] = structure.list_sfderiv_j[iesp][iat, 1:nnb, :, :]
            end
        end

        sorted_nblist = deepcopy(aux_nblist)
        N_atom_total = length(structure.order)
        for iat in 1:N_atom_total
            if structure.order[iat] != iat
                for iesp in 1:structure.N_species
                    sorted_nblist[iesp] = ifelse.(aux_nblist[iesp] .== iat, structure.order[iat], sorted_nblist[iesp])
                end
            end
        end

        #println(typeof(sorted_nblist))
        #println(typeof(structure.list_nblist))
        #println(sorted_nblist)
        structure.list_nblist = sorted_nblist
        structure.list_sfderiv_j = aux_sfderiv_j
    end
end