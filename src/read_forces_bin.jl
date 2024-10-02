using LinearAlgebra



function tf_read_header(tf::IO, sys_species::Vector{String})
    N_species = parse(Int, readline(tf))
    N_struc = parse(Int, readline(tf))

    species_names = split(readline(tf))
    aux_E_atomic = map(x -> parse(Float64, x), split(readline(tf)))

    species_index = []
    E_atomic = []

    for i in 1:N_species
        index = findfirst(isequal(species_names[i]), sys_species) - 1
        push!(species_index, index)
        push!(E_atomic, aux_E_atomic[index+1])
    end

    normalized = false
    if readline(tf) == "T"
        normalized = true
    end

    E_scaling = parse(Float64, readline(tf))
    E_shift = parse(Float64, readline(tf))

    return N_species, N_struc, species_index, E_atomic, normalized, E_scaling, E_shift
end

function tf_read_struc_info(tf::IO, species_index, E_atomic)
    coords = []
    forces = []
    descriptors = []
    species = []

    length = parse(Int, readline(tf))
    name = strip(readline(tf))
    aux = split(readline(tf))
    N_at, N_sp = parse.(Int, aux)
    E = parse(Float64, readline(tf))

    E_atomic_structure = 0.0
    for iatom in 1:N_at
        sp = parse(Int, readline(tf)) - 1
        push!(species, species_index[sp+1])
        E_atomic_structure += E_atomic[species_index[sp+1]+1]

        push!(coords, map(x -> parse(Float64, x), split(readline(tf))))
        push!(forces, map(x -> parse(Float64, x), split(readline(tf))))
        readline(tf)  # skip an empty line
        push!(descriptors, map(x -> parse(Float64, x), split(readline(tf))))
    end

    return name, E, E_atomic_structure, species, coords, forces, descriptors
end

function tf_read_footer(tf::IO, N_species::Int, species_index)
    setup_params = FPSetupParameter(N_species)
    input_size = fill(0, N_species)

    natomstot = parse(Int, readline(tf))
    E_avg, E_min, E_max = parse.(Float64, split(readline(tf)))

    has_setups = strip(readline(tf))

    for iesp in 1:N_species
        sp = parse(Int, readline(tf)) - 1
        specie_index = species_index[sp+1]

        description = strip(readline(tf))
        atomtype = strip(readline(tf))
        nenv = parse(Int, readline(tf))

        envtypes = []
        for _ in 1:nenv
            push!(envtypes, strip(readline(tf)))
        end

        rcmin = parse(Float64, readline(tf))
        rcmax = parse(Float64, readline(tf))
        sftype = strip(readline(tf))
        nsf = parse(Int, readline(tf))
        nsfparam = parse(Int, readline(tf))

        sf = parse.(Int, split(readline(tf)))
        sfparam = parse.(Float64, split(readline(tf)))
        sfenv = parse.(Int, split(readline(tf)))

        neval = parse(Int, readline(tf))

        sfval_min = parse.(Float64, split(readline(tf)))
        sfval_max = parse.(Float64, split(readline(tf)))
        sfval_avg = parse.(Float64, split(readline(tf)))
        sfval_cov = parse.(Float64, split(readline(tf)))

        if length(description) > 1024
            description = description[1:1024]
        end

        add_specie!(setup_params, specie_index, description, atomtype, nenv, envtypes, rcmin, rcmax, sftype, nsf, nsfparam,
            sf, sfparam, sfenv, neval, sfval_min, sfval_max, sfval_avg, sfval_cov)

        input_size[specie_index+1] = nsf
    end

    return natomstot, E_avg, E_min, E_max, setup_params, input_size
end

function tff_read_integer(tff::IO, N)
    pad = read(tff, Int32)  # read the length of the record
    println(pad)
    result = []
    for i = 1:N
        push!(result, read(tff, Int32))
    end
    #result = read(tff, Int32, N)
    pad = read(tff, Int32)  # read the length of the record

    if N == 1
        return result[1]
    else
        return result
    end
end

function tff_read_real8(tff::IO, N::Int)
    pad = read(tff, Int32)  # read the length of the record
    result = []
    for i = 1:N
        push!(result, read(tff, Float64))
    end
    #result = read(tff, Float64, N)
    pad = read(tff, Int32)  # read the length of the record

    if N == 1
        return result[1]
    else
        return result
    end
end

function tff_read_character(tff::IO, N)
    pad = read(tff, Int32)  # read the length of the record
    result = UInt8[]
    for i = 1:N
        tmp = read(tff, UInt8)
        #println(parse(String, tmp))
        push!(result, tmp)
        #result = String(read(tff, UInt8, N))
    end
    #println("N = $N")
    #println(result)
    #println(String(result))
    result_st = String(result)
    pad = read(tff, Int32)  # read the length of the record

    return result_st
end

function tff_read_header(tff::IO)
    N_struc = tff_read_integer(tff, 1)
    return N_struc
end

function tff_read_struc_info_grads(tff::IO, species_index, max_nnb)
    list_nblist = []
    list_sfderiv_i = []
    list_sfderiv_j = []

    length = tff_read_integer(tff, 1)
    name = tff_read_character(tff, length)
    N_at, N_sp = tff_read_integer(tff, 2)
    train_forces_struc = Bool(tff_read_integer(tff, 1))

    if train_forces_struc == 1
        for iatom in 1:N_at
            specie = tff_read_integer(tff, 1) - 1
            nsf, nnb = tff_read_integer(tff, 2)
            nblist = tff_read_integer(tff, nnb)
            sfderiv_i = tff_read_real8(tff, 3 * nsf)
            sfderiv_j = tff_read_real8(tff, 3 * nsf * nnb)

            sfderiv_i = reshape(sfderiv_i, (nsf, 3))
            sfderiv_j = reshape(sfderiv_j, (nnb, nsf, 3))

            if nnb == 1
                push!(list_nblist, [nblist])
            else
                push!(list_nblist, nblist)
            end
            push!(list_sfderiv_i, sfderiv_i)
            push!(list_sfderiv_j, sfderiv_j)

            index = species_index[specie+1]
            max_nnb[index+1] = max(max_nnb[index+1], nnb)
        end

        return train_forces_struc, max_nnb, list_nblist, list_sfderiv_i, list_sfderiv_j
    else
        return train_forces_struc, max_nnb, nothing, nothing, nothing
    end
end