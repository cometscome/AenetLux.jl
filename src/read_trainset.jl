using Base.Iterators
using LinearAlgebra

function read_train_forces_together(tin)
    """
    Read training set files when force training is requested
    Returns
        list_struct_forces :: List of Structure objects included in force training
        list_struct_energy :: List of Structure objects with only energy information
        list_removed       :: List of names of the structures above the maximum energy cutoff
        max_nnb            :: Maximum number of neighbors in the data set
        tin                :: Updated input parameters
    """
    trainfile = tin.train_file
    forcesfile = tin.train_forces_file
    sys_species = tin.sys_species
    max_energy = tin.max_energy
    max_forces = tin.max_forces

    list_removed = []
    E_max_min_avg = [-1e7, 1e7, 0.0]

    open(trainfile, "r") do tf
        open(forcesfile, "r") do tff
            # Header
            N_species, N_struc, species_index, E_atomic, normalized, E_scaling, E_shift = tf_read_header(tf, sys_species)
            _ = tff_read_header(tff)

            # Footer (Fingerprint Setup information)
            natomstot, E_avg, E_min, E_max, setup_params, input_size = tf_read_footer(tf, N_species, species_index)

            trainset_params = TrainSetParameter(trainfile, normalized, E_scaling, E_shift, N_species, sys_species, E_atomic,
                natomstot, N_struc, E_min, E_max, E_avg)

            # Structures in dataset
            list_struct_forces = Structure[]
            list_struct_energy = Structure[]
            max_nnb = fill(0, length(species_index))
            for istruc in 1:N_struc
                name, E, E_atomic_structure, species, coords, forces, descriptors = tf_read_struc_info(tf, species_index, E_atomic)
                train_forces_struc, max_nnb, list_nblist, list_sfderiv_i, list_sfderiv_j = tff_read_struc_info_grads(tff, species_index, max_nnb)

                E_per_atom = E / length(coords)
                if max_energy != nothing && E_per_atom > max_energy
                    push!(list_removed, name)
                else
                    E_max_min_avg[1] = max(E_max_min_avg[1], E_per_atom)
                    E_max_min_avg[2] = min(E_max_min_avg[2], E_per_atom)
                    E_max_min_avg[3] += E_per_atom


                    if train_forces_struc
                        # Check if F < Fmax
                        #print(forces)
                        F_max_struc = map(x -> maximum(abs.(x)), forces)
                        #F_max_struc = maximum(maximum(abs.(forces)))
                        if max_forces == nothing || F_max_struc < max_forces
                            push!(list_struct_forces, Structure(name, species, descriptors, E, E_atomic_structure, sys_species,
                                coords, forces, input_size, train_forces_struc,
                                list_nblist, list_sfderiv_i, list_sfderiv_j))
                        else
                            train_forces_struc = false
                            push!(list_struct_energy, Structure(name, species, descriptors, E, E_atomic_structure, sys_species,
                                coords, forces, input_size, train_forces_struc))
                        end
                    else
                        push!(list_struct_energy, Structure(name, species, descriptors, E, E_atomic_structure, sys_species,
                            coords, forces, input_size, train_forces_struc))
                    end
                end
            end

            # Recompute E_scfaling and E_shift if some structures have been excluded
            trainset_params.E_max = E_max_min_avg[1]
            trainset_params.E_min = E_max_min_avg[2]
            trainset_params.E_avg = E_max_min_avg[3] / (length(list_struct_forces) + length(list_struct_energy))
            get_E_normalization!(trainset_params)

            max_nnb = maximum(max_nnb)
            tin.trainset_params = trainset_params
            tin.setup_params = setup_params
            tin.networks_param["input_size"] = input_size

            return list_struct_forces, list_struct_energy, list_removed, max_nnb, tin
        end
    end
end

function read_train(tin)
    """
    Read training set files with only energy training
    Returns
        list_struct_energy :: List of Structure objects with only energy information
        list_removed       :: List of names of the structures above the maximum energy cutoff
        max_nnb            :: Maximum number of neighbors in the data set
        tin                :: Updated input parameters
    """
    trainfile = tin.train_file
    sys_species = tin.sys_species
    max_energy = tin.max_energy

    list_removed = []
    E_max_min_avg = [-1e7, 1e7, 0.0]

    open(trainfile, "r") do tf
        # Header
        N_species, N_struc, species_index, E_atomic, normalized, E_scaling, E_shift = tf_read_header(tf, sys_species)

        # Footer (Fingerprint Setup information)
        natomstot, E_avg, E_min, E_max, setup_params, input_size = tf_read_footer(tf, N_species, species_index)

        trainset_params = TrainSetParameter(trainfile, normalized, E_scaling, E_shift, N_species, sys_species, E_atomic,
            natomstot, N_struc, E_min, E_max, E_avg)

        # Structures in dataset
        list_struct_energy = []
        max_nnb = fill(0, length(species_index))
        for istruc in 1:N_struc
            name, E, E_atomic_structure, species, coords, forces, descriptors = tf_read_struc_info(tf, species_index, E_atomic)

            E_per_atom = E / length(coords)
            if max_energy != nothing && E_per_atom > max_energy
                push!(list_removed, name)
            else
                E_max_min_avg[1] = max(E_max_min_avg[1], E_per_atom)
                E_max_min_avg[2] = min(E_max_min_avg[2], E_per_atom)
                E_max_min_avg[3] += E_per_atom


                push!(list_struct_energy, Structure(name, species, descriptors, E, E_atomic_structure, sys_species, coords, forces, input_size))
            end
        end

        # Recompute E_scaling and E_shift if some structures have been excluded
        trainset_params.E_max = E_max_min_avg[1]
        trainset_params.E_min = E_max_min_avg[2]
        trainset_params.E_avg = E_max_min_avg[3] / (length(list_struct_energy))
        get_E_normalization!(trainset_params)

        max_nnb = maximum(max_nnb)
        tin.trainset_params = trainset_params
        tin.setup_params = setup_params
        tin.networks_param["input_size"] = input_size

        return list_struct_energy, list_removed, max_nnb, tin
    end
end