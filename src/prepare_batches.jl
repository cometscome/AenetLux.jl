using Random
using BSON: @save, @load

function read_list_structures(tin)
    """
    Read Training set files (*.train.ascii and *.train.forces)
    """
    if tin.train_forces
        list_structures_forces, list_structures_energy, list_removed, max_nnb, tin = read_train_forces_together(tin)
    else
        list_structures_energy, list_removed, max_nnb, tin = read_train(tin)
        list_structures_forces = []
        max_nnb = 0
    end

    input_size = tin.networks_param["input_size"]
    for struc in list_structures_energy
        padding!(struc, max_nnb, input_size)
    end
    for struc in list_structures_forces
        padding!(struc, max_nnb, input_size)
    end

    return list_structures_energy, list_structures_forces, list_removed, max_nnb, tin
end

function get_N_batch(len_dataset, batch_size)
    """
    Returns the number of batches for a given batch size and dataset size
    """
    N_batch = div(len_dataset, batch_size)
    residue = len_dataset - N_batch * batch_size

    if residue >= div(batch_size, 2) || N_batch == 0
        if residue != 0
            N_batch += 1
        end
    end

    return N_batch
end

function split_database(dataset_size, test_split)
    """
    Returns indices of the structures in the training and testing sets
    """
    indices = collect(1:dataset_size)
    shuffle!(indices)
    split = floor(Int, test_split * dataset_size)

    train_indices = indices[split+1:end]
    test_indices = indices[1:split]

    return train_indices, test_indices
end

function select_batch_size(tin, list_structures_energy, list_structures_forces)
    """
    Select batch size that best matches the requested size, avoiding the last batch being too small
    """
    N_data_E = length(list_structures_energy)
    N_data_F = length(list_structures_forces)
    train_sampler_E, valid_sampler_E = split_database(N_data_E, tin.test_split)
    train_sampler_F, valid_sampler_F = split_database(N_data_F, tin.test_split)

    forcespercent = N_data_F / (N_data_F + N_data_E)

    if forcespercent <= 0.5
        tin.batch_size = round(Int, (1 - forcespercent) * tin.batch_size)

        N_batch_train = get_N_batch(N_data_train_E, tin.batch_size)
        N_batch_valid = get_N_batch(N_data_valid_E, tin.batch_size)
    else
        tin.batch_size = round(Int, forcespercent * tin.batch_size)

        N_batch_train = get_N_batch(N_data_train_F, tin.batch_size)
        N_batch_valid = get_N_batch(N_data_valid_F, tin.batch_size)
    end

    if N_batch_train > N_data_F
        N_batch_train = N_data_F
    end

    if N_batch_valid > N_data_F
        N_batch_valid = N_data_F
    end

    return N_batch_train, N_batch_valid
end

function select_batches(tin, trainset_params, device, list_structures_energy, list_structures_forces,
    max_nnb, N_batch_train, N_batch_valid)
    """
    Select which structures belong to each batch for training.
    Returns: four objects of the class data_set_loader.PrepDataloader(), for train/test and energy/forces
    """
    if !isempty(list_structures_energy)
        dataset_energy = StructureDataset(list_structures_energy, tin.sys_species, tin.networks_param["input_size"], max_nnb)
        dataset_energy_size = length(dataset_energy)

        # Normalize
        E_scaling, E_shift = tin.trainset_params.E_scaling, tin.trainset_params.E_shift
        sfval_avg, sfval_cov = tin.setup_params.sfval_avg, tin.setup_params.sfval_cov
        normalize_E!(dataset_energy, trainset_params.E_scaling, trainset_params.E_shift)
        stp_shift, stp_scale = normalize_stp!(dataset_energy, sfval_avg, sfval_cov)

        # Split in train/test
        train_sampler_E, valid_sampler_E = split_database(dataset_energy_size, tin.test_split)

        train_energy_data = PrepDataloader(dataset=dataset_energy, train_forces=false, N_batch=N_batch_train,
            sampler=train_sampler_E, memory_mode=tin.memory_mode, device=device, dataname="train_energy")
        valid_energy_data = PrepDataloader(dataset=dataset_energy, train_forces=false, N_batch=N_batch_valid,
            sampler=valid_sampler_E, memory_mode=tin.memory_mode, device=device, dataname="valid_energy")
    else
        dataset_energy = nothing
        train_energy_data, valid_energy_data = nothing, nothing
    end

    if !isempty(list_structures_forces)
        dataset_forces = StructureDataset(list_structures_forces, tin.sys_species, tin.networks_param["input_size"], max_nnb)
        dataset_forces_size = length(dataset_forces)

        # Normalize
        normalize_E!(dataset_forces, trainset_params.E_scaling, trainset_params.E_shift)
        normalize_F!(dataset_forces, trainset_params.E_scaling, trainset_params.E_shift)
        stp_shift, stp_scale = normalize_stp!(dataset_forces, sfval_avg, sfval_cov)

        # Split in train/test
        train_sampler_F, valid_sampler_F = split_database(dataset_forces_size, tin.test_split)

        train_forces_data = PrepDataloader(dataset=dataset_forces, train_forces=true, N_batch=N_batch_train,
            sampler=train_sampler_F, memory_mode=tin.memory_mode, device=device, dataname="train_forces")
        valid_forces_data = PrepDataloader(dataset=dataset_forces, train_forces=true, N_batch=N_batch_valid,
            sampler=valid_sampler_F, memory_mode=tin.memory_mode, device=device, dataname="valid_forces")
    else
        dataset_forces = nothing
        train_forces_data, valid_forces_data = nothing, nothing
    end

    return train_forces_data, valid_forces_data, train_energy_data, valid_energy_data
end

function save_datasets(save, train_forces_data, valid_forces_data, train_energy_data, valid_energy_data)
    """
    Saves datasets created by select_batches
    """
    @save "tmp_batches/trainset_info.bson" save
    @save "tmp_batches/train_forces_data.bson" train_forces_data
    @save "tmp_batches/valid_forces_data.bson" valid_forces_data
    @save "tmp_batches/train_energy_data.bson" train_energy_data
    @save "tmp_batches/valid_energy_data.bson" valid_energy_data
end

function load_datasets(tin, device)
    """
    Loads saved datasets instead of preparing them
    """
    save = @load "tmp_batches/trainset_info.bson"
    N_removed, N_struc_E, N_struc_F, max_nnb, tin.trainset_params, tin.setup_params, tin.networks_param = save[:]

    train_forces_data = @load "tmp_batches/train_forces_data.bson"
    train_energy_data = @load "tmp_batches/train_energy_data.bson"
    gather_data!(train_forces_data, tin.memory_mode)
    gather_data!(train_energy_data, tin.memory_mode)

    grouped_train_data = GroupedDataset(train_energy_data, train_forces_data,
        memory_mode=tin.memory_mode, device=device, dataname="train")

    valid_forces_data = @load "tmp_batches/valid_forces_data.bson"
    valid_energy_data = @load "tmp_batches/valid_energy_data.bson"
    gather_data!(valid_forces_data, tin.memory_mode)
    gather_data!(valid_energy_data, tin.memory_mode)

    grouped_valid_data = GroupedDataset(valid_energy_data, valid_forces_data,
        memory_mode=tin.memory_mode, device=device, dataname="valid")

    return N_removed, N_struc_E, N_struc_F, max_nnb, grouped_train_data, grouped_valid_data
end