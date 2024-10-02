


function read_keyword_logical(keyword::String, lines::Vector{String})
    value = false
    found = false
    for line in lines
        read_keyword = split(line)[1]
        if lowercase(read_keyword) == keyword
            found = true
            value = true
            break
        end
    end
    return value, found
end

function read_keyword_argument_same_line(keyword::String, lines::Vector{String})
    value = nothing
    found = false
    for line in lines
        read_keyword = split(line)[1]
        if lowercase(read_keyword) == keyword
            found = true
            value = split(line)[2]
            break
        end
    end
    return value, found
end

function read_keyword_argument_next_line(keyword::String, lines::Vector{String})
    params = Dict{String,String}()
    found = false
    for (i, line) in enumerate(lines)
        read_keyword = split(line)[1]
        if lowercase(read_keyword) == keyword
            found = true
            for param in split(lines[i+1])
                p, val = split(param, "=")
                params[lowercase(p)] = val
            end
            break
        end
    end
    return params, found
end

function read_keyword_networks(lines::Vector{String})
    found = false
    networks_line = 0
    for (i, line) in enumerate(lines)
        read_keyword = split(line)[1]
        if lowercase(read_keyword) == "networks"
            found = true
            networks_line = i + 1
            break
        end
    end

    sys_species = []
    for line in lines[networks_line:end]
        push!(sys_species, split(line)[1])
    end
    N_species = length(sys_species)

    input_size = fill(0, N_species)
    hidden_size = fill([], N_species)
    activations = fill([], N_species)
    names = fill("", N_species)

    for line in lines[networks_line:networks_line+N_species-1]
        aux = split(line)
        specie = aux[1]
        name_i = aux[2]
        N_hidden = parse(Int, aux[3])

        hidden_size_i = []
        activations_i = []
        for i in aux[4:end]
            push!(hidden_size_i, parse(Int, split(i, ":")[1]))
            push!(activations_i, lowercase(split(i, ":")[2]))
        end

        specie_index = findfirst(isequal(specie), sys_species)

        hidden_size[specie_index] = hidden_size_i
        activations[specie_index] = activations_i
        names[specie_index] = name_i
    end

    networks_param = Dict("hidden_size" => hidden_size, "activations" => activations, "names" => names, "input_size" => input_size)
    return sys_species, networks_param
end

function read_train_in(infile::String)
    f = open(infile, "r")
    #open(infile, "r") do f
    # Initialize InputParameters with default values
    tin = InputParameters()

    # Remove comments from input file
    lines = readlines(f)
    lines = [line for line in lines if !(startswith(line, r"!|#") || isempty(split(line)))]

    # Compulsory parameters
    tin.train_file, _ = read_keyword_argument_same_line("trainingset", lines)
    test_split, _ = read_keyword_argument_same_line("testpercent", lines)
    tin.test_split = parse(Float64, test_split)
    epoch_size, _ = read_keyword_argument_same_line("iterations", lines)
    tin.epoch_size = parse(Int64, epoch_size)
    tin.method_param, _ = read_keyword_argument_next_line("method", lines)
    tin.sys_species, tin.networks_param = read_keyword_networks(lines)

    # Optional parameters
    pytorch_seed, found = read_keyword_argument_same_line("phseed", lines)
    if found
        tin.pytorch_seed = parse(Int, pytorch_seed)
    end

    numpy_seed, found = read_keyword_argument_same_line("npseed", lines)
    if found
        tin.numpy_seed = parse(Int, numpy_seed)
    end

    epoch_write, found = read_keyword_argument_same_line("iterwrite", lines)
    if found
        tin.epoch_write = parse(Int, epoch_write)
    end

    batch_size, found = read_keyword_argument_same_line("batchsize", lines)
    if found
        tin.batch_size = parse(Int, batch_size)
    end

    max_energy, found = read_keyword_argument_same_line("maxenergy", lines)
    if found
        tin.max_energy = parse(Float64, max_energy)
    end

    max_forces, found = read_keyword_argument_same_line("maxforces", lines)
    if found
        tin.max_forces = parse(Float64, max_forces)
    end

    mode, found = read_keyword_argument_same_line("mode", lines)
    if found
        tin.mode = mode
    end

    forces_param, found = read_keyword_argument_next_line("forces", lines)
    if found
        tin.train_forces = true
        tin.forces_param = forces_param
    end

    memory_mode, found = read_keyword_argument_same_line("memory_mode", lines)
    if found
        tin.memory_mode = memory_mode
    end

    save_energies, found = read_keyword_logical("save_energies", lines)
    if found
        tin.save_energies = save_energies
    end

    save_forces, found = read_keyword_logical("save_forces", lines)
    if found
        tin.save_forces = save_forces
    end

    verbose, found = read_keyword_logical("verbose", lines)
    if found
        tin.verbose = verbose
    end

    save_batches, found = read_keyword_logical("save_batches", lines)
    if found
        tin.save_batches = save_batches
    end

    load_batches, found = read_keyword_logical("load_batches", lines)
    if found
        tin.load_batches = load_batches
    end

    regularization, found = read_keyword_argument_same_line("regularization", lines)
    if found
        tin.regularization = parse(Float64, regularization)
    end
    close(f)

    initialize!(tin)

    return tin
    #end
end