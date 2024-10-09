using Dates
using Printf
using Random
using Statistics

function io_print(text)
    println(text)
    flush(stdout)
end

function io_print_center(text)
    lenght = 70
    N_blank = lenght - length(text)
    aux = divrem(N_blank, 2)
    a = aux[1]
    b = aux[1]
    if aux[2] != 0
        a += 1
    end
    io_print(" "^a * text * " "^b)
end

function io_current_time()
    aux = string(Dates.now())[1:end-7]
    io_print_center(aux)
end

function io_line()
    io_print("----------------------------------------------------------------------")
end

function io_print_title(text)
    io_line()
    io_print_center(text)
    io_line()
    io_print("")
end

function io_double_line()
    io_print("======================================================================")
end

function io_print_header()
    io_double_line()
    io_print_center("Training with aenet-PyTorch")
    io_double_line()
    io_print("")
    io_print("")
    io_current_time()
    io_print("")
    io_print("")
    io_print("Developed by Jon Lopez-Zorrilla")
    io_print("")
    io_print("This program is distributed in the hope that it will be useful,")
    io_print("but WITHOUT ANY WARRANTY; without even the implied warranty of")
    io_print("MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the")
    io_print("GNU General Public License in file 'LICENSE' for more details.")
    io_print("")
end

function io_input_reading(tin)
    io_print_title("Reading input information")

    io_print("Reading input parameters.")
    io_print("These are the parameters selected for training:")
    io_print("        - TRAININGSET: $(tin.train_file)")
    io_print("        - TESTPERCENT: $(tin.test_split)")
    io_print("        - ITERATIONS:  $(tin.epoch_size)")
    io_print("        - ITERWRITE:   $(tin.epoch_write)")
    io_print("        - BATCHSIZE:   $(tin.batch_size)")
    io_print("        - MEMORY_MODE: $(tin.memory_mode)")
    io_print("")

    if tin.train_forces
        io_print("        - FORCES:      $(tin.train_forces)")
        io_print("        - alpha:       $(tin.alpha)")
        if tin.max_forces != nothing
            io_print("        - maxforces:   $(tin.max_forces)")
        end
        io_print("")
    end
end

function io_network_initialize(tin)
    io_print_title("Networks")

    if isfile("./model.restart")
        io_print("Previous run files found. The training will be restarted from")
        io_print("that checkpoint.")
    else
        io_print("Training will be started from scratch.")
    end
    io_print("Initializing networks:")
    for iesp in 1:tin.N_species
        io_print("")
        io_print("Creating a network for $(tin.sys_species[iesp])")
        io_print("")
        io_print("Number of layers: $(length(tin.networks_param["hidden_size"][iesp])+2)")
        io_print("")
        io_print("Number of nodes and activation type per layer:")
        io_print("")
        io_print("    1 : $(tin.networks_param["input_size"][iesp])")
        for ilayer in 1:length(tin.networks_param["hidden_size"][iesp])
            io_print("   $(ilayer + 1) : $(tin.networks_param["hidden_size"][iesp][ilayer])   $(tin.networks_param["activations"][iesp][ilayer])")
        end
        ilayer = length(tin.networks_param["hidden_size"][iesp])
        io_print("   $(ilayer + 2) : 1")
        io_print("")
    end
end

function io_trainingset_information()
    io_print_title("Reading training set information")

    io_print("Training set information will be read now. If force training is")
    io_print("required this process may take some time.")
    io_print("")
end

function io_trainingset_information_done(tin, trainset_params, N_struc_E, N_struc_F=0, N_removed=0)
    io_print("The network output energy will be normalized to the interval [-1,1].")
    io_print("    Energy scaling factor:  f = $(trainset_params.E_scaling)")
    io_print("    Atomic energy shift  :  s = $(trainset_params.E_shift)")
    io_print("")
    if N_removed != 0
        io_print("$(N_removed) high-energy structures will be removed from the training set.")
    end
    io_print("")
    io_print("Number of structures in the data set: $(N_struc_E + N_struc_F)")
    if N_struc_F != 0
        io_print("Number of structures with force information: $(N_struc_F)")
    end
    io_print("")
    fmt = join(tin.sys_species, " ")
    io_print("Atomic species in the training set: $fmt")
    io_print("")
    io_print("Average energy (eV/atom) : $(trainset_params.E_avg)")
    io_print("Minimum energy (eV/atom) : $(trainset_params.E_min)")
    io_print("Maximum energy (eV/atom) : $(trainset_params.E_max)")
    io_print("")
end

function io_prepare_batches()
    io_print_title("Preparing batches for training")

    io_print("Batches for training are being prepared now. If force training is")
    io_print("required, this may take some time.")
    io_print("")
    io_print("If the number of structures is not divisible by the batch size, the actual")
    io_print("batch size may be slightly changed.")
    io_print("")
end

function io_prepare_batches_done(tin, train_energy_data, train_forces_data)
    mean_batch_size_E = 0
    mean_batch_size_F = 0

    if train_energy_data != nothing
        #display(train_energy_data.indexes)
        aux = train_energy_data.indexes[:, 2] - train_energy_data.indexes[:, 1]
        #display(aux)
        mean_batch_size_E = round(mean(aux))
    end

    if train_forces_data != nothing

        aux = train_forces_data.indexes[:, 2] - train_forces_data.indexes[:, 1]
        mean_batch_size_F = round(mean(aux))
    end

    io_print("Requested batch size: $(tin.original_batch_size)")
    io_print("Actual batch size   : $(mean_batch_size_F + mean_batch_size_E)")
    io_print("Number of batches   : $(train_energy_data.N_batch)")
    io_print("")

    if train_energy_data != nothing && train_forces_data != nothing
        io_print("Energy batch size   : $(mean_batch_size_E)")
        io_print("Forces batch size   : $(mean_batch_size_F)")
    end
    io_print("")
end

function io_save_batches()
    io_print_title("Saving batch information")

    io_print("Information about the batches will be saved now. If in the next run")
    io_print("load_batches=true and the folder 'tmp_batches' is present, ")
    io_print("the information will be read from there, saving time.")
    io_print("")
    io_print("If the batch size is changed, please remove that folder, or the training")
    io_print("may fail or not work as expected.")
    io_print("")
end

function io_load_batches()
    io_print_title("Loading batch information")

    io_print("Loading batch information from 'tmp_batches'. If the batch size has")
    io_print("been changed, the program will not work as expected.")
    io_print("")
end

function io_train_details(tin, device)
    io_print_title("Training details")

    io_print("Training method : $(tin.method)")
    io_print("Learning rate   : $(tin.lr)")
    io_print("Regularization  : $(tin.regularization)")
    io_print("")
    io_print("Training device : $(device)")
    io_print("Memory mode     : $(tin.memory_mode)")
    io_print("")
end

function io_train_start()
    io_print_title("Training process")

    fmt2 = "{:>10} :  {:>12}  {:>12}   |{:>12}  {:>12}   |{:>12}  {:>12}"
    io_print(fmt2)
    io_print("     epoch :  ERROR(train)   ERROR(test)   |   E (train)      E (test)   |   F (train)      F (test)")
    io_print("     -----    ------------   -----------        ---------      --------        ---------      --------")
end

function io_train_step(epoch, train_error, valid_error, train_E_error, valid_E_error, train_F_error, valid_F_error, E_scaling)
    fmt2 = "{: 10d} :  {: 12.6f}  {: 12.6f}   |{: 12.6f}  {: 12.6f}   |{: 12.6f}  {: 12.6f}"
    train_error /= E_scaling
    valid_error /= E_scaling
    train_E_error /= E_scaling
    valid_E_error /= E_scaling
    train_F_error /= E_scaling
    valid_F_error /= E_scaling
    io_print(Printf.format(fmt2, epoch, train_error, valid_error, train_E_error, valid_E_error, train_F_error, valid_F_error))
end

function io_train_finalize(t, mem_CPU, mem_GPU)
    io_print("")
    io_print("Time needed for training:  $(time() - t) s")
    io_print("Maximum CPU memory used:   $(mem_CPU) GB")
    io_print("Maximum GPU memory used:   $(mem_GPU) GB")
    io_print("")
end

function io_save_networks(tin)
    io_print_title("Storing results")

    io_print("saving train energy error to : energy.train")
    io_print("saving test energy error to  : energy.test")
    io_print("")

    for iesp in 1:tin.N_species
        io_print("Saving the $(tin.sys_species[iesp]) network to file : $(tin.networks_param["names"][iesp]).ascii")
    end
end

function io_save_error_iteration(tin, E_scaling, iter_error_trn, iter_error_tst)
    iter_error_tst = iter_error_tst / E_scaling
    open("train.error", "w") do f
        fmt1 = "{: 10d}   {: 12.6f}  {: 12.6f}   {: 12.6f}  {: 12.6f}   {: 12.6f}  {: 12.6f}\n"
        fmt2 = "{: 10d}   {: 12.6f}  {: 12.6f}\n"
        iter_i = Int.(iter_error_tst[:, 1])
        error = iter_error_tst[:, 2:end] / E_scaling
        for i in 1:length(iter_i)
            if tin.train_forces
                f.write(Printf.format(fmt1, iter_i[i], error[i, :]))
            else
                f.write(Printf.format(fmt2, iter_i[i], error[i, 1], error[i, 2]))
            end
        end
    end
end

function io_footer()
    io_print("")
    io_print("")
    io_current_time()
    io_print("")
    io_print("")
    io_double_line()
    io_print_center("Neural Network training done.")
    io_double_line()
end