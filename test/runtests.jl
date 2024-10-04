using AenetLux
using Test

function test()
    inputfilename = "./train.in"
    tin = AenetLux.read_train_in(inputfilename)
    show(tin)
    AenetLux.io_trainingset_information()

    list_structures_energy, list_structures_forces, list_removed, max_nnb, tin = AenetLux.read_list_structures(tin)
    for s in tin.setup_params.sfval_cov
        display(s)
    end

    N_removed = length(list_removed)
    N_struc_E = length(list_structures_energy)
    N_struc_F = length(list_structures_forces)
    if tin.verbose
        AenetLux.io_trainingset_information_done(tin, tin.trainset_params, N_struc_E, N_struc_F, N_removed)
    end

    #display(tin.setup_params.sfval_cov)
end

@testset "AenetLux.jl" begin
    # Write your tests here.
    test()
end
