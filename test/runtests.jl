using AenetLux
using Test

function test()
    inputfilename = "./train.in"
    tin = AenetLux.read_train_in(inputfilename)
    show(tin)
    list_structures_energy, list_structures_forces, list_removed, max_nnb, tin = AenetLux.read_list_structures(tin)

end

@testset "AenetLux.jl" begin
    # Write your tests here.
    test()
end
