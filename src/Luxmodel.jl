struct BPChain{L<:NamedTuple} <: Lux.AbstractLuxWrapperLayer{:layers}
    layers::L
end

function get_activation(act::String)
    if act == "linear"
        return x -> x  # Identity function
    elseif act == "tanh"
        return Lux.tanh
    elseif act == "sigmoid"
        return Lux.sigmoid
    else
        error("Unsupported activation function: $act")
    end
end

function BPChain(atomkinds, models)
    @assert length(atomkinds) == length(models) "Num. of atomic kinds might be wrong!"
    keys = Tuple(Symbol.(atomkinds))
    nt = NamedTuple{keys}(Tuple(models))
    return BPChain(nt)
end

function make_model(input_size, hidden_size, activations)
    display(input_size)
    display(hidden_size)
    display(activations)
    layers = []
    push!(layers, Dense(input_size, hidden_size[1]))
    for j in 1:length(hidden_size)-1
        push!(layers, Dense(hidden_size[j], hidden_size[j+1], get_activation(activations[j])))
    end
    push!(layers, Dense(hidden_size[end], 1))
    return Chain(layers...)
end

function BPChain(input_size, hidden_size, species, activations)
    nums = length(species)
    models = []
    for i = 1:nums
        push!(models, make_model(input_size[i], hidden_size[i], activations[i]))
    end
    return BPChain(species, models)
end