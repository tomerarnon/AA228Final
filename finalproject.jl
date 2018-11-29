using POMDPs
using StaticArrays
using Distributions
using StatsBase
using Statistics
using LinearAlgebra


"""
TruckMaintenance
    sensor_dict::Dict{Symbol, Vector{<:Distribution}}


Below is how sensor_dict is arranged. Each entry in sensor_dict[:fault] is a
distribution of a sensor measurement in the :fault state. sensor_dict[:nofault]
is the corresponding distribution for the nofault case

    sensor_dict:                     # Dict
        :fault -                       # Vector
            [1]: Normal(μ, σ)          # any Distribution type
            [2]: Normal(μ, σ)
            [3]: Normal(μ, σ)
            ...

        :nofault -                     # Vector
            [1]: Normal(μ, σ)          # any Distribution type
            [2]: Normal(μ, σ)
            [3]: Normal(μ, σ)
            ...

"""
struct TruckMaintenance
    sensor_dict::Dict{Symbol, Vector{<:Distribution}} # consider parameterizing the type
end

# random Normal constructor
function TruckMaintenance(n_sensors::Int)

    sensor_dict = Dict{Symbol, Vector{Normal}}()
    sensor_dict[:fault]   = Vector{Normal}()
    sensor_dict[:nofault] = Vector{Normal}()
    for i in 1:n_sensors
        μ = 10*rand()
        σ = 10*rand()

        push!(sensor_dict[:nofault], Normal(μ, σ))

        ### make slightly different distributions for :fault state
        # just pick values of 2 for no reason
        μ += 2*(rand()-0.5)  # ± 1
        σ *= 2*rand()        # scale by 0-2
        push!(sensor_dict[:fault], Normal(μ, σ))
    end
    TruckMaintenance(sensor_dict)
end


function reward(p::TruckMaintenance, s::Symbol, a::Symbol)
    r = 0.0

    r -= (s == :fault)  ? 100 : 0
    r -= (a == :repair) ?  10 : 0

    return r
end

function generate_s(p::TruckMaintenance, s::Symbol, a::Symbol)
    a == :repair && return (:nofault)
    # ∴ a is :norepair —
    s == :fault  && return (:fault)
    # ∴ s is :nofault —
    reliability = exp(-1/10000) ## 1 needs to be "distance" and 10000 needs to be "characterisitic distance"

    if rand() > reliability
        return (:fault)
    end

    return (:nofault)
end

# assumes each sensor measurement is independent and does not depend on previous measurements
# (maybe they should be linked to previous state for smooth transitions?)
function generate_o(p::TruckMaintenance, s::Symbol, a::Symbol, sp::Symbol) # only sp matters
    distributions = p.sensor_dict[s]
    o_vec = zeros(Float64, length(distributions))
    for i in 1:length(o_vec)
        o_vec[i] = rand(distributions[i])
    end
    o_vec
end




#=
The other formulation method:

function O(p::TruckMaintenance, o_vec::Vector, s::Symbol, a::Symbol) # not necessary for obs

    total_prob = 1.0
    for (i, o) in enumerate(o_vec)
        total_prob *= pdf(p.sensor_dict[s][i], o)
    end
    return total_prob
end

# returns probabilty of entering :fault state
function T(p::TruckMaintenance, s::Symbol, a::Symbol, sp::Symbol)

    if a == :repair
        sp == :fault          &&  return 0.0
        sp == :nofault        &&  return 1.0
    elseif a == :norepair
        if s == :fault
            sp == :fault      &&  return 1.0
            sp == :nofault    &&  return 0.0
        elseif s == :nofault
            reliability = exp(-1/10000)
            sp == :nofault    && return reliability    # f(distance/θ)
            sp == :fault      && return 1-reliability
        end
    end
end

=#