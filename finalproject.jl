using POMDPs
using StaticArrays
using Distributions
using StatsBase
using Statistics
using LinearAlgebra
using Random
using ParticleFilters


"""
TruckState

    fault - true/false
    d     - distance since last repair
"""
struct TruckState
    fault::Bool
    d::Float64
end

struct TruckAction
    repair::Bool
    d::Float64
end


"""
TruckMaintenance

    λ::Float  # mean failure rate
    sensor_dict::Dict{Symbol, Vector{<:Distribution}}


Below is how sensor_dict is arranged. Each entry in sensor_dict[:fault] is a
distribution of a sensor measurement in the :fault state. sensor_dict[:nofault]
is the corresponding distribution for the nofault case

    sensor_dict:                     # Dict
        :fault -                       # Vector
            [1]: Normal(μ, σ)          # any Distribution type
            [2]: Normal(μ, σ)
            ...

        :nofault -                     # Vector
            [1]: Normal(μ, σ)          # any Distribution type
            [2]: Normal(μ, σ)
            ...
"""
struct TruckMaintenance <: POMDP{TruckState, TruckAction, Vector{Float64}}
    λ::Float64
    sensor_dict::Dict{Bool, Vector{<:Distribution}} # consider parameterizing the type
end

reliability(p::TruckMaintenance, s::TruckState) = exp(-s.d/ p.λ)
rand_distance(p::TruckMaintenance, rng::AbstractRNG  = Random.GLOBAL_RNG= GLOBAL_RNG) = 3000.0*rand(rng) + 5.0

# getindex(D::Dict, s::TruckState) = D[s.fault]

# random Normal constructor
function TruckMaintenance(n_sensors::Int; λ = 10_000)

    sensor_dict = Dict{Bool, Vector{Normal}}()
    sensor_dict[true]   = Vector{Normal}()
    sensor_dict[false] = Vector{Normal}()
    for i in 1:n_sensors
        μ = 10*rand()
        σ = 10*rand()

        push!(sensor_dict[false], Normal(μ, σ))

        ### make slightly different distributions for :fault state
        # just pick values of 2 for no reason
        μ += 2*(rand()-0.5)  # ± 1
        σ *= 2*rand()        # scale by 0-2
        push!(sensor_dict[true], Normal(μ, σ))
    end
    TruckMaintenance(λ, sensor_dict)
end


initial_state(p::TruckMaintenance) = TruckState(false, 0)
discount(::TruckMaintenance) = 0.9






function reward(p::TruckMaintenance, s::TruckState, a::TruckAction)
    r = 0.0

    r -= s.fault  ? 100 : 0
    r -= a.repair ?  10 : 0

    return r
end

function generate_s(p::TruckMaintenance, s::TruckState, a::TruckAction, rng::AbstractRNG = Random.GLOBAL_RNG)
    if a.repair
        return TruckState(false, 0)
    end

    if s.fault
        return TruckState(true, s.d+a.d)
    end
    # ∴ s is :nofault —
    if rand() > reliability(p, s)
        return TruckState(true, s.d+a.d)
    else
        return TruckState(false, s.d+a.d)
    end
end

# assumes each sensor measurement is independent and does not depend on previous measurements
# (maybe they should be linked to previous state for smooth transitions?)
# function generate_o(p::TruckMaintenance, s::TruckState, a::Bool, sp::TruckState) # only sp matters
#     o_vec = zeros(Float64, length(p.sensor_dict[sp]))
#     for i in 1:length(o_vec)
#         o_vec[i] = rand(p.sensor_dict[sp][i])
#     end
#     o_vec
# end

scl(d::Normal, x) = x.*d.σ .+ d.μ
lerp(a, b, x) = a + x*(b-a)

# TYPE PIRACY
Distributions.pdf(Nvec::Vector{<:Normal}, x) = prod(pdf.(Nvec, x))

# assumes each sensor measurement is independent and does not depend on previous measurements
function generate_o(p::TruckMaintenance, s::TruckState, a::TruckAction, sp::TruckState, rng::AbstractRNG = Random.GLOBAL_RNG) # only sp matters
    fault_dists   = p.sensor_dict[true]
    nofault_dists = p.sensor_dict[false]

    o_vec = zeros(Float64, length(fault_dists))
    if sp.fault
        for i in 1:length(o_vec)
            o_vec[i] = rand(p.sensor_dict[sp][i])
        end
        return o_vec
    end

    for i in 1:length(o_vec)
        x = randn(rng)

        fault_obs   = scl(fault_dists[i], x)
        nofault_obs = scl(nofault_dists[i], x)

        o_vec[i] = lerp(fault_obs, nofault_obs, 0.9*reliability(p, sp)) # when reliability is close to 1, the value is close to nofault but never reaches it
    end
    o_vec
end

function generate_sr(p::TruckMaintenance, s::TruckState, a::TruckAction, rng::AbstractRNG = Random.GLOBAL_RNG)
    sp = generate_s(p, s, a, rng)
    r = reward(p, sp, a)

    return sp, r
end


# The other formulation method:

function observation(p::TruckMaintenance, sp::TruckState)

    return p.sensor_dict[sp]
end

# returns probabilty of entering :fault state
function transition(p::TruckMaintenance, s::TruckState, a::TruckAction)
    a.repair && return BoolDistribution(0.0)
    # elseif:
    s.fault && return BoolDistribution(1.0)
    # else:
    return BoolDistribution(1-reliability(p, s))
end




using MCTS
using BeliefUpdaters
using POMDPModelTools


p = TruckMaintenance(3)

n_particles = 100
b = ParticleCollection([initial_state(p) for i in 1:n_particles])
belief_updater = DiscreteUpdater(p)

s = initial_state(p)
a = TruckAction(false, 100)

sp = generate_s(p, s, a)
o = generate_o(p, s, a, sp)

# update(belief_updater, b, a, o)










